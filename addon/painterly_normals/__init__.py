bl_info = {
    "name": "Painterly Normals",
    "author": "La Ire Deu",
    "version": (0, 3, 2),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Painterly",
    "description": "Bake normal map from texture, restyle it into painterly brush strokes with isophote in your browser, apply to material.",
    "category": "Paint",
}

import bpy
import os
import json
import base64
import threading
import http.server
import webbrowser
from bpy.props import StringProperty, IntProperty, FloatProperty, BoolProperty, EnumProperty

AUTO_NAME = {}   # tracks the last auto-filled Name so user edits are kept


def _open_url(url):
    try:
        webbrowser.open(url)
    except Exception:
        pass


# ------------------------------------------------------------- operator
class PN_OT_send_to_isophote(bpy.types.Operator):
    """Bake the normal map and send it to the isophote web app for painting"""

    bl_idname = "painterly.send_to_isophote"
    bl_label = "Send to Isophote"
    bl_options = {'REGISTER'}

    def execute(self, context):
        obj = context.active_object
        mat = obj.active_material if obj else None
        if mat is None or not mat.use_nodes:
            self.report({'ERROR'}, "Active object needs a material with nodes.")
            return {'CANCELLED'}

        scene = context.scene
        s = scene.painterly_settings
        if not s.tex_name:
            s.tex_name = obj.name or "Painterly"
        base = s.tex_name.strip() or "Painterly"

        import os
        size = int(s.tex_size)
        baked_path = os.path.join(bpy.app.tempdir, base + "_normal.png")
        err = _bake_mesh_normal(obj, size, baked_path)
        if err:
            self.report({'ERROR'}, f"Bake failed: {err}")
            s.status = "Bake failed"
            return {'CANCELLED'}

        # queue it for the web app (embedded relay, in-process)
        with open(baked_path, "rb") as f:
            _bridge_state["pending_png"] = base64.b64encode(f.read()).decode()
        _bridge_state["result_png"] = None
        _open_url("https://jordandubu.github.io/isophote/")
        s.status = "Sent to isophote — paint it in the browser, then Apply"
        self.report({'INFO'}, "Normal sent to isophote — paint it there, click Send to Blender, then Apply")
        return {'FINISHED'}


class PN_OT_apply_from_isophote(bpy.types.Operator):
    """Fetch the painting from the isophote web app and apply it to the material"""

    bl_idname = "painterly.apply_from_isophote"
    bl_label = "Apply from Isophote"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        s = scene.painterly_settings
        base = s.tex_name.strip() or "Painterly"

        png_b64 = _bridge_state.get("result_png")
        if not png_b64:
            self.report({'WARNING'}, "No painted result yet — paint it in isophote, click Send to Blender, then Apply.")
            return {'CANCELLED'}
        _bridge_state["result_png"] = None

        raw = base64.b64decode(png_b64)
        result_path = os.path.join(bpy.app.tempdir, base + "_ai.png")  # same downstream name
        with open(result_path, "wb") as f:
            f.write(raw)

        final = result_path
        if s.renormalize:
            res = {"path": None, "error": None}
            _renormalize_file(result_path, res, base)
            if res.get("error"):
                self.report({'ERROR'}, f"Renormalize failed: {res['error']}")
                return {'CANCELLED'}
            final = res["path"]

        obj = context.active_object
        _apply_normal(obj.active_material, final)
        s.status = "Applied"
        self.report({'INFO'}, f"Painterly normal applied: {final}")
        return {'FINISHED'}


# ------------------------------------------------------------- helpers
def _find_image(mat):
    if not mat or not mat.use_nodes:
        return None
    for n in mat.node_tree.nodes:
        if n.type == 'TEX_IMAGE' and n.image:
            return n.image
    return None


def _bake_mesh_normal(obj, size: int, out_path: str):
    """True object-space normal bake via Blender's bake system.
    Space=Object, swizzle +X +Y +Z. Returns error string or None."""
    import os
    try:
        if obj.type != 'MESH':
            return "active object is not a mesh"
        if not obj.data.uv_layers:
            obj.data.uv_layers.new(name="UVMap")

        img = bpy.data.images.get("Painterly_BakedNormal")
        if img is None or img.size[0] != size or img.size[1] != size:
            if img:
                bpy.data.images.remove(img)
            img = bpy.data.images.new("Painterly_BakedNormal", size, size)
        img.generated_color = (0.5, 0.5, 1.0, 1.0)  # flat normal default
        img.update()

        # temp bake material; bake writes into the ACTIVE image node
        mat = bpy.data.materials.new("Painterly_BakeMat")
        mat.use_nodes = True
        nt = mat.node_tree
        for n in list(nt.nodes):
            if n.type != 'OUTPUT_MATERIAL':
                nt.nodes.remove(n)
        tex = nt.nodes.new('ShaderNodeTexImage')
        tex.image = img
        nt.nodes.active = tex
        tex.select = True

        saved_slots = [(ms.link, ms.material) for ms in obj.material_slots]
        obj.data.materials.clear()
        obj.data.materials.append(mat)

        scene = bpy.context.scene
        saved_engine = scene.render.engine
        scene.render.engine = 'CYCLES'
        saved_device = scene.cycles.device
        scene.cycles.device = 'CPU'
        saved_samples = scene.cycles.samples
        scene.cycles.samples = 1
        saved_mode = None
        if bpy.context.mode != 'OBJECT':
            saved_mode = bpy.context.mode
            bpy.ops.object.mode_set(mode='OBJECT')
        saved_active = bpy.context.view_layer.objects.active
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        b = scene.render.bake
        saved_bake = (b.normal_space, b.normal_r, b.normal_g, b.normal_b, b.margin, b.use_clear, b.use_selected_to_active)
        b.normal_space = 'OBJECT'
        b.normal_r = 'POS_X'
        b.normal_g = 'POS_Y'
        b.normal_b = 'POS_Z'
        b.margin = 8
        b.use_selected_to_active = False

        try:
            bpy.ops.object.bake(type='NORMAL', use_clear=True)
        finally:
            scene.render.engine = saved_engine
            scene.cycles.device = saved_device
            scene.cycles.samples = saved_samples
            (b.normal_space, b.normal_r, b.normal_g, b.normal_b, b.margin, b.use_clear, b.use_selected_to_active) = saved_bake
            obj.data.materials.clear()
            seen = set()
            for link, m in saved_slots:
                if m is None or m.name in seen:
                    continue
                seen.add(m.name)
                obj.data.materials.append(m)
            if saved_active:
                bpy.context.view_layer.objects.active = saved_active
            if saved_mode:
                bpy.ops.object.mode_set(mode=saved_mode)

        # save baked PNG in top-down orientation for the engine (flip once here)
        if img.has_data is False and img.size[0] == 0:
            return "bake produced empty image"
        import numpy as _np
        w, h = img.size
        ch = img.channels
        buf = _np.empty(w * h * ch, dtype=_np.float32)
        img.pixels.foreach_get(buf)
        buf = buf.reshape(h, w, ch)[::-1]  # flip: Blender bottom-up -> top-down
        flat = bpy.data.images.new("Painterly_FlipTmp", w, h, alpha=True)
        flat.pixels.foreach_set(buf[::-1].ravel())  # back to Blender order in datablock
        flat.filepath_raw = out_path
        flat.file_format = 'PNG'
        flat.save()
        bpy.data.images.remove(flat)
        return None
    except Exception as e:
        return str(e)



def _bake_mesh_normal(obj, size: int, out_path: str):
    """True object-space normal bake via Blender's bake system.
    Space=Object, swizzle +X +Y +Z. Returns error string or None."""
    import os
    try:
        if obj.type != 'MESH':
            return "active object is not a mesh"
        if not obj.data.uv_layers:
            obj.data.uv_layers.new(name="UVMap")

        img = bpy.data.images.get("Painterly_BakedNormal")
        if img is None or img.size[0] != size or img.size[1] != size:
            if img:
                bpy.data.images.remove(img)
            img = bpy.data.images.new("Painterly_BakedNormal", size, size)
        img.generated_color = (0.5, 0.5, 1.0, 1.0)  # flat normal default
        img.update()

        # temp bake material; bake writes into the ACTIVE image node
        mat = bpy.data.materials.new("Painterly_BakeMat")
        mat.use_nodes = True
        nt = mat.node_tree
        for n in list(nt.nodes):
            if n.type != 'OUTPUT_MATERIAL':
                nt.nodes.remove(n)
        tex = nt.nodes.new('ShaderNodeTexImage')
        tex.image = img
        nt.nodes.active = tex
        tex.select = True

        saved_slots = [(ms.link, ms.material) for ms in obj.material_slots]
        obj.data.materials.clear()
        obj.data.materials.append(mat)

        scene = bpy.context.scene
        saved_engine = scene.render.engine
        scene.render.engine = 'CYCLES'
        saved_device = scene.cycles.device
        scene.cycles.device = 'CPU'
        saved_samples = scene.cycles.samples
        scene.cycles.samples = 1
        saved_mode = None
        if bpy.context.mode != 'OBJECT':
            saved_mode = bpy.context.mode
            bpy.ops.object.mode_set(mode='OBJECT')
        saved_active = bpy.context.view_layer.objects.active
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        b = scene.render.bake
        saved_bake = (b.normal_space, b.normal_r, b.normal_g, b.normal_b, b.margin, b.use_clear, b.use_selected_to_active)
        b.normal_space = 'OBJECT'
        b.normal_r = 'POS_X'
        b.normal_g = 'POS_Y'
        b.normal_b = 'POS_Z'
        b.margin = 8
        b.use_selected_to_active = False

        try:
            bpy.ops.object.bake(type='NORMAL', use_clear=True)
        finally:
            scene.render.engine = saved_engine
            scene.cycles.device = saved_device
            scene.cycles.samples = saved_samples
            (b.normal_space, b.normal_r, b.normal_g, b.normal_b, b.margin, b.use_clear, b.use_selected_to_active) = saved_bake
            obj.data.materials.clear()
            seen = set()
            for link, m in saved_slots:
                if m is None or m.name in seen:
                    continue
                seen.add(m.name)
                obj.data.materials.append(m)
            if saved_active:
                bpy.context.view_layer.objects.active = saved_active
            if saved_mode:
                bpy.ops.object.mode_set(mode=saved_mode)

        # save baked PNG in top-down orientation for the engine (flip once here)
        if img.has_data is False and img.size[0] == 0:
            return "bake produced empty image"
        import numpy as _np
        w, h = img.size
        ch = img.channels
        buf = _np.empty(w * h * ch, dtype=_np.float32)
        img.pixels.foreach_get(buf)
        buf = buf.reshape(h, w, ch)[::-1]  # flip: Blender bottom-up -> top-down
        flat = bpy.data.images.new("Painterly_FlipTmp", w, h, alpha=True)
        flat.pixels.foreach_set(buf[::-1].ravel())  # back to Blender order in datablock
        flat.filepath_raw = out_path
        flat.file_format = 'PNG'
        flat.save()
        bpy.data.images.remove(flat)
        return None
    except Exception as e:
        return str(e)



def _renormalize_file(path, result, base_name="painterly"):
    """Reproject painted pixels to valid normal vectors using numpy, no Image API."""
    import numpy as np, os
    img = bpy.data.images.load(path)
    arr = np.empty(img.size[0] * img.size[1] * img.channels, dtype=np.float32)
    img.pixels.foreach_get(arr)
    arr = arr.reshape(img.size[1], img.size[0], img.channels)
    rgb = arr[..., :3] * 2.0 - 1.0
    length = np.linalg.norm(rgb, axis=-1, keepdims=True)
    length[length < 0.001] = 1.0
    rgb = rgb / length
    rgb[..., 2] = np.abs(rgb[..., 2])
    arr[..., :3] = rgb * 0.5 + 0.5
    final_path = os.path.join(os.path.dirname(path), base_name + "_final.png")
    out_img = bpy.data.images.new("painterly_final", width=img.size[0], height=img.size[1], alpha=True)
    # keep Blender's internal orientation (bottom-up); no manual flip:
    # load() flipped once on read, save() flips once on write => net = source orientation
    out_img.pixels.foreach_set(arr.ravel())
    out_img.filepath_raw = final_path
    out_img.file_format = 'PNG'
    out_img.save()
    bpy.data.images.remove(img)
    bpy.data.images.remove(out_img)
    result["path"] = final_path


def _apply_normal(mat, path):
    """Wire (or update) the Painterly normal chain. Always replaces the image
    in the existing PainterlyNormalsTex node once baking is done."""
    if not mat or not mat.use_nodes:
        return
    nt = mat.node_tree
    dname = "Painterly_" + os.path.basename(path)
    prev = bpy.data.images.get(dname)
    if prev:
        bpy.data.images.remove(prev)
    img = bpy.data.images.load(path, check_existing=False)
    img.name = dname
    img.colorspace_settings.name = 'Non-Color'  # normal map, not color data

    # find Principled BSDF
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if bsdf is None:
        bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')

    # existing painterly nodes or create fresh
    tex = next((n for n in nt.nodes if n.type == 'TEX_IMAGE' and n.name == "PainterlyNormalsTex"), None)
    nmap = next((n for n in nt.nodes if n.type == 'NORMAL_MAP' and n.space == 'OBJECT'), None)
    if tex is None:
        tex = nt.nodes.new('ShaderNodeTexImage')
        tex.name = "PainterlyNormalsTex"
    if nmap is None:
        nmap = nt.nodes.new('ShaderNodeNormalMap')
        nmap.space = 'OBJECT'

    # ALWAYS replace the image
    tex.image = img
    tex.location = (bsdf.location.x - 400, bsdf.location.y + 220)
    nmap.location = (bsdf.location.x - 200, bsdf.location.y + 200)

    # ensure wiring exists (idempotent)
    nt.links.new(tex.outputs['Color'], nmap.inputs['Color'])
    nt.links.new(nmap.outputs['Normal'], bsdf.inputs['Normal'])


# ------------------------------------------------------------- preview
class PN_OT_preview_bake(bpy.types.Operator):
    """Show the baked normal map"""
    bl_idname = "painterly.preview_bake"
    bl_label = ""
    bl_options = {'REGISTER'}

    def execute(self, context):
        base = context.scene.painterly_settings.tex_name.strip() or "Painterly"
        path = os.path.join(bpy.app.tempdir, base + "_normal.png")
        if not os.path.isfile(path):
            self.report({'ERROR'}, f"No bake found at {path}. Run Send to Isophote first.")
            return {'CANCELLED'}
        return _open_view(path, "Baked Normal")


class PN_OT_preview_result(bpy.types.Operator):
    """Show the AI-painted normal"""
    bl_idname = "painterly.preview_result"
    bl_label = ""
    bl_options = {'REGISTER'}

    def execute(self, context):
        base = context.scene.painterly_settings.tex_name.strip() or "Painterly"
        path = os.path.join(bpy.app.tempdir, base + "_ai.png")
        if not os.path.isfile(path):
            self.report({'ERROR'}, f"No painted result at {path}. Run Send to Isophote first.")
            return {'CANCELLED'}
        return _open_view(path, "Painted")


def _open_view(path, title):
    if not os.path.isfile(path):
        return {'CANCELLED'}
    name = "PainterlyPreview_" + os.path.basename(path)
    old = bpy.data.images.get(name)
    if old:
        bpy.data.images.remove(old)
    img = bpy.data.images.load(path, check_existing=False)
    img.name = name
    try:
        # new window: duplicate current area into image editor
        for area in bpy.context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                break
        else:
            # change most area to image editor temporarily? simplest: open as new window
            pass
    except Exception:
        pass
    # Simplest robust approach: reuse an IMAGE_EDITOR area if present, else convert one
    area = next((a for a in bpy.context.screen.areas if a.type == 'IMAGE_EDITOR'), None)
    if area is None:
        # convert a Spreadsheet area if exists else any
        src = next((a for a in bpy.context.screen.areas if a.type not in ('VIEW_3D',)), None) or bpy.context.screen.areas[-1]
        with bpy.context.temp_override(area=src):
            bpy.ops.screen.area_type(type='IMAGE_EDITOR')
        area = src
    space = area.spaces.active
    space.image = img
    space.ui_mode = 'VIEW'
    return {'FINISHED'}



# ------------------------------------------------------------- scene props
class PN_scene_settings(bpy.types.PropertyGroup):
    tex_name: StringProperty(
        name="Texture Name",
        description="Base name for generated textures (final: <name>_normal.png, <name>_ai.png, <name>_final.png)",
        default="",
        maxlen=120,
    )
    tex_size: EnumProperty(
        name="Texture Size",
        items=[
            ('1024', "1K", "1024 x 1024"),
            ('2048', "2K", "2048 x 2048"),
            ('4096', "4K", "4096 x 4096"),
        ],
        default='2048',
    )
    renormalize: BoolProperty(
        name="Renormalize Painted Output", default=True,
        description="Re-project AI pixels to valid normal vectors",
    )
    status: StringProperty(
        name="Status", default="",
    )


# ------------------------------------------------------------- UI
class PN_PT_panel(bpy.types.Panel):
    bl_label = "Painterly Normals"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Painterly"

    bl_order = 0

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        mat = obj.active_material if obj else None
        layout.label(text="Material: " + (mat.name if mat else "none"))

        s = context.scene.painterly_settings
        # keep Name synced to the mesh name until the user types their own
        last = AUTO_NAME.get("last")
        if obj and (s.tex_name == last or not s.tex_name):
            s.tex_name = obj.name or "Painterly"
        AUTO_NAME["last"] = s.tex_name
        layout.operator(PN_OT_send_to_isophote.bl_idname, text="Send to Isophote")
        layout.operator(PN_OT_apply_from_isophote.bl_idname, text="Apply from Isophote")
        row = layout.row(align=True)
        row.operator(PN_OT_preview_bake.bl_idname, text="👁 Baked")
        row.operator(PN_OT_preview_result.bl_idname, text="👁 Painted")
        if s.status:
            layout.label(text=s.status, icon='TIME')

        layout.prop(s, "tex_name", text="Name")
        layout.prop(s, "tex_size")


class PN_PT_misc(bpy.types.Panel):
    bl_label = "Misc"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Painterly"
    bl_parent_id = "PN_PT_panel"
    bl_order = 1
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        s = context.scene.painterly_settings
        layout.prop(s, "renormalize")


classes = (
    PN_OT_preview_bake,
    PN_OT_preview_result,
    PN_scene_settings,
    PN_OT_send_to_isophote,
    PN_OT_apply_from_isophote,
    PN_PT_panel,
    PN_PT_misc,
)


# ------------------------------------------------------------- embedded bridge
# Tiny HTTP relay living inside the addon (no numpy/cv2 needed — painting
# happens in the browser). The web app at
# https://jordandubu.github.io/isophote/ polls it on 127.0.0.1:8778.
BRIDGE_PORT = 8778

_bridge_state = {"pending_png": None, "result_png": None}
_bridge_server = None


class _BridgeHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _reply(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._reply({"error": "bad json"}, 400)

        if self.path == "/queue":            # Blender -> app
            _bridge_state["pending_png"] = req.get("png")
            _bridge_state["result_png"] = None
            return self._reply({"ok": True})

        if self.path == "/poll":             # app pulls from Blender
            png = _bridge_state["pending_png"]
            _bridge_state["pending_png"] = None
            return self._reply({"kind": "image", "bgr_png": png} if png
                               else {"kind": "none"})

        if self.path == "/send_to_blender":  # app result -> Blender
            _bridge_state["result_png"] = req.get("png")
            return self._reply({"ok": True})

        if self.path == "/result":           # addon pulls the app's painting
            png = _bridge_state["result_png"]
            _bridge_state["result_png"] = None
            return self._reply({"kind": "image", "png": png} if png
                               else {"kind": "none"})

        return self._reply({"error": "unknown path"}, 404)


def _bridge_start():
    global _bridge_server
    if _bridge_server is not None:
        return True
    try:
        _bridge_server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", BRIDGE_PORT), _BridgeHandler)
        t = threading.Thread(target=_bridge_server.serve_forever, daemon=True)
        t.start()
        return True
    except OSError as e:
        print(f"[painterly_normals] bridge start failed (port {BRIDGE_PORT}): {e}")
        _bridge_server = None
        return False


def _bridge_stop():
    global _bridge_server
    if _bridge_server is not None:
        _bridge_server.shutdown()
        _bridge_server = None


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.painterly_settings = bpy.props.PointerProperty(type=PN_scene_settings)
    _bridge_start()


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.painterly_settings
    _bridge_stop()
