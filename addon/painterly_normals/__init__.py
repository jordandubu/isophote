bl_info = {
    "name": "Painterly Normals",
    "author": "La Ire Deu",
    "version": (0, 2, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Painterly > Bake & Paint",
    "description": "Bake normal map from texture, restyle it into painterly brush strokes with the local isophote engine (no API key), apply to material.",
    "category": "Paint",
}

import bpy
import os
from bpy.props import StringProperty, IntProperty, FloatProperty, BoolProperty, EnumProperty


# ------------------------------------------------------------- operator
class PN_OT_bake_and_paint(bpy.types.Operator):
    """Bake a normal map from the active material's image, restyle it with the local isophote engine, apply it"""

    bl_idname = "painterly.bake_and_paint"
    bl_label = "Paintify"
    bl_options = {'REGISTER', 'UNDO'}

    _timer_handle = None
    _thread = None

    def execute(self, context):
        obj = context.active_object
        mat = obj.active_material if obj else None
        if mat is None or not mat.use_nodes:
            self.report({'ERROR'}, "Active object needs a material with nodes.")
            return {'CANCELLED'}

        scene = context.scene
        s = scene.painterly_settings
        # default texture name = mesh name; user-editable afterwards
        if not s.tex_name:
            s.tex_name = obj.name or "Painterly"
        base = s.tex_name.strip() or "Painterly"

        import os
        size = int(s.tex_size)

        # true object-space mesh bake (Blender bake API)
        baked_path = os.path.join(bpy.app.tempdir, base + "_normal.png")
        err = _bake_mesh_normal(obj, size, baked_path)
        if err:
            self.report({'ERROR'}, f"Bake failed: {err}")
            s.status = "Bake failed"
            return {'CANCELLED'}
        self.report({'INFO'}, f"Normal baked: {baked_path}")

        # step 2: isophote restyle on a background thread; UI stays responsive
        import threading
        self._result = {"path": None, "error": None}
        self._thread = threading.Thread(
            target=_bridge_worker,
            args=(baked_path, s.renormalize, s.stroke_size, s, self._result, base),
            daemon=True,
        )
        self._thread.start()

        modal_mgr = bpy.context.window_manager
        self._timer_handle = modal_mgr.event_timer_add(0.4, window=context.window)
        modal_mgr.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'TIMER':
            if self._thread is not None and not self._thread.is_alive():
                wm = bpy.context.window_manager
                wm.event_timer_remove(self._timer_handle)
                self._timer_handle = None
                err = self._result.get("error")
                result_path = self._result.get("path")
                if err:
                    self.report({'ERROR'}, f"isophote failed: {err}")
                    return {'CANCELLED'}
                if not result_path:
                    self.report({'ERROR'}, "isophote finished but no result path (internal error).")
                    return {'CANCELLED'}
                # step 3: apply to material
                obj = context.active_object
                _apply_normal(obj.active_material, result_path)
                self.report({'INFO'}, f"Painterly normal applied: {result_path}")
                return {'FINISHED'}
        return {'PASS_THROUGH'}


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

        # push to the bridge (same endpoint the web app polls)
        import json, base64, urllib.request
        with open(baked_path, "rb") as f:
            png = base64.b64encode(f.read()).decode()
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8778/queue",
                data=json.dumps({"png": png}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=30)
        except Exception as e:
            self.report({'ERROR'}, f"Bridge offline (run: python3 serve.py in the isophote repo): {e}")
            return {'CANCELLED'}
        s.status = "Sent to isophote — paint it in the browser, then Apply"
        self.report({'INFO'}, "Normal sent to isophote (http://jordandubu.github.io/isophote/)")
        return {'FINISHED'}


class PN_OT_apply_from_isophote(bpy.types.Operator):
    """Fetch the painting from the isophote web app and apply it to the material"""

    bl_idname = "painterly.apply_from_isophote"
    bl_label = "Apply from Isophote"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import json, base64, urllib.request, os
        scene = context.scene
        s = scene.painterly_settings
        base = s.tex_name.strip() or "Painterly"

        try:
            req = urllib.request.Request("http://127.0.0.1:8778/result", data=b"{}", method="POST")
            d = json.loads(urllib.request.urlopen(req, timeout=30).read())
        except Exception as e:
            self.report({'ERROR'}, f"Bridge offline (run: python3 serve.py): {e}")
            return {'CANCELLED'}
        if d.get("kind") != "image" or not d.get("png"):
            self.report({'WARNING'}, "No painted result yet — paint it in isophote, click Send to Blender, then Apply.")
            return {'CANCELLED'}

        raw = base64.b64decode(d["png"])
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


def _get_or_create_work_image(w: int, h: int):
    """Create (or reuse) the addon's working texture: procedural blotches that
    give height structure. No user texture needed."""
    name = "Painterly_WorkTexture"
    img = bpy.data.images.get(name)
    if img is None:
        img = bpy.data.images.new(name, width=w, height=h, alpha=False)
    if img.size[0] != w or img.size[1] != h:
        img.scale(w, h)
        img.update()
    import numpy as np, math
    arr = np.zeros((h, w, 4), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    u = xx / w; v = yy / h
    # big soft blobs + medium noise = painterly height source
    blobs = (
        0.5 + 0.5 * np.sin(u * 6.0 + np.sin(v * 4.0) * 2.0) * np.cos(v * 5.0 + np.sin(u * 3.0))
    )
    noise = np.mod(np.sin(u * 57.3) * np.cos(v * 41.7) + np.sin((u + v) * 91.0), 1.0)
    height = np.clip(0.65 * blobs + 0.35 * noise, 0, 1)
    arr[..., 0] = height
    arr[..., 1] = height
    arr[..., 2] = height
    arr[..., 3] = 1.0
    img.pixels.foreach_set(arr[::-1].ravel())
    img.update()
    return img


def _bake_normal(img, out_path, stroke_size):
    """CPU bake: sample the image, gradients -> normal RGB. No GPU needed."""
    import numpy as np
    from bpy.types import Image as BImage

    src = np.empty(img.size[0] * img.size[1] * img.channels, dtype=np.float32)
    img.pixels.foreach_get(src)
    ch = img.channels
    sw, sh = img.size
    src = src.reshape(sh, sw, ch)[::-1]  # top-down for numpy
    height = src[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    k = max(int(stroke_size), 1)
    # wide smooth radius + narrow detail radius averaged = painterly base
    gx_l = np.roll(height, k, axis=1); gx_r = np.roll(height, -k, axis=1)
    gy_t = np.roll(height, k, axis=0); gy_b = np.roll(height, -k, axis=0)
    dx = gx_l - gx_r
    dy = gy_b - gy_t  # flip y for GL-style tangent normal


    norm = np.sqrt(dx * dx + dy * dy + 1.0)
    nx = dx / norm; ny = dy / norm; nz = 1.0 / norm
    # Bake in Blender Object space to match SHD_SPACE_BLENDER_OBJECT ("Blender Object").
    # Blender decodes: n = (2R-1, -2G+1, -2B+1). Inverse encoding below.
    out = np.stack([(nx * 0.5 + 0.5),
                    (0.5 - ny * 0.5),
                    (0.5 - nz * 0.5)], axis=-1)
    out = np.repeat(out, 1, axis=-1)  # RGB
    alpha = np.ones(out.shape[:2] + (1,), dtype=np.float32)
    out = np.concatenate([out, alpha], axis=-1)

    from bpy.types import Image as BImage
    baked = bpy.data.images.new("painterly_normal_baked", width=sw, height=sh, alpha=True)
    baked.pixels.foreach_set(out[::-1].ravel())
    baked.filepath_raw = out_path
    baked.file_format = 'PNG'
    baked.save()
    return out_path


def _bridge_worker(baked_path, renormalize, stroke_size, status=None, result=None, base_name="painterly"):
    """Threaded: paint the baked normal with the local isophote engine.

    Talks to the isophote web app's Blender bridge (blender_bridge.py, started
    by serve.py on port 8778). No API key, no upload — everything runs on this
    machine. Replaces the old OpenAI images-API worker."""
    import json, base64, urllib.request, os

    PORT_DEFAULT = 8778

    def step(msg: str) -> None:
        if status is not None:
            status.status = msg

    try:
        with open(baked_path, "rb") as f:
            png = f.read()
        img_b64 = base64.b64encode(png).decode()

        step("Step 2/3 — sending normal map to isophote")
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT_DEFAULT}/paint",
            data=json.dumps({
                "image_png": img_b64,
                "params": {
                    # chunky strokes for normal-map relief; scale knob follows
                    # the addon's Stroke Size slider
                    "brush_len": 16 * int(stroke_size),
                    "brush_w": 2 + int(stroke_size),
                    "layers": 3,
                    "relief": 0.0,          # normal maps: keep flat
                    "base_radius": 6,
                    "flow_smooth": 8,
                    "seed": 42,
                },
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        step("Step 3/3 — isophote painting (~1-3 s)")
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
        if data.get("error"):
            raise RuntimeError(data["error"])
        raw = base64.b64decode(data["png"])

        ai_path = os.path.join(os.path.dirname(baked_path), base_name + "_ai.png")
        with open(ai_path, "wb") as f:
            f.write(raw)
        final = ai_path
        if renormalize:
            step("Renormalizing")
            _renormalize_file(ai_path, result, base_name)
            final = result["path"]
        step(f"Done — painting took {data.get('secs', '?')} s")

        result["path"] = final
        result["error"] = None
    except Exception as e:
        result["error"] = str(e) + "  (is isophote running? `python3 serve.py` starts the bridge on port 8778)"


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
            self.report({'ERROR'}, f"No bake found at {path}. Run Paintify first.")
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
            self.report({'ERROR'}, f"No painted result at {path}. Run Paintify first.")
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
    stroke_size: EnumProperty(
        name="Stroke Size",
        items=[
            ('1', "Tiny", "Fine strokes, maximum 1x kernel"),
            ('2', "Medium", "Default, balanced"),
            ('4', "Big", "Broad chunky strokes"),
            ('8', "Super Big", "Maximum broad strokes"),
        ],
        default='2',
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
        layout.operator(PN_OT_send_to_isophote.bl_idname, text="Send to Isophote")
        layout.operator(PN_OT_apply_from_isophote.bl_idname, text="Apply from Isophote")
        layout.operator(PN_OT_bake_and_paint.bl_idname, text="Paintify (direct)")
        row = layout.row(align=True)
        row.operator(PN_OT_preview_bake.bl_idname, text="👁 Baked")
        row.operator(PN_OT_preview_result.bl_idname, text="👁 Painted")
        if s.status:
            layout.label(text=s.status, icon='TIME')

        layout.prop(s, "tex_name", text="Name")
        layout.prop(s, "tex_size")


class PN_PT_style(bpy.types.Panel):
    bl_label = "Style"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Painterly"
    bl_parent_id = "PN_PT_panel"
    bl_order = 1
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        s = context.scene.painterly_settings
        layout.prop(s, "stroke_size")
        layout.prop(s, "renormalize")
        layout.label(text="Engine: local isophote (port 8778)")


class PN_PT_options(bpy.types.Panel):
    bl_label = "Misc"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Painterly"
    bl_parent_id = "PN_PT_panel"
    bl_order = 2
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        s = context.scene.painterly_settings
        layout.prop(s, "renormalize")


classes = (
    PN_OT_preview_bake,
    PN_OT_preview_result,
    PN_scene_settings,
    PN_OT_bake_and_paint,
    PN_OT_send_to_isophote,
    PN_OT_apply_from_isophote,
    PN_PT_panel,
    PN_PT_style,
    PN_PT_options,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.painterly_settings = bpy.props.PointerProperty(type=PN_scene_settings)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.painterly_settings
