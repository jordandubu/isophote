bl_info = {
    "name": "Painterly Normals",
    "author": "La Ire Deu",
    "version": (0, 5, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Painterly",
    "description": "Bake normal map from texture, restyle it into painterly brush strokes with isophote in your browser, apply to material.",
    "category": "Paint",
}

import bpy
import os
import json
import base64
import shutil
import threading
import http.server
import webbrowser
from bpy.props import StringProperty, IntProperty, FloatProperty, BoolProperty, EnumProperty

AUTO_NAME = {}   # tracks the last auto-filled Name so user edits are kept

KIND_LABELS = {
    'normal': 'normal map', 'color': 'base color', 'rough': 'roughness',
    'metal': 'metallic', 'emission': 'emission',
}


def _open_url(url):
    try:
        webbrowser.open(url)
    except Exception:
        pass


# ------------------------------------------------------------- operator
# ------------------------------------------------------------- animated apply
class PN_OT_apply_animated_from_isophote(bpy.types.Operator):
    """Fetch the animated frames from the isophote web app and wire them
    into the material as a playing image sequence"""

    bl_idname = "painterly.apply_animated_from_isophote"
    bl_label = "Apply animated from Isophote"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        s = scene.painterly_settings
        base = s.tex_name.strip() or "Painterly"

        anim = _bridge_state.get("result_anim")
        kinds = _bridge_state.get("result_anim_kinds")
        if not anim:
            self.report({'WARNING'},
                        "No animated frames — use 'Send animated' in isophote first.")
            return {'CANCELLED'}
        if kinds:
            return self._apply_multi(context, s, base, kinds, anim)
        _bridge_state["result_anim"] = None

        tmp = bpy.app.tempdir
        norm_dir = os.path.join(tmp, base + "_anim")
        os.makedirs(norm_dir, exist_ok=True)
        obj = context.active_object
        mat = obj.active_material if obj else None
        if mat is None or not mat.use_nodes:
            self.report({'ERROR'}, "Active object needs a material with nodes.")
            return {'CANCELLED'}

        paths = []
        try:
            for i, png_b64 in enumerate(anim):
                raw = base64.b64decode(png_b64)
                src = os.path.join(tmp, f"{base}_anim_{i:03d}.png")
                with open(src, "wb") as f:
                    f.write(raw)
                final = src
                if s.renormalize:
                    res = {"path": None, "error": None}
                    _renormalize_file(src, res, f"{base}_anim_{i:03d}")
                    if res.get("error"):
                        self.report({'ERROR'}, f"Renormalize failed: {res['error']}")
                        return {'CANCELLED'}
                    final = res["path"]
                # move into the anim folder with a stable sequence name
                seq = os.path.join(norm_dir, f"{base}_anim_{i:03d}.png")
                if final != seq:
                    shutil.move(final, seq)
                paths.append(seq)
        except Exception as e:
            self.report({'ERROR'}, f"Frame write failed: {e}")
            return {'CANCELLED'}

        # image sequence: Blender plays frames <base>_anim_###.png
        img = bpy.data.images.load(paths[0], check_existing=False)
        img.source = 'SEQUENCE'
        img.colorspace_settings.name = 'Non-Color'
        # sequence expects frame files named with the END number; the loaded
        # first frame sets the pattern
        _apply_animated_map(mat, img, start_frame=1, duration=len(paths), kind='normal')
        s.status = f"Applied animated ({len(paths)} frames)"
        self.report({'INFO'}, f"Animated painterly normal: {len(paths)} frames from frame 1")
        return {'FINISHED'}

    def _apply_multi(self, context, s, base, kinds, anim):
        """Coherent animated multi: anim[i] = [b64 per kind] (kinds order).
        Writes <base>_<kind>_anim/ sequences and wires every kind."""
        _bridge_state["result_anim"] = None
        _bridge_state["result_anim_kinds"] = None
        tmp = bpy.app.tempdir
        obj = context.active_object
        mat = obj.active_material if obj else None
        if mat is None or not mat.use_nodes:
            self.report({'ERROR'}, "Active object needs a material with nodes.")
            return {'CANCELLED'}
        applied = []
        try:
            for ki, kind in enumerate(kinds):
                kind_dir = os.path.join(tmp, f"{base}_{kind}_anim")
                os.makedirs(kind_dir, exist_ok=True)
                img = None
                for i, frame_kinds in enumerate(anim):
                    if ki >= len(frame_kinds):
                        self.report({'ERROR'},
                                    "Corrupted multi anim (frame kind mismatch).")
                        return {'CANCELLED'}
                    src = os.path.join(tmp, f"{base}_{kind}_anim_{i:03d}.png")
                    with open(src, "wb") as f:
                        f.write(base64.b64decode(frame_kinds[ki]))
                    final = src
                    if kind == 'normal' and s.renormalize:
                        res = {"path": None, "error": None}
                        _renormalize_file(src, res, f"{base}_{kind}_anim_{i:03d}")
                        if res.get("error"):
                            self.report({'ERROR'}, f"Renormalize failed: {res['error']}")
                            return {'CANCELLED'}
                        final = res["path"]
                    seq = os.path.join(kind_dir, f"{base}_{kind}_anim_{i:03d}.png")
                    if final != seq:
                        shutil.move(final, seq)
                    if i == 0:
                        img = bpy.data.images.load(seq, check_existing=False)
                img.source = 'SEQUENCE'
                _, _, colorspace = _paint_targets_for(kind)
                img.colorspace_settings.name = colorspace
                _apply_animated_map(mat, img, start_frame=1,
                                    duration=len(anim), kind=kind)
                applied.append(KIND_LABELS[kind])
        except Exception as e:
            self.report({'ERROR'}, f"Frame write failed: {e}")
            return {'CANCELLED'}
        s.status = f"Applied animated ({len(anim)} frames): " + ", ".join(applied)
        self.report({'INFO'},
                    f"Animated multi ({', '.join(applied)}): {len(anim)} frames from frame 1")
        return {'FINISHED'}


def _paint_targets_for(kind):
    """(socket, node_name, colorspace) for a kind; normal gets its chain."""
    if kind == 'normal':
        return ('Normal', 'PainterlyNormalsTex', 'Non-Color')
    return _PAINT_TARGETS[kind]


def _apply_animated_map(mat, img, start_frame=1, duration=10, kind='normal'):
    """Wire `img` (a SEQUENCE image) into the material for `kind`.
    normal: tex -> object-space NormalMap -> Normal. Others: tex -> socket."""
    if not mat or not mat.use_nodes:
        return
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if bsdf is None:
        bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')

    if kind == 'normal':
        tex = next((n for n in nt.nodes if n.type == 'TEX_IMAGE' and n.name == "PainterlyNormalsTex"), None)
        nmap = next((n for n in nt.nodes if n.type == 'NORMAL_MAP' and n.space == 'OBJECT'), None)
        if tex is None:
            tex = nt.nodes.new('ShaderNodeTexImage')
            tex.name = "PainterlyNormalsTex"
        if nmap is None:
            nmap = nt.nodes.new('ShaderNodeNormalMap')
            nmap.space = 'OBJECT'
        tex.image = img
        tex.image_user.frame_duration = duration
        tex.image_user.frame_start = start_frame
        tex.image_user.frame_offset = 0
        tex.image_user.use_cyclic = True
        tex.image_user.use_auto_refresh = True
        tex.location = (bsdf.location.x - 400, bsdf.location.y + 220)
        nmap.location = (bsdf.location.x - 200, bsdf.location.y + 200)
        nt.links.new(tex.outputs['Color'], nmap.inputs['Color'])
        nt.links.new(nmap.outputs['Normal'], bsdf.inputs['Normal'])
        return

    target_socket, node_name, _cs = _PAINT_TARGETS[kind]
    tex = next((n for n in nt.nodes if n.type == 'TEX_IMAGE' and n.name == node_name), None)
    if tex is None:
        tex = nt.nodes.new('ShaderNodeTexImage')
        tex.name = node_name
    tex.image = img          # replaces the single-frame image of the same node
    tex.image_user.frame_duration = duration
    tex.image_user.frame_start = start_frame
    tex.image_user.frame_offset = 0
    tex.image_user.use_cyclic = True
    tex.image_user.use_auto_refresh = True
    y = 220 - 120 * (_PAINT_ORDER[kind] + 1)
    tex.location = (bsdf.location.x - 400, bsdf.location.y + y)
    nt.links.new(tex.outputs['Color'], bsdf.inputs[target_socket])


KIND_SUFFIX = {'normal': '_normal.png', 'color': '_color.png', 'rough': '_rough.png',
               'metal': '_metal.png', 'emission': '_emission.png'}
KIND_PROP = {'color': 'bake_color', 'rough': 'bake_rough',
             'metal': 'bake_metallic', 'emission': 'bake_emission'}
# order used for kinds array on the wire (color first as replay reference)
KIND_ORDER = ['color', 'normal', 'rough', 'metal', 'emission']


def _selected_kinds(s):
    """Checked + enabled kinds in wire order ('normal' always included)."""
    return [k for k in KIND_ORDER if k == 'normal' or getattr(s, KIND_PROP[k])]


class PN_OT_send_to_isophote(bpy.types.Operator):
    """Bake the selected maps and send them to the isophote web app for painting"""

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
        kinds = _selected_kinds(s)
        baked = {}
        for kind in kinds:
            path = os.path.join(bpy.app.tempdir, base + KIND_SUFFIX[kind])
            err = _bake_map(obj, size, kind, path)
            if err:
                self.report({'ERROR'}, f"Bake failed ({kind}): {err}")
                s.status = f"Bake failed ({kind})"
                return {'CANCELLED'}
            with open(path, "rb") as f:
                baked[kind] = base64.b64encode(f.read()).decode()
        # replay reference: color if present, else first selected in wire order
        ordered = [k for k in KIND_ORDER if k in baked]

        # queue for the web app (embedded relay, in-process); 1 channel keeps
        # the legacy {png} shape for old pages
        _bridge_state["result_png"] = None
        _bridge_state["result_kinds"] = None
        _bridge_state["result_kind"] = None
        _bridge_state["result_anim"] = None
        _bridge_state["result_anim_kinds"] = None
        if len(ordered) == 1:
            kind = ordered[0]
            _bridge_state["pending_png"] = baked[kind]
            _bridge_state["pending_kind"] = kind
        else:
            _bridge_state["pending_png"] = None
            _bridge_state["pending_kind"] = None
            _bridge_state["pending_multi"] = {
                "kind": "multi",
                "kinds": ordered,
                "pngs": [baked[k] for k in ordered],
            }
        _bridge_state["pending_ack"] = False
        _open_url("https://jordandubu.github.io/isophote/")
        label = " + ".join(KIND_LABELS[k] for k in ordered)
        s.status = "Sent to isophote — paint it in the browser, then Apply"
        self.report({'INFO'}, f"{label} sent to isophote — paint, click Send to Blender, then Apply")
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

        kinds = _bridge_state.get("result_kinds")
        if kinds:
            pngs = _bridge_state.get("result_pngs") or []
            if len(pngs) != len(kinds):
                self.report({'ERROR'}, "Corrupted multi result (kinds/pngs mismatch).")
                return {'CANCELLED'}
            _bridge_state["result_kinds"] = None
            _bridge_state["result_pngs"] = None
            obj = context.active_object
            mat = obj.active_material if obj else None
            if mat is None or not mat.use_nodes:
                self.report({'ERROR'}, "Active object needs a material with nodes.")
                return {'CANCELLED'}
            applied = []
            for kind, png_b64 in zip(kinds, pngs):
                final = os.path.join(bpy.app.tempdir, f"{base}_{kind}_painted.png")
                with open(final, "wb") as f:
                    f.write(base64.b64decode(png_b64))
                if kind == 'normal':
                    final = self._normal_chain(context, s, base, final)
                    if final is None:
                        return {'CANCELLED'}
                else:
                    _apply_paint(mat, final, kind)
                applied.append(KIND_LABELS[kind])
            s.status = "Applied: " + ", ".join(applied)
            self.report({'INFO'}, f"Applied: {', '.join(applied)}")
            return {'FINISHED'}

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

    @staticmethod
    def _normal_chain(context, s, base, result_path):
        """Renormalize + wire the normal chain; returns final path or None."""
        final = result_path
        if s.renormalize:
            res = {"path": None, "error": None}
            _renormalize_file(result_path, res, base + "_normal")
            if res.get("error"):
                return None
            final = res["path"]
        obj = context.active_object
        _apply_normal(obj.active_material, final)
        return final


# ------------------------------------------------------------- helpers
def _find_image(mat):
    if not mat or not mat.use_nodes:
        return None
    for n in mat.node_tree.nodes:
        if n.type == 'TEX_IMAGE' and n.image:
            return n.image
    return None


# kind -> (bake_type, generated default color). normal keeps the classic
# object-space swizzle; the rest bake the corresponding Cycles pass.
_BAKE_TYPES = {
    'normal': 'NORMAL',
    'color': 'DIFFUSE',
    'rough': 'ROUGHNESS',
    'metal': 'METALLIC',
    'emission': 'EMISSION',
}


def _bake_map(obj, size: int, kind: str, out_path: str):
    """Bake one map via Blender's bake system. kind in _BAKE_TYPES;
    'normal' is object space with swizzle +X +Y +Z, 'color' is the unlit
    diffuse albedo (pass_filter COLOR). Returns error string or None."""
    import os
    try:
        bake_type = _BAKE_TYPES[kind]
        if obj.type != 'MESH':
            return "active object is not a mesh"
        if not obj.data.uv_layers:
            obj.data.uv_layers.new(name="UVMap")

        img_name = "Painterly_Baked" + kind.title()
        img = bpy.data.images.get(img_name)
        if img is None or img.size[0] != size or img.size[1] != size:
            if img:
                bpy.data.images.remove(img)
            img = bpy.data.images.new(img_name, size, size)
        img.generated_color = ((0.5, 0.5, 1.0, 1.0) if kind == 'normal'
                               else (0.0, 0.0, 0.0, 1.0))  # neutral default
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
        is_normal = kind == 'normal'
        saved_bake = (b.normal_space, b.normal_r, b.normal_g, b.normal_b,
                      b.margin, b.use_clear, b.use_selected_to_active)
        b.margin = 8
        b.use_selected_to_active = False
        if is_normal:
            b.normal_space = 'OBJECT'
            b.normal_r = 'POS_X'
            b.normal_g = 'POS_Y'
            b.normal_b = 'POS_Z'

        try:
            if is_normal:
                bpy.ops.object.bake(type='NORMAL', use_clear=True)
            elif bake_type == 'DIFFUSE':
                # pass_filter=COLOR: unlit albedo (direct color only)
                bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'},
                                    use_clear=True)
            else:
                bpy.ops.object.bake(type=bake_type, use_clear=True)
        finally:
            scene.render.engine = saved_engine
            scene.cycles.device = saved_device
            scene.cycles.samples = saved_samples
            (b.normal_space, b.normal_r, b.normal_g, b.normal_b, b.margin,
             b.use_clear, b.use_selected_to_active) = saved_bake
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


# non-normal kinds -> (socket, tex node name, colorspace). Roughness and
# Metallic are VALUE sockets; Blender converts a color input by the red
# channel, and the painted grayscale maps are r=g=b, so this is exact.
_PAINT_TARGETS = {
    'color':    ('Base Color', 'PainterlyColorTex', 'sRGB'),
    'rough':    ('Roughness', 'PainterlyRoughTex', 'Non-Color'),
    'metal':    ('Metallic', 'PainterlyMetalTex', 'Non-Color'),
    'emission': ('Emission Color', 'PainterlyEmissionTex', 'sRGB'),
}
_PAINT_ORDER = {'color': 0, 'rough': 1, 'metal': 2, 'emission': 3}


def _apply_paint(mat, path, kind):
    """Wire (or update) a painted non-normal map into the Principled BSDF.
    tex node name is stable per kind ('PainterlyColorTex', ...) so repeated
    applies replace the image instead of stacking nodes."""
    target_socket, node_name, colorspace = _PAINT_TARGETS[kind]
    nt = mat.node_tree
    dname = "Painterly_" + os.path.basename(path)
    prev = bpy.data.images.get(dname)
    if prev:
        bpy.data.images.remove(prev)
    img = bpy.data.images.load(path, check_existing=False)
    img.name = dname
    img.colorspace_settings.name = colorspace

    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if bsdf is None:
        bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')

    tex = next((n for n in nt.nodes if n.type == 'TEX_IMAGE' and n.name == node_name), None)
    if tex is None:
        tex = nt.nodes.new('ShaderNodeTexImage')
        tex.name = node_name
    tex.image = img
    # stacked below/beside the normal chain, one row per kind
    y = 220 - 120 * (_PAINT_ORDER[kind] + 1)
    tex.location = (bsdf.location.x - 400, bsdf.location.y + y)
    only_new = not any(l.from_node is tex and l.from_socket.name == 'Color'
                       for l in nt.links if l.to_node is bsdf
                       and l.to_socket.name == target_socket)
    if only_new:
        nt.links.new(tex.outputs['Color'], bsdf.inputs[target_socket])


# ------------------------------------------------------------- preview
class PN_OT_preview_bake(bpy.types.Operator):
    """Show the baked maps"""
    bl_idname = "painterly.preview_bake"
    bl_label = ""
    bl_options = {'REGISTER'}

    def execute(self, context):
        s = context.scene.painterly_settings
        base = s.tex_name.strip() or "Painterly"
        for kind in _selected_kinds(s):
            path = os.path.join(bpy.app.tempdir, base + KIND_SUFFIX[kind])
            if not os.path.isfile(path):
                self.report({'ERROR'}, f"No bake found at {path}. Run Send to Isophote first.")
                return {'CANCELLED'}
            return _open_view(path, f"Baked {KIND_LABELS[kind]}")
        return {'CANCELLED'}


class PN_OT_preview_result(bpy.types.Operator):
    """Show the AI-painted normal"""
    bl_idname = "painterly.preview_result"
    bl_label = ""
    bl_options = {'REGISTER'}

    def execute(self, context):
        s = context.scene.painterly_settings
        base = s.tex_name.strip() or "Painterly"
        for kind in _selected_kinds(s):
            if kind == 'normal':
                path = os.path.join(bpy.app.tempdir, base + "_ai.png")
                name = "Painted"
            else:
                path = os.path.join(bpy.app.tempdir, f"{base}_{kind}_painted.png")
                name = f"Painted {KIND_LABELS[kind]}"
            if not os.path.isfile(path):
                self.report({'ERROR'}, f"No painted result at {path}. Run Send to Isophote first.")
                return {'CANCELLED'}
            return _open_view(path, name)
        return {'CANCELLED'}


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
        description="Base name for generated textures (final: <name>_normal.png, <name>_ai.png, <name>_final.png, multi: <name>_<kind>.png)",
        default="",
        maxlen=120,
    )
    tex_size: EnumProperty(
        name="Texture Size",
        items=[
            ('256', "256px", "256 x 256"),
            ('512', "512px", "512 x 512"),
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
    bake_color: BoolProperty(
        name="Base Color", default=False,
        description="Also bake and paint the Base Color (diffuse albedo) map",
    )
    bake_rough: BoolProperty(
        name="Roughness", default=False,
        description="Also bake and paint the Roughness map",
    )
    bake_metallic: BoolProperty(
        name="Metallic", default=False,
        description="Also bake and paint the Metallic map",
    )
    bake_emission: BoolProperty(
        name="Emission", default=False,
        description="Also bake and paint the Emission map",
    )
    strokes_coherent: BoolProperty(
        name="Same Brushstrokes", default=True,
        description="Same brushstrokes across channels (recommended). "
                    "Off: each channel painted independently with the same seed",
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

    @staticmethod
    def _principled(mat):
        """First Principled BSDF of the material tree (same rule as apply)."""
        if not mat or not mat.use_nodes:
            return None
        return next((n for n in mat.node_tree.nodes
                     if n.type == 'BSDF_PRINCIPLED'), None)

    @staticmethod
    def _has_input(bsdf, socket_name, default, tol=0.01):
        """Socket is 'real': linked, or default differs from its flat value."""
        if bsdf is None:
            return False
        sock = bsdf.inputs.get(socket_name)
        if sock is None:
            return False
        if sock.is_linked:
            return True
        dv = sock.default_value
        if isinstance(default, tuple):
            return any(abs(dv[i] - default[i]) > tol for i in range(len(default)))
        return abs(float(dv) - default) > tol

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        mat = obj.active_material if obj else None
        layout.label(text="Material: " + (mat.name if mat else "none"))

        s = context.scene.painterly_settings
        # auto-fill happens in the timer (ID-property writes are not allowed
        # during UI draw); draw only reads
        bsdf = self._principled(mat)

        # --- channel table: checkbox per map; only enable what's real ---
        box = layout.box()
        box.label(text="Maps to paint:")
        if bsdf is None:
            box.label(text="needs Principled BSDF", icon='ERROR')
        else:
            # normal: always available (the headline feature)
            box.label(text="✓ Normal map", icon='CHECKMARK')
            row = box.row()
            row.enabled = self._has_input(bsdf, 'Base Color', (0.8, 0.8, 0.8, 1.0))
            row.prop(s, "bake_color")
            row = box.row()
            row.enabled = self._has_input(bsdf, 'Roughness', 0.5)
            row.prop(s, "bake_rough")
            row = box.row()
            row.enabled = self._has_input(bsdf, 'Metallic', 0.0)
            row.prop(s, "bake_metallic")
            emi = (bsdf.inputs.get('Emission Strength') is not None
                   and bsdf.inputs['Emission Strength'].default_value > 0.0) or \
                  self._has_input(bsdf, 'Emission Color', (1.0, 1.0, 1.0, 1.0), 0.001)
            row = box.row()
            row.enabled = emi
            row.prop(s, "bake_emission")
            box.prop(s, "strokes_coherent")

        n = 1 + sum(1 for p in ("bake_color", "bake_rough", "bake_metallic",
                                "bake_emission") if getattr(s, p))
        layout.operator(PN_OT_send_to_isophote.bl_idname,
                        text=f"Send to Isophote ({n} map{'s' if n != 1 else ''})")
        layout.operator(PN_OT_apply_from_isophote.bl_idname, text="Apply from Isophote")
        layout.operator(PN_OT_apply_animated_from_isophote.bl_idname, text="Apply animated (seeds)")
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
    PN_OT_apply_animated_from_isophote,
    PN_PT_panel,
    PN_PT_misc,
)


# ------------------------------------------------------------- embedded bridge
# Tiny HTTP relay living inside the addon (no numpy/cv2 needed — painting
# happens in the browser). The web app at
# https://jordandubu.github.io/isophote/ polls it on 127.0.0.1:8778.
BRIDGE_PORT = 8778

_bridge_state = {"pending_png": None, "pending_ack": False,
                 "pending_kind": None,     # kind of pending_png ('normal' by default)
                 "pending_multi": None,    # {kind:"multi", kinds, pngs} queue
                 "result_png": None, "result_kind": None,   # legacy single result
                 "result_kinds": None,     # multi single-frame result: [kinds]
                 "result_pngs": None,      # multi single-frame result: [b64]
                 "result_anim": None,      # legacy anim: [b64]
                 "result_anim_kinds": None}  # multi anim: kinds for anim
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
            _bridge_state["pending_kind"] = req.get("kind", "normal")
            _bridge_state["pending_multi"] = None
            _bridge_state["pending_ack"] = False
            _bridge_state["result_png"] = None
            _bridge_state["result_kinds"] = None
            return self._reply({"ok": True})

        if self.path == "/poll":             # app pulls from Blender
            # sticky: do NOT clear on read; the page acks after painting.
            # Prevents a second tab / slow load from stealing the image.
            multi = _bridge_state["pending_multi"]
            if multi is not None:
                return self._reply({"kind": "multi",
                                    "kinds": multi["kinds"],
                                    "pngs": multi["pngs"],
                                    "ack": True})
            png = _bridge_state["pending_png"]
            if png is None:
                return self._reply({"kind": "none"})
            return self._reply({"kind": "image", "bgr_png": png,
                                "map_kind": _bridge_state["pending_kind"] or "normal",
                                "ack": True})

        if self.path == "/ack":              # page says: painted it, clear
            _bridge_state["pending_png"] = None
            _bridge_state["pending_kind"] = None
            _bridge_state["pending_multi"] = None
            _bridge_state["pending_ack"] = False
            return self._reply({"ok": True})

        if self.path == "/send_to_blender":  # app result -> Blender
            anim = req.get("anim")
            if isinstance(anim, list) and anim and isinstance(anim[0], list):
                # multi animated: outer = frames, inner = one b64 per kind
                _bridge_state["result_anim_kinds"] = req.get("kinds")
                _bridge_state["result_anim"] = anim
                _bridge_state["result_png"] = None
                _bridge_state["result_kinds"] = None
                _bridge_state["result_pngs"] = None
            elif isinstance(anim, list):
                # legacy animated (single channel: list of b64)
                _bridge_state["result_anim_kinds"] = None
                _bridge_state["result_anim"] = anim
                _bridge_state["result_png"] = None
                _bridge_state["result_kinds"] = None
                _bridge_state["result_pngs"] = None
            elif "kinds" in req and "pngs" in req:
                # multi single-frame result
                _bridge_state["result_kinds"] = req.get("kinds")
                _bridge_state["result_pngs"] = req.get("pngs")
                _bridge_state["result_png"] = None
                _bridge_state["result_anim"] = None
                _bridge_state["result_anim_kinds"] = None
            else:
                # legacy single: {png, kind?}
                _bridge_state["result_png"] = req.get("png")
                _bridge_state["result_kind"] = req.get("kind", "normal")
                _bridge_state["result_kinds"] = None
                _bridge_state["result_pngs"] = None
                _bridge_state["result_anim"] = None
                _bridge_state["result_anim_kinds"] = None
            return self._reply({"ok": True})

        if self.path == "/result":           # addon pulls the app's painting
            anim = _bridge_state.get("result_anim")
            if anim is not None:
                kinds = _bridge_state.get("result_anim_kinds")
                if kinds:
                    _bridge_state["result_anim"] = None
                    _bridge_state["result_anim_kinds"] = None
                    return self._reply({"kind": "anim_multi", "kinds": kinds,
                                        "frames": anim})
                _bridge_state["result_anim"] = None
                return self._reply({"kind": "anim", "frames": anim})
            kinds = _bridge_state.get("result_kinds")
            if kinds:
                return self._reply({"kind": "multi", "kinds": kinds,
                                    "pngs": _bridge_state.get("result_pngs")})
            png = _bridge_state["result_png"]
            _bridge_state["result_png"] = None
            return self._reply({"kind": "image", "png": png,
                                "map_kind": _bridge_state.get("result_kind") or "normal"}
                               if png else {"kind": "none"})

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


def _autofill_name():
    """Keep Name synced to the active mesh name until the user types their own.
    Runs on a timer: writing ID properties is forbidden inside UI draw."""
    try:
        ctx = bpy.context
        s = getattr(ctx.scene, "painterly_settings", None)
        if s is not None:
            obj = ctx.active_object
            last = AUTO_NAME.get("last")
            if obj and (s.tex_name == last or not s.tex_name):
                s.tex_name = obj.name or "Painterly"
            AUTO_NAME["last"] = s.tex_name
            # surface what the bridge is holding, so the user knows the
            # next step is Apply (the web app cannot apply by itself)
            kinds = _bridge_state.get("result_kinds")
            if kinds:
                waiting = (", ".join(KIND_LABELS[k] for k in kinds)
                           + " waiting — click 'Apply from Isophote'")
            elif _bridge_state.get("result_anim_kinds"):
                waiting = (", ".join(KIND_LABELS[k] for k in _bridge_state["result_anim_kinds"])
                           + " animated frames waiting — click 'Apply animated (seeds)'")
            elif _bridge_state.get("result_png"):
                waiting = "frame waiting — click 'Apply from Isophote'"
            elif _bridge_state.get("result_anim"):
                waiting = "animated frames waiting — click 'Apply animated (seeds)'"
            else:
                waiting = ""
            s.status = waiting if waiting else (
                "" if s.status.startswith(("frame waiting", "animated frames waiting",
                                           "waiting", ", "))
                else s.status)
    except Exception:
        pass
    return 0.5


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.painterly_settings = bpy.props.PointerProperty(type=PN_scene_settings)
    _bridge_start()
    if not bpy.app.timers.is_registered(_autofill_name):
        bpy.app.timers.register(_autofill_name, first_interval=0.5, persistent=True)


def unregister():
    if bpy.app.timers.is_registered(_autofill_name):
        bpy.app.timers.unregister(_autofill_name)
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.painterly_settings
    _bridge_stop()
