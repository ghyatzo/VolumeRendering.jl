# The one piece of UI the package ships: `show!(view; size)` — draw a FieldView as an interactive
# image inside the host's current CImGui window/layout, and turn mouse/keyboard into camera moves.
# Loaded automatically when a host `using`s both VolumeFields and CImGui. The core has no CImGui dep.

module CImGuiExt

using VolumeFields
import VolumeFields: show!, orbit!, flylook!, pan!, zoom!, fly!
import ModernGL as GL
import CImGui
using CImGui: ImVec2

const ROT_SPEED  = 0.007   # rad per pixel of drag
const ZOOM_SPEED = 0.12    # per mouse-wheel notch

# Per-view drag state (a drag continues even if the cursor leaves the image). Kept in a WeakKeyDict so
# it neither pollutes the core FieldView type nor leaks when a view is dropped.
mutable struct _Drag
    rotating::Bool
    panning::Bool
end
const _DRAG = Base.WeakKeyDict{FieldView,_Drag}()
_drag(view::FieldView) = get!(() -> _Drag(false, false), _DRAG, view)

# Draw `view` as an interactive image at `size` (ImGui item-size convention via CalcItemSize:
# negative = fill the content region, positive = explicit px, 0 = default). Renders at device
# resolution (logical size × DisplayFramebufferScale), shows the G-buffer texture V-flipped, and wires
# camera input scoped to this view's id (so multiple views — even in one window — don't collide).
function VolumeFields.show!(view::FieldView; size::ImVec2 = ImVec2(-CImGui.FLT_MIN, -CImGui.FLT_MIN))
    sz = CImGui.CalcItemSize(size, 256.0, 256.0)                 # 256² default if a component is 0
    dw = max(1, round(Int, sz.x)); dh = max(1, round(Int, sz.y)) # on-screen logical points
    fbs = unsafe_load(CImGui.GetIO().DisplayFramebufferScale)    # Retina: (2,2)
    gw = max(1, round(Int, dw * fbs.x)); gh = max(1, round(Int, dh * fbs.y))   # device px
    tex = VolumeFields.render!(view, gw, gh)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)                   # restore default FB so imgui draws to screen
    CImGui.PushID(pointer_from_objref(view))                     # per-view id scoping
    ref = CImGui.ImTextureRef(CImGui.ImTextureID(UInt64(tex)))
    CImGui.Image(ref, ImVec2(dw, dh), ImVec2(0, 1), ImVec2(1, 0))   # flip V: GL origin is bottom-left
    _handle_camera_input!(view, CImGui.IsItemHovered(), dw, dh)
    CImGui.PopID()
    nothing
end

function _handle_camera_input!(view::FieldView, hovered::Bool, w, h)
    io = CImGui.GetIO()
    d = unsafe_load(io.MouseDelta)
    drag = _drag(view); cam = view.camera; params = view.params
    hovered && CImGui.IsMouseClicked(0) && (drag.rotating = true)
    hovered && (CImGui.IsMouseClicked(1) || CImGui.IsMouseClicked(2)) && (drag.panning = true)
    CImGui.IsMouseDown(0) || (drag.rotating = false)
    (CImGui.IsMouseDown(1) || CImGui.IsMouseDown(2)) || (drag.panning = false)

    if drag.rotating
        dyaw = -d.x * ROT_SPEED; dpitch = -d.y * ROT_SPEED
        params.flymode ? flylook!(cam, dyaw, dpitch) : orbit!(cam, dyaw, -dpitch)
    elseif drag.panning
        pan!(cam, d.x / w, d.y / h)
    end

    if hovered
        wheel = unsafe_load(io.MouseWheel)
        wheel != 0 && zoom!(cam, exp(-ZOOM_SPEED * wheel))
        if params.flymode
            k(key) = CImGui.IsKeyDown(key) ? 1.0 : 0.0
            fwd = k(CImGui.ImGuiKey_W) - k(CImGui.ImGuiKey_S)
            rgt = k(CImGui.ImGuiKey_D) - k(CImGui.ImGuiKey_A)
            upd = k(CImGui.ImGuiKey_E) - k(CImGui.ImGuiKey_Q)
            speed = params.fly_speed * (unsafe_load(io.KeyShift) ? 5.0 : 1.0)   # Shift = sprint
            (fwd != 0 || rgt != 0 || upd != 0) && fly!(cam, fwd, rgt, upd, speed)
        end
    end
end

end # module CImGuiExt
