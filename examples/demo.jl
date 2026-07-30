# Minimal VolumeRendering example: build a FieldView from a 3-D KeyedArray and show it in an interactive window
using VolumeRendering, AxisKeys
import CImGui, GLFW, ModernGL            # GLFW + ModernGL activate the CImGui GLFW/OpenGL backend

# ── the field: the hydrogen 3d_xy orbital's electron density |ψ|² ∝ (x·y)² · e^(−2r/3). ──
function orbital(x, y, z)
    r = 8 * hypot(x, y, z)               # radius in Bohr radii (box scaled so the cloud fills the view)
    (x * y * exp(-r / 3))^2
end

xs = ys = zs = range(-1, 1, length = 140)
field = KeyedArray([orbital(x, y, z) for x in xs, y in ys, z in zs]; x = xs, y = ys, z = zs)

# ── interactive window (the render loop is owned by the host, per the package design) ──
CImGui.set_backend(:GlfwOpenGL3)
ctx = CImGui.CreateContext()
let io = CImGui.GetIO()
    io.ConfigFlags = unsafe_load(io.ConfigFlags) | CImGui.ImGuiConfigFlags_DockingEnable  # enable docking
end

view = Ref{Any}(nothing)

CImGui.render(ctx;
    window_title = "VolumeRendering",
    window_size = (900, 900),
    opengl_version = v"4.1") do          # the shaders are #version 410 core
    if view[] === nothing
        view[] = FieldView(field)
        orbit!(view[].camera, 0.6, 0.5)  # tilt off face-on so the lobes read as 3-D
    end
    
    dock = CImGui.DockSpaceOverViewport(0, CImGui.GetMainViewport(), CImGui.ImGuiDockNodeFlags_PassthruCentralNode)
    
    CImGui.Begin("Volume", C_NULL)
    VolumeRendering.ShowVolume(view[])
    CImGui.End()
end
