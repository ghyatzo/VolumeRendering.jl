# Minimal VolumeRendering example: volume-render a 3-D KeyedArray and show it in an interactive window
using VolumeRendering, AxisKeys, ColorSchemes
import CImGui, GLFW, ModernGL            # GLFW + ModernGL activate the CImGui GLFW/OpenGL backend

# ── the field: the hydrogen 3d_xy orbital's electron density |ψ|² ∝ (x·y)² · e^(−2r/3). ──
function orbital(x, y, z)
    r = 8 * hypot(x, y, z)               # radius in Bohr radii (box scaled so the cloud fills the view)
    (x * y * exp(-r / 3))^2
end

xs = ys = zs = range(-1, 1, length = 140)
grid = GridField(KeyedArray([orbital(x, y, z) for x in xs, y in ys, z in zs]; x = xs, y = ys, z = zs))

# ── the scene: one volume over the grid's box, coloured through a colormap spanning the data range ──
vol = Volume(Mapped(grid, Colormap(ColorSchemes.viridis, value_range(grid))); region = Box(grid))
# the camera frames the box, tilted off face-on so the lobes read as 3-D ("reset view" returns to this pose)
view = View([vol]; camera = orbit!(Camera(; bounds(vol.region)...), 0.6, 0.5))

# ── interactive window (the render loop is owned by the host, per the package design) ──
CImGui.set_backend(:GlfwOpenGL3)
ctx = CImGui.CreateContext()
let io = CImGui.GetIO()
    io.ConfigFlags = unsafe_load(io.ConfigFlags) | CImGui.ImGuiConfigFlags_DockingEnable  # enable docking
end

CImGui.render(ctx;
    window_title = "VolumeRendering",
    window_size = (900, 900),
    opengl_version = v"4.1") do          # the shaders are #version 410 core
    CImGui.DockSpaceOverViewport(0, CImGui.GetMainViewport(), CImGui.ImGuiDockNodeFlags_PassthruCentralNode)
    CImGui.Begin("Volume", C_NULL)
    VolumeRendering.ShowView(view)
    CImGui.End()
end
