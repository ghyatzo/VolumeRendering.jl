# Milky Way volumetric example: an RGBA colour field. The volume is OpenSpace's Milky Way model, based on NAOJ
# simulations: a 1024×1024×128 RGBA grid, RGB = emission colour. Download it (512 MiB) into examples/milkyway/ from
#   https://liu-se.cdn.openspaceproject.com/files/milkyway/milkyway/volume/1/MilkyWayRGBAVolume1024x1024x128.raw
# Five data-prep sliders shape each voxel before it is rendered; each edit re-maps the whole grid and swaps the field.

using VolumeRendering, AxisKeys
using Colors, FixedPointNumbers           # RGBA, red/green/blue; N0f8
import CImGui, GLFW, ModernGL
using CImGui.CSyntax                       # @c

const RAW = joinpath(@__DIR__, "milkyway", "MilkyWayRGBAVolume1024x1024x128.raw")
const NX, NY, NZ = 1024, 1024, 128
# physical half-extents (kpc): 38.9 × 38.9 × 4.86, centered on the Galactic Center
const XS = range(-19.45, 19.45, NX)
const YS = range(-19.45, 19.45, NY)
const ZS = range(-2.43, 2.43, NZ)

# raw bytes are channel-fastest → x → y → z, matching Julia's column-major RGBA{N0f8} (nx,ny,nz)
load_raw() = reshape(reinterpret(RGBA{N0f8}, read(RAW)), NX, NY, NZ)

# per-cell shaping. RGB: (1) raw per-channel power, (2) power the average keeping RGB ratio (hue),
# (3) brightness scale. Alpha: power + scale of the shaped-color average (independent of RGB brightness).
function cell(c, p1, p2, k, pa, ka)
    r = red(c)^p1; g = green(c)^p1; b = blue(c)^p1
    m = (r + g + b) / 3
    s = m > 0 ? m^(p2 - 1) : 0.0f0        # average → m^p2, channels keep their ratio
    r *= s; g *= s; b *= s
    a = clamp(ka * ((r + g + b) / 3)^pa, 0, 1)
    RGBA{Float32}(r * k, g * k, b * k, a)
end
# an RGBA grid is a :color field: rgb = emission, alpha = extinction
mw_field(p1, p2, k, pa, ka) = GridField(KeyedArray(map(c -> cell(c, p1, p2, k, pa, ka), BASE); x = XS, y = YS, z = ZS))

const BASE = load_raw()

p1 = Ref(Cfloat(2.0)); p2 = Ref(Cfloat(0.6)); k = Ref(Cfloat(40.0))   # RGB: raw power, color power, brightness
pa = Ref(Cfloat(0.5)); ka = Ref(Cfloat(5.0))                          # alpha: power, scale

vol = let f = mw_field(p1[], p2[], k[], pa[], ka[])   # `let`: no global keeps the 2 GiB grid alive after a remap
    Volume(f; region = Box(f), density = 0.5)
end
# the camera frames the box, tilted off face-on so the disk reads as 3-D
view = View([vol]; camera = orbit!(Camera(; bounds(vol.region)...), 0.6, 0.9))

# a new field each edit; the old texture is freed before the new one is uploaded
remap!() = (vol.color = mw_field(p1[], p2[], k[], pa[], ka[]))

CImGui.set_backend(:GlfwOpenGL3)
ctx = CImGui.CreateContext()
let io = CImGui.GetIO()
    io.ConfigFlags = unsafe_load(io.ConfigFlags) | CImGui.ImGuiConfigFlags_DockingEnable
end

CImGui.render(ctx;
    window_title = "VolumeRendering — Milky Way",
    window_size = (1100, 800),
    opengl_version = v"4.1") do
    CImGui.DockSpaceOverViewport(0, CImGui.GetMainViewport(), CImGui.ImGuiDockNodeFlags_PassthruCentralNode)

    CImGui.Begin("Milky Way", C_NULL)
    # sliders bound to persistent Refs: a fresh per-frame Ref would snap back mid-drag
    ch = false
    ch |= @c CImGui.SliderFloat("raw power",   &p1[], 0.2f0, 4.0f0,   "%.2f")
    ch |= @c CImGui.SliderFloat("color power", &p2[], 0.2f0, 3.0f0,   "%.2f")
    ch |= @c CImGui.SliderFloat("brightness",  &k[],  0.1f0, 200.0f0, "%.2f", CImGui.ImGuiSliderFlags_Logarithmic)
    ch |= @c CImGui.SliderFloat("alpha power", &pa[], 0.05f0, 3.0f0,  "%.2f")
    ch |= @c CImGui.SliderFloat("alpha scale", &ka[], 0.1f0, 20.0f0,  "%.2f", CImGui.ImGuiSliderFlags_Logarithmic)
    ch && remap!()
    VolumeRendering.ShowView(view)
    CImGui.End()
end
