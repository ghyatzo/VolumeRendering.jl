module VolumeFields

include("gl.jl")
include("camera.jl")
include("transfer_function.jl")
include("region.jl")
include("field_source.jl")
include("vector_field.jl")
include("geometry_renderer.jl")
include("volume_renderer.jl")   # defines the Overlay protocol + GeomContext, used by overlays.jl
include("overlays.jl")

# ── public API ──
export FieldView, render!, RenderParams, TransferFunction, Camera
export default_window!, normalize_points!   # transfer-function window / opacity-curve helpers
export FieldSource, GLSLField, axis_index_glsl
export Region, BoxRegion, SphereRegion
# interface functions users overload for custom sources / regions:
export field_glsl, region, value_range, fingerprint, upload_field, bind_field!, free_field!
export region_glsl, bind_region!, bounds
# camera controls (host wires input to these):
export orbit!, flylook!, pan!, zoom!, fly!, reset!, view_proj
# overlay protocol + concrete overlays:
export Overlay, enabled, refresh!, draw!, aabb
export SliceOverlay, BoxOutlineOverlay, SphereOverlay, LineOverlay, AxesOverlay
export StreamlinesOverlay, GlyphsOverlay, default_overlays
# vector fields:
export VectorField, compute_streamlines

# The interactive view widget. The ONLY UI the package ships; its method lives in the CImGui
# extension (ext/CImGuiExt.jl), so the core carries no CImGui dependency. A host that
# `using`s CImGui gets `ShowField(view; size)` automatically.
function ShowField end
export ShowField

end # module VolumeFields
