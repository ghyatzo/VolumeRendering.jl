# Colour (§4), marks and their validation (§5), the View (§8), and the render-on-demand snapshot (§7.3).

# ── Colormap (§4.1) ──

mutable struct Colormap
    scheme::Symbol
    window::ClosedInterval{Float64}   # value units, also when log
    log::Bool

    Colormap(scheme, window, log) = validate(new(scheme, window, log))
end
Colormap(scheme::Symbol, window::ClosedInterval; log = false) = Colormap(scheme, window, log)

function validate(c::Colormap)
    lo, hi = endpoints(c.window)
    lo < hi || throw(ArgumentError("Colormap window must have minimum < maximum, got $(c.window)"))
    c.log && !(lo > 0) && throw(ArgumentError("a log Colormap needs a positive window minimum, got $(c.window)"))
    c
end

# ── Mapped (§4.2) ──

mutable struct Mapped
    source::Any
    colormap::Colormap

    Mapped(source, colormap) = validate(new(source, colormap))
end

validate(m::Mapped) = (source_kind(m.source); validate(m.colormap); m)

source_kind(::Vector{<:Vector{<:Real}}) = :values
source_kind(f) = isfield(f) && value_kind(f) === :scalar ? :field :
    throw(ArgumentError("Mapped source must be a :scalar field or per-vertex values (Vector{Vector{<:Real}}), got $(summary(f))"))

value_range(vals::Vector{<:Vector{<:Real}}) = nan_extrema(Iterators.flatten(vals))

# ── colour sources (§4.3) ──

color_kind(::Color) = :constant
color_kind(c::TransparentColor) =
    throw(ArgumentError("a constant colour must be an alpha-less Color (a mark's opacity is its alpha), got $(typeof(c))"))
color_kind(::Vector{<:Vector{<:Color}}) = :vertex_colors
color_kind(c::Vector{<:Vector{<:TransparentColor}}) =
    throw(ArgumentError("per-vertex colours must be alpha-less Colors, got $(eltype(eltype(c)))"))
color_kind(m::Mapped) = (validate(m); source_kind(m.source) === :values ? :mapped_values : :mapped_field)
color_kind(f) = isfield(f) && value_kind(f) === :color ? :color_field :
    throw(ArgumentError("a mark colour must be a Color, per-vertex Colors, a :color field or a Mapped; got $(summary(f))"))

# ── marks (§5) ──

Base.@kwdef mutable struct Volume
    color::Any
    region::Any
    extinction::Vector{SVector{2,Float64}} = [SVector(0.0, 0.0), SVector(1.0, 1.0)]   # (u, σ), Mapped colours only
    mode::Symbol = :dvr
    density::Float64 = 1.0
    step_scale::Float64 = 0.5
    label::Union{Nothing,String} = nothing

    Volume(color, region, extinction, mode, density, step_scale, label) =
        validate(new(color, region, extinction, mode, density, step_scale, label))
end
Volume(color; kw...) = Volume(; color, kw...)

Base.@kwdef mutable struct Surface
    shape::Any
    color::Any = RGB(0.5, 0.5, 0.55)
    opacity::Float64 = 1.0
    label::Union{Nothing,String} = nothing

    Surface(shape, color, opacity, label) = validate(new(shape, color, opacity, label))
end
Surface(shape; kw...) = Surface(; shape, kw...)

Base.@kwdef mutable struct Lines
    polylines::Vector{Vector{SVector{3,Float64}}}
    color::Any = RGB(0.8, 0.8, 0.85)
    width_px::Float64 = 1.0
    label::Union{Nothing,String} = nothing

    Lines(polylines, color, width_px, label) = validate(new(polylines, color, width_px, label))
end
Lines(polylines; kw...) = Lines(; polylines, kw...)

check_shape(s, what) = isshape(s) ? s :
    throw(ArgumentError("$what must implement the shape interface (shape_glsl, bounds, aabb), got $(summary(s))"))

function validate(m::Volume)
    kind = color_kind(m.color)
    kind in (:mapped_field, :color_field) || throw(ArgumentError("a Volume colour must be a Mapped field or a :color field, got $kind"))
    m.mode in (:dvr, :mip, :average) || throw(ArgumentError("Volume mode must be :dvr, :mip or :average, got $(repr(m.mode))"))
    m.mode === :dvr || kind === :mapped_field || throw(ArgumentError("Volume mode $(repr(m.mode)) requires a Mapped colour"))
    check_shape(m.region, "a Volume region")
    kind === :mapped_field && check_extinction(m.extinction)
    m
end

function check_extinction(e)
    length(e) >= 2 || throw(ArgumentError("Volume extinction needs at least 2 points, got $(length(e))"))
    issorted(e; by = first) || throw(ArgumentError("Volume extinction points must be sorted in u"))
    all(p -> all(in(0..1), p), e) || throw(ArgumentError("Volume extinction coordinates must lie in 0..1"))
end

function validate(m::Surface)
    kind = color_kind(m.color)
    kind in (:constant, :color_field, :mapped_field) ||
        throw(ArgumentError("a Surface colour must be a constant, a :color field or a Mapped field, got $kind"))
    0 < m.opacity <= 1 || throw(ArgumentError("Surface opacity must be in (0, 1], got $(m.opacity)"))
    check_shape(m.shape, "a Surface shape")
    m
end

function validate(m::Lines)
    kind = color_kind(m.color)
    kind === :vertex_colors && check_nesting(m.color, m.polylines)
    kind === :mapped_values && check_nesting(m.color.source, m.polylines)
    m.width_px > 0 || throw(ArgumentError("Lines width_px must be positive, got $(m.width_px)"))
    m
end

check_nesting(pervertex, polylines) = map(length, pervertex) == map(length, polylines) ||
    throw(ArgumentError("per-vertex colours/values must have the polylines' nesting and lengths"))

# drives progressive refinement (§6.6)
jittered(::Volume) = true
jittered(::Union{Surface,Lines}) = false

# ── View (§8) ──

mutable struct View
    marks::Vector{Any}
    camera::Camera
    initial_camera::Camera       # reset!(view) restores it; zoom clamps relative to it
    background::RGB{Float64}
    max_volume_px::Int           # cap on the volume march's long side (device px)
    refine_frames::Int           # jittered frames averaged while idle; 1 disables
    flymode::Bool
    fly_speed::Float64
    cache::Union{Nothing,ResourceCache}   # created by the first render!
    last::Any                    # snapshot of the last rendered frame
    nsamp::Int                   # refinement frames accumulated since `last` changed
end

View(marks::AbstractVector; camera::Camera, background = RGB(0.0, 0.0, 0.0), max_volume_px = 2048, refine_frames = 16,
     flymode = false, fly_speed = default_fly_speed(camera)) =
    View(marks, camera, deepcopy(camera), background, max_volume_px, refine_frames, flymode, fly_speed, nothing, nothing, 0)

default_fly_speed(cam::Camera) = (cam.projection === :ortho ? cam.ortho_half : distance(cam)) / 200

# in place: hosts may hold a reference to view.camera
reset!(view::View) = (reset!(view.camera, view.initial_camera); view)

# ── snapshot (§7.3): identity leaves only for bulk arrays, everything else by value; O(marks), never O(data) ──

fieldvalues(x) = ntuple(i -> getfield(x, i), fieldcount(typeof(x)))

snapshot(view::View, w, h) = (w, h, fieldvalues(view.camera)..., view.background, view.max_volume_px, view.refine_frames,
                              view.flymode, view.fly_speed, Tuple(map(snapshot, view.marks)))

# extinction is copied: it is compared by value, so an in-place edit must not change the stored snapshot
snapshot(m::Volume) = (:Volume, snapshot(m.color), snapshot_shape(m.region), copy(m.extinction), m.mode, m.density, m.step_scale)
snapshot(m::Surface) = (:Surface, snapshot_shape(m.shape), snapshot(m.color), m.opacity)
snapshot(m::Lines) = (:Lines, Identity(m.polylines), snapshot(m.color), m.width_px)

# colour sources and Mapped sources
snapshot(c::Colorant) = c
snapshot(pervertex::AbstractVector) = Identity(pervertex)
snapshot(m::Mapped) = (snapshot(m.source), fieldvalues(m.colormap)...)
snapshot(field) = snapshot_glsl(field_glsl(field), textures(field), uniforms(field))
snapshot_shape(s) = snapshot_glsl(shape_glsl(s), (;), uniforms(s), aabb(s))

snapshot_glsl(glsl, texs, unis, rest...) = (glsl, map(t -> (Identity(cache_key(t.data)), t.wrap, t.filter), texs), unis, rest...)
