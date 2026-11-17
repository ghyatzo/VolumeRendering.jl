"""    Colormap(scheme::ColorScheme, window::ClosedInterval; log = false)

Maps values in `window` onto the colour scheme (a ColorSchemes.jl object), linearly or, with `log = true`, in log
scale; values outside the window clamp to its ends. `window` is in value units in both cases; a log window must be
positive.

Mutable and shared by reference: setting `scheme`, `window` or `log` updates every [`Mapped`](@ref) that uses it.
"""
mutable struct Colormap
    scheme::ColorScheme
    window::ClosedInterval{Float64}
    log::Bool

    Colormap(scheme, window, log) = validate(new(scheme, window, log))
end
Colormap(scheme::ColorScheme, window::ClosedInterval; log = false) = Colormap(scheme, window, log)

function validate(c::Colormap)
    lo, hi = endpoints(c.window)
    lo < hi || throw(ArgumentError("Colormap window must have minimum < maximum, got $(c.window)"))
    c.log && !(lo > 0) && throw(ArgumentError("a log Colormap needs a positive window minimum, got $(c.window)"))
    c
end

"""    Mapped(source, colormap::Colormap)

A colour from a scalar `source` through `colormap`. `source` is a `:scalar` field, or per-vertex values
(`Vector{Vector{<:Real}}`, nested like a [`Lines`](@ref) mark's polylines).

Mutable and shared by reference: setting `source` or `colormap` updates every mark that uses it.
"""
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

color_kind(::Color) = :constant
color_kind(c::TransparentColor) =
    throw(ArgumentError("a constant colour must be an alpha-less Color (a mark's opacity is its alpha), got $(typeof(c))"))
color_kind(::Vector{<:Vector{<:Color}}) = :vertex_colors
color_kind(c::Vector{<:Vector{<:TransparentColor}}) =
    throw(ArgumentError("per-vertex colours must be alpha-less Colors, got $(eltype(eltype(c)))"))
color_kind(m::Mapped) = (validate(m); source_kind(m.source) === :values ? :mapped_values : :mapped_field)
color_kind(f) = isfield(f) && value_kind(f) === :color ? :color_field :
    throw(ArgumentError("a mark colour must be a Color, per-vertex Colors, a :color field or a Mapped; got $(summary(f))"))

"""    Volume(color; region, extinction = [SVector(0.0, 0.0), SVector(1.0, 1.0)], mode = :dvr, density = 1.0, step_scale = 0.5, label = nothing)

A mark ray-marching `color` through the shape `region`. `color` is a [`Mapped`](@ref) scalar field or a `:color`
field.

- `mode`: `:dvr` (emission–absorption), `:mip` (maximum) or `:average` along the ray; `:mip` and `:average` need a
  `Mapped` colour.
- `extinction`: for a `Mapped` colour, extinction σ as a piecewise-linear function of the colormap position `u`:
  `(u, σ)` points sorted in `u`, all coordinates in `0..1`; at least two. A `:color` field brings its own σ (alpha).
- `density`: σ scale (DVR).
- `step_scale`: the march step is `step_scale × stepSize(p)` of the field.
- `label`: the mark's title in the controls panel.

A volume is translucent: composited over the opaque marks in list order with other translucent marks, unsorted.
Overlapping volumes are marched separately, not jointly.
"""
Base.@kwdef mutable struct Volume
    color::Any
    region::Any
    extinction::Vector{SVector{2,Float64}} = [SVector(0.0, 0.0), SVector(1.0, 1.0)]
    mode::Symbol = :dvr
    density::Float64 = 1.0
    step_scale::Float64 = 0.5
    label::Union{Nothing,String} = nothing

    Volume(color, region, extinction, mode, density, step_scale, label) =
        validate(new(color, region, extinction, mode, density, step_scale, label))
end
Volume(color; kw...) = Volume(; color, kw...)

"""    Surface(shape; color = RGB(0.5, 0.5, 0.55), opacity = 1.0, label = nothing)

A mark drawing the surface of `shape`. `color` is a constant `Color` (no alpha), a [`Mapped`](@ref) scalar field, or a
`:color` field, evaluated on the surface.

With `opacity < 1` the surface is translucent: both its front and back faces are blended, composited in list order
with the other translucent marks, unsorted. Opaque faces that coincide resolve by list order (later on top).
"""
Base.@kwdef mutable struct Surface
    shape::Any
    color::Any = RGB(0.5, 0.5, 0.55)
    opacity::Float64 = 1.0
    label::Union{Nothing,String} = nothing

    Surface(shape, color, opacity, label) = validate(new(shape, color, opacity, label))
end
Surface(shape; kw...) = Surface(; shape, kw...)

"""    Lines(polylines; color = RGB(0.8, 0.8, 0.85), width_px = 1.0, label = nothing)

A mark drawing `polylines` (`Vector{Vector{SVector{3,Float64}}}`), `width_px` device pixels wide, hard-edged.
`color` is a constant `Color` (no alpha), per-vertex `Color`s, a [`Mapped`](@ref) of per-vertex values or of a scalar
field, or a `:color` field. Per-vertex vectors are nested like `polylines`.

The `polylines` and per-vertex vectors are uploaded once per array: assign new arrays to change them, in-place edits
are not seen. Depth is constant across a line's width, so lines lying exactly on a sloped surface may lose edge
pixels.
"""
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
    m.density > 0 || throw(ArgumentError("Volume density must be positive, got $(m.density)"))
    m.step_scale > 0 || throw(ArgumentError("Volume step_scale must be positive, got $(m.step_scale)"))
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

# drives progressive refinement
jittered(::Volume) = true
jittered(::Union{Surface,Lines}) = false

"""    View(marks; camera::Camera, background = RGB(0.0, 0.0, 0.0), max_volume_px = 2048, refine_frames = 16,
         flymode = false, fly_speed)

A scene: the `marks` ([`Volume`](@ref), [`Surface`](@ref), [`Lines`](@ref)), drawn depth-composited, seen through
`camera`. Opaque marks are depth-tested; translucent ones composite in list order.

- `max_volume_px`: volumes are marched at most this many device pixels along the image's longer side, then upscaled.
- `refine_frames`: while nothing changes, this many jittered frames are averaged to smooth volume sampling noise;
  `1` disables.
- `flymode`, `fly_speed`: navigation in [`ShowView`](@ref): fly with WASD at `fly_speed` world units per frame
  instead of orbiting; the default speed is 1/200 of the camera's distance to `lookat` (ortho: of `ortho_half`).

Everything is plain mutable data, read at each render: set properties of the view, its camera, marks, fields and
colormaps to update; add or remove marks in `view.marks` to show or hide them. Nothing touches the GPU until the first
[`render!`](@ref), which renders only when something changed. `view.initial_camera` keeps a copy of `camera` for
[`reset!`](@ref) and as the [`zoom!`](@ref) reference.
"""
mutable struct View
    marks::Vector{Any}
    camera::Camera
    initial_camera::Camera
    background::RGB{Float64}
    max_volume_px::Int
    refine_frames::Int
    flymode::Bool
    fly_speed::Float64
    cache::Union{Nothing,ResourceCache}   # created by the first render!
    last::Any                    # snapshot of the last rendered frame
    nsamp::Int                   # refinement frames accumulated since `last` changed
end

View(marks::AbstractVector; camera::Camera, background = RGB(0.0, 0.0, 0.0), max_volume_px = 2048, refine_frames = 16,
     flymode = false, fly_speed = default_fly_speed(camera)) =
    validate(View(marks, camera, deepcopy(camera), background, max_volume_px, refine_frames, flymode, fly_speed, nothing, nothing, 0))

# runs at construction and at render, like the marks'
function validate(view::View)
    view.refine_frames >= 1 || throw(ArgumentError("View refine_frames must be ≥ 1, got $(view.refine_frames)"))
    view.max_volume_px >= 1 || throw(ArgumentError("View max_volume_px must be ≥ 1, got $(view.max_volume_px)"))
    view
end

default_fly_speed(cam::Camera) = (cam.projection === :ortho ? cam.ortho_half : distance(cam)) / 200

reset!(view::View) = (reset!(view.camera, view.initial_camera); view)

# render-on-demand snapshot: identity leaves only for bulk arrays, everything else by value; O(marks), never O(data)

fieldvalues(x) = ntuple(i -> getfield(x, i), fieldcount(typeof(x)))

snapshot(view::View, w, h) = (w, h, fieldvalues(view.camera)..., view.background, view.max_volume_px, view.refine_frames,
                              view.flymode, view.fly_speed, Tuple(map(snapshot, view.marks)))

# extinction is copied: it is compared by value, so an in-place edit must not change the stored snapshot
snapshot(m::Volume) = (:Volume, snapshot(m.color), snapshot_shape(m.region), copy(m.extinction), m.mode, m.density, m.step_scale)
snapshot(m::Surface) = (:Surface, snapshot_shape(m.shape), snapshot(m.color), m.opacity)
snapshot(m::Lines) = (:Lines, Identity(m.polylines), snapshot(m.color), m.width_px)

snapshot(c::Colorant) = c
snapshot(pervertex::AbstractVector) = Identity(pervertex)
snapshot(m::Mapped) = (snapshot(m.source), fieldvalues(m.colormap)...)
snapshot(field) = snapshot_glsl(field_glsl(field), textures(field), uniforms(field))
snapshot_shape(s) = snapshot_glsl(shape_glsl(s), (;), uniforms(s), aabb(s))

snapshot_glsl(glsl, texs, unis, rest...) = (glsl, map(t -> (Identity(cache_key(t.data)), t.wrap, t.filter), texs), unis, rest...)
