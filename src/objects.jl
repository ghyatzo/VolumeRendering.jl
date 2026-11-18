# ── Texture ──

"""    Texture(data; wrap = :clamp, filter = :linear)

A 1-, 2- or 3-D array for a field's GLSL to sample as `uniform samplerND <name>` (see [`textures`](@ref)). Array
dim 1, 2, 3 are the texture's s, t, r. Elements: `Real` → one float channel (`.r`); `Colorant` → RGBA, 8-bit for
fixed-point colours (`N0f8`), float otherwise.

`wrap` is `:clamp`, `:repeat` or `:mirror`, one per dim or one for all; `filter` is `:linear` or `:nearest`.

The upload is cached by the identity of `data` (unwrapped from a `KeyedArray`): a texture with a new array uploads
it, while in-place edits of an uploaded array are not seen.
"""
struct Texture{N,A<:AbstractArray{<:Any,N}}
    data::A
    wrap::NTuple{N,Symbol}
    filter::Symbol

    function Texture(data::A, wrap::NTuple{N,Symbol}, filter::Symbol) where {N,A<:AbstractArray{<:Any,N}}
        N in 1:3 || throw(ArgumentError("Texture data must be 1-, 2- or 3-D, got $N-D"))
        all(in(keys(WRAP)), wrap) || throw(ArgumentError("Texture wrap must be one of $(keys(WRAP)), got $wrap"))
        filter in keys(FILTER) || throw(ArgumentError("Texture filter must be one of $(keys(FILTER)), got $(repr(filter))"))
        new{N,A}(data, wrap, filter)
    end
end

Texture(data::AbstractArray{<:Any,N}; wrap = :clamp, filter = :linear) where {N} =
    Texture(data, wrap isa Symbol ? ntuple(Returns(wrap), N) : wrap, filter)

# textures are cached by their unwrapped array, so a new KeyedArray wrapper around the same parent does not re-upload
cache_key(data::AbstractArray) = AxisKeys.keyless_unname(data)

# ── field and shape interfaces ──

"""    field_glsl(f)::String

The GLSL of a field `f`; with [`value_kind`](@ref), the required part of the field interface. Any type implementing
it is a field; extend these functions qualified (`VolumeRendering.field_glsl(f::MyField) = …`).

The GLSL defines `float value(vec3 p)` (a `:scalar` field) or `vec4 value(vec3 p)` (a `:color` field: rgb emission and
extinction per unit length in alpha, not premultiplied), and `float stepSize(vec3 p)`, the positive march step
scale at `p`. It may define helpers, but declares no uniforms or samplers — those come from [`textures`](@ref) and
[`uniforms`](@ref) — and has no `#version` line. Fields and shapes assembled into one program share a namespace:
helper names must not collide, and `vr_`/`gl_` prefixes are reserved.

`field_glsl`, `textures` and `uniforms` are called every frame: keep them cheap, and store the arrays in `f`.
Optional parts: [`textures`](@ref), [`uniforms`](@ref), [`value_range`](@ref), [`controls!`](@ref).
"""
function field_glsl end

"""    value_kind(f)::Symbol

`:scalar` or `:color`: what the field's GLSL `value` returns (see [`field_glsl`](@ref)).
"""
function value_kind end

"""    textures(x)::NamedTuple

Textures of a field, as `name => Texture`; each is declared for its GLSL as `uniform samplerND <name>`. Defaults to
none. Return the same `Texture` arrays every call: uploads are cached by array identity. A program has at most 16
samplers, the renderer's own included.
"""
const MAX_SAMPLERS = 16   # GL_MAX_TEXTURE_IMAGE_UNITS on macOS GL 4.1
const MAX_BRICKS = MAX_SAMPLERS - 3   # leaves units for vr_cmap, vr_extinction, vr_scene_depth

textures(_) = (;)

"""    uniforms(x)::NamedTuple

Uniforms of a field or shape, as `name => value`, declared for its GLSL as `uniform <type> <name>`. Defaults to
none. Value types: `Bool`, `Integer`, `Real`, `SVector`s of 2–4 integers or reals, 3×3 or 4×4 `SMatrix`
(column-major: GLSL `M[col][row]`).

Names of uniforms and textures must be valid GLSL identifiers, not keywords, not `value`, `stepSize`, `main`, not
starting with `vr_` or `gl_`, and unique among all objects assembled into one program.
"""
uniforms(_) = (;)

"""    value_range(f)
    value_range(values::Vector{<:Vector{<:Real}})

The range of a scalar field's values, as a `ClosedInterval`, for building a [`Colormap`](@ref) window; `0.0..1.0`
unless a field type defines it. For per-vertex values, their NaN-skipping extrema. The renderer never reads it.
"""
value_range(_) = 0.0..1.0

"""    shape_glsl(s)::String

The GLSL of a shape `s`; with [`bounds`](@ref) and [`aabb`](@ref), the required part of the shape interface. Any type
implementing them is a shape, usable as a Volume region or a Surface; extend these functions qualified. Optional:
[`uniforms`](@ref).

The GLSL defines `vec2 vr_hit(vec3 ro, vec3 rd)`: the ray parameters of entry and exit along the ray `ro + t·rd`
(`rd` unit length), unclamped (negative is behind the eye), with exit < entry on a miss. Shapes must be convex: one
entry and one exit per ray.
"""
function shape_glsl end

"""    bounds(s)

A sphere enclosing the shape `s`, as `(; center, radius)`; `Camera(; bounds(s)...)` frames it.
"""
function bounds end

"""    aabb(s)

The axis-aligned bounding box of the shape `s`, as `(; lo, hi)` corner `SVector`s.
"""
function aabb end

isfield(x) = applicable(field_glsl, x) && applicable(value_kind, x)
isshape(x) = applicable(shape_glsl, x) && applicable(bounds, x) && applicable(aabb, x)

nan_extrema(itr) = ((lo, hi) = extrema(Iterators.filter(!isnan, itr)); convert(Float64, lo)..convert(Float64, hi))

# ── GridField ──

"""    GridField(A::KeyedArray; interpolation = :linear)

A field sampled from a grid. Dim names choose the coordinates, in any order:
- `(x, y, z)`: Cartesian; [`Box(g)`](@ref Box) is the box of its keys;
- `(lon, lat)`: directions from the origin, in radians (or dimensionless Unitful angles); longitude wraps around.
  The march step is proportional to the distance from the origin; near it, the Volume's `max_samples` floors the step.

Axis keys are `Real` ranges; other axis types need an [`axis_index_glsl`](@ref) method. `Real` elements make a
`:scalar` field; `Colorant` elements a `:color` field (rgb emission, alpha extinction). `interpolation` is `:linear` or
`:nearest`.

Mutable: assign `g.data` or `g.interpolation`. The data are uploaded once per array: assign a new array to change them,
in-place edits are not seen.
"""
mutable struct GridField
    data::KeyedArray
    interpolation::Symbol

    function GridField(data::KeyedArray, interpolation::Symbol)
        coordinates(data)   # validates the dim names
        interpolation in keys(FILTER) || throw(ArgumentError("GridField interpolation must be one of $(keys(FILTER)), got $(repr(interpolation))"))
        new(data, interpolation)
    end
end
GridField(A::KeyedArray; interpolation = :linear) = GridField(A, interpolation)

struct Cartesian end
struct LonLat end

coordinates(A::KeyedArray) = coordinates(Val(Tuple(sort!(collect(dimnames(A))))))
coordinates(::Val{(:x, :y, :z)}) = Cartesian()
coordinates(::Val{(:lat, :lon)}) = LonLat()
coordinates(::Val{names}) where {names} = throw(ArgumentError("GridField dim names must be (x, y, z) or (lon, lat), in any order; got $names"))

# GLSL variable holding each coordinate
coordvar(::Cartesian, name) = (x = "p.x", y = "p.y", z = "p.z")[name]
coordvar(::LonLat, name) = (lon = "vr_lon", lat = "vr_lat")[name]

# GLSL literals come from convert(Float64, ·): plain numbers pass, dimensionless Unitful quantities become plain
# numbers (u"°" → radians), dimensionful ones throw
glsl_float(x) = string(convert(Float64, x))

"""    axis_index_glsl(key, coordvar::AbstractString)::String

GLSL expression mapping the coordinate held by the GLSL expression `coordvar` to a fractional 0-based cell index along
the axis keys `key`. Defined for `AbstractRange`s; add a method to use another axis type in a [`GridField`](@ref).
"""
axis_index_glsl(key::AbstractRange, coordvar::AbstractString)::String =
    "(($coordvar) - $(glsl_float(first(key)))) * $(glsl_float(inv(convert(Float64, step(key)))))"

_axis_min_cell(key::AbstractRange) = abs(convert(Float64, step(key)))
_axis_min_cell(key) = minimum(d -> abs(convert(Float64, d)), diff(key))

min_cell(A::KeyedArray) = minimum(_axis_min_cell, axiskeys(A))

coords_glsl(::Cartesian, A) = ""
function coords_glsl(::LonLat, A)
    k = named_axiskeys(A).lon
    lon0 = glsl_float(first(k))
    lon = convert(Float64, last(k)) > convert(Float64, first(k)) ? "$lon0 + mod(atan(p.y, p.x) - $lon0, 6.283185307179586)" :
                                          "$lon0 - mod($lon0 - atan(p.y, p.x), 6.283185307179586)"
    """
        float vr_lat = asin(clamp(p.z / length(p), -1.0, 1.0));
        float vr_lon = $lon;
    """
end

step_glsl(::Cartesian, A) = glsl_float(min_cell(A))
step_glsl(::LonLat, A) = "$(glsl_float(min_cell(A))) * length(p)"

function field_glsl(f::GridField)
    A = f.data
    c = coordinates(A)
    vec = "vec$(ndims(A))"
    idx = join(map((name, key) -> axis_index_glsl(key, coordvar(c, name)), dimnames(A), axiskeys(A)), ", ")
    T, channel = value_kind(f) === :color ? ("vec4", "") : ("float", ".r")
    """
    $T value(vec3 p) {
    $(coords_glsl(c, A))    $vec idx  = $vec($idx);
        $vec dims = $vec($(join(map(glsl_float, size(A)), ", ")));
        return texture(grid, (idx + 0.5) / dims)$channel;
    }
    float stepSize(vec3 p) { return $(step_glsl(c, A)); }
    """
end

# longitude is periodic; every other axis clamps
textures(f::GridField) = (; grid = Texture(f.data; wrap = map(n -> n === :lon ? :repeat : :clamp, dimnames(f.data)), filter = f.interpolation))
value_kind(f::GridField) = eltype(f.data) <: Colorant ? :color : :scalar
value_range(f::GridField) = data_range(f.data)
data_range(A::AbstractArray{<:Colorant}) = 0.0..1.0
data_range(A::AbstractArray{<:Real}) = nan_extrema(A)

# ── TiledField ──

"""    TiledField(A::KeyedArray; interpolation = :linear, maxdim = 2048)

A Cartesian `(x, y, z)` grid field split into overlapping 3-D texture bricks of at most `maxdim` texels per axis, so
grids larger than `GL_MAX_3D_TEXTURE_SIZE` render at full resolution with seamless `:linear` filtering across brick
boundaries. At most $MAX_BRICKS bricks may be used.

Mutable: assign `t.data`, `t.interpolation` or `t.maxdim`.
"""
mutable struct TiledField
    data::KeyedArray
    interpolation::Symbol
    maxdim::Int
    k::NTuple{3,Int}
    L::NTuple{3,Int}
    D::NTuple{3,Int}
    bricks::Vector{<:Array{<:Any,3}}

    function TiledField(data::KeyedArray, interpolation::Symbol, maxdim::Integer = 2048)
        interpolation in keys(FILTER) || throw(ArgumentError("TiledField interpolation must be one of $(keys(FILTER)), got $(repr(interpolation))"))
        k, L, D, bricks = tile_bricks(data, Int(maxdim))
        new(data, interpolation, Int(maxdim), k, L, D, bricks)
    end
end
TiledField(A::KeyedArray; interpolation = :linear, maxdim = 2048) = TiledField(A, interpolation, maxdim)

function tile_bricks(A::KeyedArray, maxdim::Int)
    coordinates(A) isa Cartesian || throw(ArgumentError("TiledField needs a Cartesian (x, y, z) grid, got $(dimnames(A))"))
    maxdim >= 2 || throw(ArgumentError("TiledField maxdim must be at least 2, got $maxdim"))
    n = size(A)
    all(>(0), n) || throw(ArgumentError("TiledField axes must not be empty, got $n"))
    k = ntuple(d -> n[d] > maxdim ? cld(n[d] - 1, maxdim - 1) : 1, 3)
    prod(k) <= MAX_BRICKS || throw(ArgumentError("TiledField needs $(prod(k)) bricks, more than the maximum of $MAX_BRICKS"))
    L = ntuple(d -> k[d] == 1 ? n[d] : cld(n[d] - 1, k[d]), 3)
    D = ntuple(d -> k[d] == 1 ? n[d] : L[d] + 1, 3)
    raw = AxisKeys.keyless_unname(A)
    bricks = [brick_data(raw, k, L, D, _brick_coords(k, i)) for i in 0:prod(k)-1]
    k, L, D, bricks
end

_brick_coords(k::NTuple{3,Int}, i::Int) =
    let (b1, r) = divrem(i, k[2] * k[3]), (b2, b3) = divrem(r, k[3]); (b1, b2, b3) end
_brick_coords(t::TiledField, i::Int) = _brick_coords(t.k, i)

function _brick_axis_index(A, L::NTuple{3,Int}, D::NTuple{3,Int}, d::Int, b::Int)
    nd = size(A, d)
    i0 = firstindex(A, d)
    [i0 + clamp(b * L[d] + loc, 0, nd - 1) for loc in 0:D[d]-1]
end

function brick_data(A, k::NTuple{3,Int}, L::NTuple{3,Int}, D::NTuple{3,Int}, b::NTuple{3,Int})
    src = ntuple(d -> _brick_axis_index(A, L, D, d, b[d]), 3)
    B = Array{eltype(A)}(undef, D)
    for I in CartesianIndices(B)
        B[I] = A[src[1][I[1]], src[2][I[2]], src[3][I[3]]]
    end
    B
end
brick_data(t::TiledField, b::NTuple{3,Int}) = brick_data(AxisKeys.keyless_unname(t.data), t.k, t.L, t.D, b)

function _map_axis(t::TiledField, d::Int, g::Real)
    kd, Ld, Dd = t.k[d], t.L[d], t.D[d]
    b = clamp(floor(Int, g / Ld), 0, kd - 1)
    l = clamp(g - b * Ld, 0.0, Dd - 1.0)
    (b, l)
end

function Base.setproperty!(t::TiledField, name::Symbol, x)
    if name === :data
        if AxisKeys.keyless_unname(x) === AxisKeys.keyless_unname(t.data)
            coordinates(x) isa Cartesian || throw(ArgumentError("TiledField needs a Cartesian (x, y, z) grid, got $(dimnames(x))"))
            return setfield!(t, :data, x)
        end
        k, L, D, bricks = tile_bricks(x, t.maxdim)
        setfield!(t, :data, x); setfield!(t, :k, k); setfield!(t, :L, L); setfield!(t, :D, D); setfield!(t, :bricks, bricks)
        return x
    elseif name === :maxdim
        k, L, D, bricks = tile_bricks(t.data, Int(x))
        setfield!(t, :maxdim, Int(x)); setfield!(t, :k, k); setfield!(t, :L, L); setfield!(t, :D, D); setfield!(t, :bricks, bricks)
        return x
    end
    setfield!(t, name, convert(fieldtype(TiledField, name), x))
end

function field_glsl(t::TiledField)
    A = t.data
    c = coordinates(A)
    idx = join(map((name, key) -> axis_index_glsl(key, coordvar(c, name)), dimnames(A), axiskeys(A)), ", ")
    T, channel = value_kind(t) === :color ? ("vec4", "") : ("float", ".r")
    L = join(map(glsl_float, t.L), ", ")
    D = join(map(glsl_float, t.D), ", ")
    Dmax = join(map(d -> glsl_float(d - 1), t.D), ", ")
    kmax = join(map(k -> string(k - 1), t.k), ", ")
    nB = length(t.bricks)
    branches = join([i == 1 ? (nB == 1 ? "        return texture(brick_0, u)$channel;" :
                                         "        if (bid == 0) return texture(brick_0, u)$channel;") :
                     i < nB ? "        else if (bid == $(i - 1)) return texture(brick_$(i - 1), u)$channel;" :
                              "        else return texture(brick_$(i - 1), u)$channel;"
                     for i in 1:nB], "\n")
    """
    $T value(vec3 p) {
        vec3 g = vec3($idx);
        ivec3 b = clamp(ivec3(floor(g / vec3($L))), ivec3(0), ivec3($kmax));
        vec3 l = clamp(g - vec3(b) * vec3($L), vec3(0.0), vec3($Dmax));
        vec3 u = (l + 0.5) / vec3($D);
        int bid = (b.x * $(t.k[2]) + b.y) * $(t.k[3]) + b.z;
$branches
    }
    float stepSize(vec3 p) { return $(step_glsl(c, A)); }
    """
end

textures(t::TiledField) = NamedTuple{ntuple(i -> Symbol(:brick_, i - 1), length(t.bricks))}(
    ntuple(i -> Texture(t.bricks[i]; filter = t.interpolation), length(t.bricks)))
value_kind(t::TiledField) = eltype(t.data) <: Colorant ? :color : :scalar
value_range(t::TiledField) = data_range(t.data)

# ── GLSLField ──

"""    GLSLField(glsl; kind = :scalar, textures = (;), uniforms = (;), value_range = 0.0..1.0)

A field given directly by its GLSL: each argument is what the corresponding interface function returns (see
[`field_glsl`](@ref)). Immutable: build a new one to change it.
"""
struct GLSLField
    glsl::String
    kind::Symbol
    textures::NamedTuple
    uniforms::NamedTuple
    value_range::ClosedInterval{Float64}

    function GLSLField(glsl, kind, textures, uniforms, value_range)
        kind in (:scalar, :color) || throw(ArgumentError("GLSLField kind must be :scalar or :color, got $(repr(kind))"))
        new(glsl, kind, textures, uniforms, value_range)
    end
end
GLSLField(glsl; kind = :scalar, textures = (;), uniforms = (;), value_range = 0.0..1.0) =
    GLSLField(glsl, kind, textures, uniforms, value_range)

field_glsl(f::GLSLField) = f.glsl
value_kind(f::GLSLField) = f.kind
textures(f::GLSLField) = f.textures
uniforms(f::GLSLField) = f.uniforms
value_range(f::GLSLField) = f.value_range

# ── SurfaceMesh ──

"""    check_surface(surface::KeyedArray{<:Real,2})

Validate a 2-D height grid: at least 2 finite, strictly monotonic coordinates on each axis, and heights that are
finite or `NaN`. Returns `(x, y, heights)` as `Float64` arrays.
"""
function check_surface(surface)
    surface isa KeyedArray{<:Real,2} ||
        throw(ArgumentError("A surface must be a 2-D KeyedArray of heights, got $(summary(surface))"))
    x, y = map(k -> Vector{Float64}(k), axiskeys(surface))
    heights = Matrix{Float64}(AxisKeys.keyless_unname(surface))
    all(>=(2), size(heights)) || throw(ArgumentError("A surface needs at least 2 samples on each axis, got $(size(heights))"))
    for coordinates in (x, y)
        all(isfinite, coordinates) || throw(ArgumentError("Surface coordinates must be finite"))
        spacing = diff(coordinates)
        (all(>(0), spacing) || all(<(0), spacing)) ||
            throw(ArgumentError("Surface coordinates must be strictly increasing or decreasing"))
    end
    all(z -> isfinite(z) || isnan(z), heights) ||
        throw(ArgumentError("Surface heights must be finite or NaN"))
    6 * Int128(length(x) - 1) * (length(y) - 1) <= typemax(GL.GLsizei) ||
        throw(ArgumentError("The surface has too many triangles for one OpenGL draw"))
    x, y, heights
end

"""    SurfaceMesh(surface::KeyedArray{<:Real,2})

Indexed triangle mesh of a 2-D height grid for [`HeightSurface`](@ref); cells touching a `NaN` height are omitted.
Can be built off the render thread with no OpenGL context.
"""
struct SurfaceMesh
    surface::KeyedArray
    vertices::Matrix{Float32}
    indices::Vector{UInt32}
end

function SurfaceMesh(surface)
    x, y, heights = check_surface(surface)
    nx, ny = size(heights)
    vertices = Matrix{Float32}(undef, 3, length(heights))
    for (j, yj) in enumerate(y), (i, xi) in enumerate(x)
        k = i + (j - 1) * nx
        vertices[1, k] = xi
        vertices[2, k] = yj
        vertices[3, k] = heights[i, j]
    end
    indices = UInt32[]
    sizehint!(indices, 6 * (nx - 1) * (ny - 1))
    for j in 1:ny-1, i in 1:nx-1
        a = i + (j - 1) * nx
        b, c, d = a + 1, a + nx + 1, a + nx
        for triangle in ((a, b, c), (a, c, d))
            all(k -> isfinite(heights[k]), triangle) || continue
            for k in triangle
                push!(indices, UInt32(k - 1))
            end
        end
    end
    SurfaceMesh(surface, vertices, indices)
end
SurfaceMesh(m::SurfaceMesh) = m
Base.convert(::Type{SurfaceMesh}, s::KeyedArray) = SurfaceMesh(s)

function _surface_value_range(surface)
    lo, hi = Inf, -Inf
    for z in AxisKeys.keyless_unname(surface)
        isfinite(z) || continue
        lo, hi = min(lo, z), max(hi, z)
    end
    isfinite(lo) || return (0.0, 1.0)
    if lo == hi
        padding = max(abs(lo), 1.0) / 100
        return (lo - padding, hi + padding)
    end
    (Float64(lo), Float64(hi))
end

value_range(s::KeyedArray{<:Real,2}) = ((lo, hi) = _surface_value_range(s); lo..hi)
value_range(m::SurfaceMesh) = value_range(m.surface)

# ── built-in shapes ──

"""    Sphere(; center = SVector(0.0, 0.0, 0.0), radius)

A sphere shape.
"""
struct Sphere
    center::SVector{3,Float64}
    radius::Float64
end
Sphere(; center = SVector(0.0, 0.0, 0.0), radius) = Sphere(center, radius)

"""    Box(x, y, z)
    Box(g::GridField)

An axis-aligned box shape, from three `ClosedInterval`s (`Box(-1..1, -1..1, 0..2)`), or the box of the keys of a
Cartesian grid.
"""
struct Box
    x::ClosedInterval{Float64}
    y::ClosedInterval{Float64}
    z::ClosedInterval{Float64}
end

Box(g::Union{GridField,TiledField}) = grid_box(coordinates(g.data), g.data)
grid_box(::Cartesian, A) =
    Box(map(k -> convert(Float64, minimum(k))..convert(Float64, maximum(k)), values(named_axiskeys(A)[(:x, :y, :z)]))...)
grid_box(::LonLat, A) = throw(ArgumentError("Box(::GridField) needs a Cartesian (x, y, z) grid, got (lon, lat)"))

"""    Plane(origin, u, v)
    Plane(axis::Symbol, pos, box::Box)
    Plane(normal, pos, box::Box)

A flat parallelogram shape `{origin + s·u + t·v : s, t ∈ [0, 1]}`, the cross-section of `box` at `axis = pos`
(`axis` is `:x`, `:y` or `:z`), or the oblique plane `dot(normalize(normal), p) = pos` clipped to `box`.
"""
struct Plane
    origin::SVector{3,Float64}
    u::SVector{3,Float64}
    v::SVector{3,Float64}
end

# u, v span the two other axes in x, y, z order
function Plane(axis::Symbol, pos::Real, box::Box)
    i = findfirst(==(axis), (:x, :y, :z))
    isnothing(i) && throw(ArgumentError("Plane axis must be :x, :y or :z, got $(repr(axis))"))
    (; lo, hi) = aabb(box)
    edge(n) = setindex(zero(lo), hi[n] - lo[n], n)
    j, k = filter(!=(i), 1:3)
    Plane(setindex(lo, pos, i), edge(j), edge(k))
end

"""    Clipped(shape, box::Box)

The surface of `shape` clipped to `box`: a ray whose near hit with `shape` lies outside `box` falls back to its far
hit, so the back wall of a closed shape remains visible through a clipped front cap.
"""
struct Clipped{S}
    shape::S
    box::Box

    function Clipped(shape::S, box::Box) where {S}
        isshape(shape) || throw(ArgumentError("Clipped shape must implement the shape interface (shape_glsl, bounds, aabb), got $(summary(shape))"))
        new{S}(shape, box)
    end
end

function Plane(normal::Union{AbstractVector{<:Real},NTuple{3,Real}}, pos::Real, box::Box)
    n = SVector{3,Float64}(normal)
    mag = norm(n)
    isfinite(mag) && mag > 0 || throw(ArgumentError("Plane normal must be finite and nonzero, got $normal"))
    isfinite(pos) || throw(ArgumentError("Plane pos must be finite, got $pos"))
    n /= mag
    ref = abs(n[3]) < 0.9 ? SVector(0.0, 0.0, 1.0) : SVector(0.0, 1.0, 0.0)
    u = normalize(cross(n, ref))
    v = cross(n, u)
    (; lo, hi) = aabb(box)
    center = (lo + hi) / 2
    origin = center + n * (Float64(pos) - dot(n, center))
    halfsize = (hi - lo) / 2
    u *= dot(abs.(u), halfsize)
    v *= dot(abs.(v), halfsize)
    Clipped(Plane(origin - u - v, 2u, 2v), box)
end

uniforms(s::Sphere) = (; sphere_center = s.center, sphere_radius = s.radius)
uniforms(b::Box) = (; box_lo = aabb(b).lo, box_hi = aabb(b).hi)
uniforms(p::Plane) = (; plane_origin = p.origin, plane_u = p.u, plane_v = p.v)
uniforms(c::Clipped) = merge(uniforms(c.shape), (; clip_lo = aabb(c.box).lo, clip_hi = aabb(c.box).hi))

bounds(s::Sphere) = (; s.center, s.radius)
bounds(b::Box) = let (; lo, hi) = aabb(b); (; center = (lo + hi) / 2, radius = norm(hi - lo) / 2) end
# the longer diagonal, so that a non-rectangular parallelogram is enclosed too
bounds(p::Plane) = (; center = p.origin + (p.u + p.v) / 2, radius = max(norm(p.u + p.v), norm(p.u - p.v)) / 2)
bounds(c::Clipped) = let (; lo, hi) = aabb(c); (; center = (lo + hi) / 2, radius = norm(max.(hi - lo, 0.0)) / 2) end

aabb(s::Sphere) = (; lo = s.center .- s.radius, hi = s.center .+ s.radius)
aabb(b::Box) = (; lo = SVector(minimum.((b.x, b.y, b.z))), hi = SVector(maximum.((b.x, b.y, b.z))))
aabb(p::Plane) = let corners = (p.origin, p.origin + p.u, p.origin + p.v, p.origin + p.u + p.v)
    (; lo = min.(corners...), hi = max.(corners...))
end
aabb(c::Clipped) = let s = aabb(c.shape), b = aabb(c.box); (; lo = max.(s.lo, b.lo), hi = min.(s.hi, b.hi)) end

shape_glsl(::Sphere) = """
    vec2 vr_hit(vec3 ro, vec3 rd) {
        vec3 oc = ro - sphere_center; float b = dot(oc, rd), c = dot(oc, oc) - sphere_radius * sphere_radius;
        float disc = b * b - c; if (disc < 0.0) return vec2(1e30, -1e30);
        float s = sqrt(disc); return vec2(-b - s, -b + s);
    }
    """

# a zero rd component (axis-aligned ortho views) contributes no bound unless ro lies outside that slab
shape_glsl(::Box) = """
    vec2 vr_hit(vec3 ro, vec3 rd) {
        float tn = -1e30, tf = 1e30;
        for (int i = 0; i < 3; i++) {
            if (rd[i] == 0.0) { if (ro[i] < box_lo[i] || ro[i] > box_hi[i]) return vec2(1e30, -1e30); }
            else { float t0 = (box_lo[i] - ro[i]) / rd[i], t1 = (box_hi[i] - ro[i]) / rd[i];
                   tn = max(tn, min(t0, t1)); tf = min(tf, max(t0, t1)); }
        }
        return vec2(tn, tf);
    }
    """

shape_glsl(::Plane) = """
    vec2 vr_hit(vec3 ro, vec3 rd) {
        vec3 n = cross(plane_u, plane_v); float dn = dot(rd, n);
        if (dn == 0.0) return vec2(1e30, -1e30);
        float t = dot(plane_origin - ro, n) / dn; vec3 q = ro + rd * t - plane_origin;
        float s = dot(q, plane_u) / dot(plane_u, plane_u), w = dot(q, plane_v) / dot(plane_v, plane_v);
        return (s < 0.0 || s > 1.0 || w < 0.0 || w > 1.0) ? vec2(1e30, -1e30) : vec2(t, t);
    }
    """

shape_glsl(c::Clipped) = """
    $(replace(shape_glsl(c.shape), "vr_hit" => "vr_unclipped_hit"))
    vec2 vr_hit(vec3 ro, vec3 rd) {
        vec2 h = vr_unclipped_hit(ro, rd);
        if (h.y < h.x) return vec2(1e30, -1e30);
        float cn = -1e30, cf = 1e30;
        for (int i = 0; i < 3; i++) {
            if (rd[i] == 0.0) { if (ro[i] < clip_lo[i] || ro[i] > clip_hi[i]) return vec2(1e30, -1e30); }
            else { float t0 = (clip_lo[i] - ro[i]) / rd[i], t1 = (clip_hi[i] - ro[i]) / rd[i];
                   cn = max(cn, min(t0, t1)); cf = min(cf, max(t0, t1)); }
        }
        if (cf < cn) return vec2(1e30, -1e30);
        float t = h.x;
        if (t <= 0.0 || t < cn || t > cf) t = h.y;
        if (t <= 0.0 || t < cn || t > cf) return vec2(1e30, -1e30);
        return vec2(t, t);
    }
    """

# ── declared-name validation ──

# GLSL 4.10 §3.6: keywords, then words reserved for future use
const GLSL_RESERVED = Set(split("""
    attribute const uniform varying layout centroid flat smooth noperspective patch sample
    break continue do for while switch case default if else subroutine in out inout
    float double int void bool true false invariant discard return
    mat2 mat3 mat4 dmat2 dmat3 dmat4 mat2x2 mat2x3 mat2x4 dmat2x2 dmat2x3 dmat2x4
    mat3x2 mat3x3 mat3x4 dmat3x2 dmat3x3 dmat3x4 mat4x2 mat4x3 mat4x4 dmat4x2 dmat4x3 dmat4x4
    vec2 vec3 vec4 ivec2 ivec3 ivec4 bvec2 bvec3 bvec4 dvec2 dvec3 dvec4 uint uvec2 uvec3 uvec4
    lowp mediump highp precision
    sampler1D sampler2D sampler3D samplerCube sampler1DShadow sampler2DShadow samplerCubeShadow
    sampler1DArray sampler2DArray sampler1DArrayShadow sampler2DArrayShadow
    isampler1D isampler2D isampler3D isamplerCube isampler1DArray isampler2DArray
    usampler1D usampler2D usampler3D usamplerCube usampler1DArray usampler2DArray
    sampler2DRect sampler2DRectShadow isampler2DRect usampler2DRect samplerBuffer isamplerBuffer usamplerBuffer
    sampler2DMS isampler2DMS usampler2DMS sampler2DMSArray isampler2DMSArray usampler2DMSArray
    samplerCubeArray samplerCubeArrayShadow isamplerCubeArray usamplerCubeArray struct

    common partition active asm class union enum typedef template this packed goto inline noinline volatile
    public static extern external interface long short half fixed unsigned superp input output
    hvec2 hvec3 hvec4 fvec2 fvec3 fvec4 sampler3DRect filter
    image1D image2D image3D imageCube iimage1D iimage2D iimage3D iimageCube uimage1D uimage2D uimage3D uimageCube
    image1DArray image2DArray iimage1DArray iimage2DArray uimage1DArray uimage2DArray
    image1DShadow image2DShadow image1DArrayShadow image2DArrayShadow imageBuffer iimageBuffer uimageBuffer
    sizeof cast namespace using row_major
    precise
    """))   # `precise` is a 4.10 qualifier (§4.7) missing from the §3.6 list; compilers reject it as a name

# functions every program defines: field GLSL defines value and stepSize, the renderer main
const RENDERER_NAMES = ("value", "stepSize", "main")

function check_identifier(name::Symbol)
    s = String(name)
    occursin(r"\A[A-Za-z_][A-Za-z0-9_]*\z", s) || throw(ArgumentError("invalid GLSL identifier $(repr(s))"))
    startswith(s, "gl_") && throw(ArgumentError("GLSL identifier $(repr(s)) must not start with gl_ (reserved by GLSL)"))
    startswith(s, "vr_") && throw(ArgumentError("GLSL identifier $(repr(s)) must not start with vr_ (reserved by the renderer)"))
    occursin("__", s) && throw(ArgumentError("GLSL identifier $(repr(s)) must not contain __ (reserved by GLSL)"))
    s in RENDERER_NAMES && throw(ArgumentError("GLSL identifier $(repr(s)) is defined by the field GLSL or the renderer"))
    s in GLSL_RESERVED && throw(ArgumentError("GLSL identifier $(repr(s)) is a GLSL keyword or reserved word"))
    name
end

declared_names(o) = Symbol[keys(textures(o))..., keys(uniforms(o))...]

# names declared through `textures`/`uniforms` of the objects assembled into one program: each valid, none twice
function check_declared_names(objects)
    foreach(objects) do o
        names = declared_names(o)
        foreach(check_identifier, names)
        allunique(names) || throw(ArgumentError("$(summary(o)) declares a GLSL name twice in its textures and uniforms: $names"))
    end
    foreach(pairs(objects)) do (i, a)
        foreach(objects[i+1:end]) do b
            common = intersect(declared_names(a), declared_names(b))
            isempty(common) || throw(ArgumentError("GLSL names $common declared by both $(summary(a)) and $(summary(b))"))
        end
    end
end
