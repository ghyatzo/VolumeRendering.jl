# GLSL-bearing objects: textures, fields (§2), shapes (§3), and validation of the names they declare (§6.4).

# ── Texture (§2.1) ──

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

# ── field interface (§2) and the parts shared with shapes (§3) ──

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

# interface conformance: the required methods exist
isfield(x) = applicable(field_glsl, x) && applicable(value_kind, x)
isshape(x) = applicable(shape_glsl, x) && applicable(bounds, x) && applicable(aabb, x)

nan_extrema(itr) = ((lo, hi) = extrema(Iterators.filter(!isnan, itr)); convert(Float64, lo)..convert(Float64, hi))

# ── GridField (§2.2) ──

"""    GridField(A::KeyedArray; interpolation = :linear)

A field sampled from a grid. Dim names choose the coordinates, in any order:
- `(x, y, z)`: Cartesian; [`Box(g)`](@ref Box) is the box of its keys;
- `(lon, lat)`: directions from the origin, in radians (or dimensionless Unitful angles); longitude wraps around.
  As a Volume source its region must exclude the origin, where the march step shrinks to 0.

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

# smallest cell edge along an axis
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

# ── GLSLField (§2.3) ──

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

# ── built-in shapes (§3) ──

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

Box(g::GridField) = grid_box(coordinates(g.data), g.data)
grid_box(::Cartesian, A) =
    Box(map(k -> convert(Float64, minimum(k))..convert(Float64, maximum(k)), values(named_axiskeys(A)[(:x, :y, :z)]))...)
grid_box(::LonLat, A) = throw(ArgumentError("Box(::GridField) needs a Cartesian (x, y, z) grid, got (lon, lat)"))

"""    Plane(origin, u, v)
    Plane(axis::Symbol, pos, box::Box)

A flat parallelogram shape `{origin + s·u + t·v : s, t ∈ [0, 1]}`, or the cross-section of `box` at `axis = pos`
(`axis` is `:x`, `:y` or `:z`).
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

uniforms(s::Sphere) = (; sphere_center = s.center, sphere_radius = s.radius)
uniforms(b::Box) = (; box_lo = aabb(b).lo, box_hi = aabb(b).hi)
uniforms(p::Plane) = (; plane_origin = p.origin, plane_u = p.u, plane_v = p.v)

bounds(s::Sphere) = (; s.center, s.radius)
bounds(b::Box) = let (; lo, hi) = aabb(b); (; center = (lo + hi) / 2, radius = norm(hi - lo) / 2) end
# the longer diagonal, so that a non-rectangular parallelogram is enclosed too
bounds(p::Plane) = (; center = p.origin + (p.u + p.v) / 2, radius = max(norm(p.u + p.v), norm(p.u - p.v)) / 2)

aabb(s::Sphere) = (; lo = s.center .- s.radius, hi = s.center .+ s.radius)
aabb(b::Box) = (; lo = SVector(minimum.((b.x, b.y, b.z))), hi = SVector(maximum.((b.x, b.y, b.z))))
aabb(p::Plane) = let corners = (p.origin, p.origin + p.u, p.origin + p.v, p.origin + p.u + p.v)
    (; lo = min.(corners...), hi = max.(corners...))
end

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

# ── declared-name validation (§6.4) ──

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
