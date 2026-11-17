# GLSL-bearing objects: textures, fields (§2), shapes (§3), and validation of the names they declare (§6.4).

# ── Texture (§2.1) ──

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

function field_glsl end
function value_kind end
textures(_) = (;)
uniforms(_) = (;)
value_range(_) = 0.0..1.0

function shape_glsl end
function bounds end
function aabb end

# interface conformance: the required methods exist
isfield(x) = applicable(field_glsl, x) && applicable(value_kind, x)
isshape(x) = applicable(shape_glsl, x) && applicable(bounds, x) && applicable(aabb, x)

nan_extrema(itr) = ((lo, hi) = extrema(Iterators.filter(!isnan, itr)); convert(Float64, lo)..convert(Float64, hi))

# ── GridField (§2.2) ──

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

# fractional 0-based cell index along `key` of the GLSL expression `coordvar`; extension seam for other axis types
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

struct Sphere
    center::SVector{3,Float64}
    radius::Float64
end
Sphere(; center = SVector(0.0, 0.0, 0.0), radius) = Sphere(center, radius)

struct Box
    x::ClosedInterval{Float64}
    y::ClosedInterval{Float64}
    z::ClosedInterval{Float64}
end

# the bounding box of a Cartesian grid's keys
Box(g::GridField) = grid_box(coordinates(g.data), g.data)
grid_box(::Cartesian, A) =
    Box(map(k -> convert(Float64, minimum(k))..convert(Float64, maximum(k)), values(named_axiskeys(A)[(:x, :y, :z)]))...)
grid_box(::LonLat, A) = throw(ArgumentError("Box(::GridField) needs a Cartesian (x, y, z) grid, got (lon, lat)"))

# the parallelogram {origin + s·u + t·v, s,t ∈ [0,1]}
struct Plane
    origin::SVector{3,Float64}
    u::SVector{3,Float64}
    v::SVector{3,Float64}
end

# the box's cross-section {axis = pos}; u, v span the two other axes in x, y, z order
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
