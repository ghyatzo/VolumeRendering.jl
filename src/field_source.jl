# FieldSource interface + built-in Cartesian `KeyedArray` source + `GLSLField`.
#
# A FieldSource supplies the GLSL that samples the scalar field (`sampleField` + `stepSize`), the
# spatial `region` it lives in, a `value_range` for the default TF window, a `fingerprint` for the
# render key, and a GPU lifecycle: `upload_field` (once → a handle holding GL texture ids),
# `bind_field!` (per frame), `free_field!` (teardown). See SPEC "FieldSource".
#
# `abstract type FieldSource end` is an OPTIONAL base for user structs. A bare
# `KeyedArray{<:Real,3}` satisfies the interface directly (methods defined on it below).

using AxisKeys, StaticArrays, ColorTypes

abstract type FieldSource end

# TF-colored scalar (colormap+opacity, all modes) vs. intrinsically-colored RGBA (own color, DVR-only).
# Drives which controls the UI shows AND which GLSL bridge `field_glsl_full` appends. Color sources
# override to `false`.
uses_transfer_function(field) = true

# ---- interface (generic functions; methods below overload these) --------------------------------
# field_glsl(field)::String                    — defines `float stepSize(vec3 p)` + the field's ONE native
#                                                sampler: `float sampleField(vec3 p)` (scalar) OR
#                                                `vec4 sample4(vec3 p)` (color). The renderer splices
#                                                `field_glsl_full`, which adds the sibling (see below).
# region(field)::Region
# value_range(field)::Tuple{Float64,Float64}   — (min,max) for the default TF window
# fingerprint(field)                           — hashable, for render_key
# upload_field(field)::Handle                  — once; allocates GL textures (units 0..13)
# bind_field!(handle, prog)                    — per frame; binds the handle's textures/uniforms
# free_field!(handle)                          — delete the handle's GL resources

# The march references BOTH `sample4` (DVR branch) and `sampleField` (MIP/avg branches), so every linked
# program must define both symbols even though a field supplies only its native one. Append the missing
# sibling per the field kind: a scalar field gets `sample4 = tf(sampleField)`; a color field gets a
# luminance `sampleField` stand-in (it renders DVR-only, but the reduction branches still compile).
field_glsl_full(field) = string(field_glsl(field), "\n",
    uses_transfer_function(field) ?
        "vec4 sample4(vec3 p){ return tf(sampleField(p)); }" :
        "float sampleField(vec3 p){ return dot(sample4(p).rgb, vec3(0.2126, 0.7152, 0.0722)); }",
    "\n")

# ================================================================================================
# Per-axis dispatch — the extension seam.
# ================================================================================================
# Returns a GLSL EXPRESSION mapping a world coordinate on this axis (`coordvar`) to a fractional
# 0-based cell index. Users overload `axis_index_glsl` for their own axis-key types (the extension seam).
axis_index_glsl(key::AbstractRange, coordvar::AbstractString)::String =
    "((" * coordvar * ") - " * repr(Float64(first(key))) * ") * " * repr(inv(Float64(step(key))))

# Smallest cell edge along an axis (world units), used to set the constant march step. For a uniform
# range that's |step|; for any other monotonic axis vector (e.g. a log-spaced custom type) it's the
# smallest consecutive spacing. Users may overload this for their axis type to skip the `collect`.
_axis_min_cell(kk::AbstractRange) = abs(step(kk))
_axis_min_cell(kk) = minimum(abs.(diff(kk)))

# ================================================================================================
# Built-in Cartesian source — methods on `KeyedArray{<:Real,3}`.
# ================================================================================================
# The scalar grid is uploaded as an R32F 3D texture. `axiskeys(A)[1]` (dim 1, column-major fastest)
# maps to texture S, `[2]`→T, `[3]`→R — matching `tex3d_r32f`'s (w,h,d)=size upload. So the coordvars
# are p.x, p.y, p.z for axes 1, 2, 3 respectively (no permutation).

struct KeyedFieldGPU
    tex::UInt32
end

function field_glsl(A::KeyedArray{<:Real,3})
    k = axiskeys(A)
    nx, ny, nz = size(A)
    ix = axis_index_glsl(k[1], "p.x")
    iy = axis_index_glsl(k[2], "p.y")
    iz = axis_index_glsl(k[3], "p.z")
    mincell = repr(Float64(minimum(_axis_min_cell(kk) for kk in k)))
    """
    uniform sampler3D fieldTex;
    uniform int interp;
    float sampleField(vec3 p){
        vec3 idx = vec3( $ix, $iy, $iz );
        vec3 dims = vec3($(repr(Float64(nx))), $(repr(Float64(ny))), $(repr(Float64(nz))));
        vec3 uvw = (idx + 0.5) / dims;
        if (interp == 0) uvw = (floor(uvw*dims) + 0.5) / dims;
        return texture(fieldTex, uvw).r;
    }
    float stepSize(vec3 p){ return $mincell; }
    """
end

region(A::KeyedArray{<:Real,3}) =
    (k = axiskeys(A); BoxRegion(SVector(Float64.(first.(k))...), SVector(Float64.(last.(k))...)))

# much faster than Base extrema — its tuple-accumulator mapreduce doesn't vectorize (julia#31442)
_extrema(A) = (minimum(A), maximum(A))

value_range(A::KeyedArray{<:Real,3}) = Float64.(_extrema(A))

# Identity fingerprint: the field is swapped (a new object set on the view), not mutated in place, to
# change what is displayed — so `objectid` distinguishes distinct fields without hashing the data.
fingerprint(A::KeyedArray{<:Real,3}) = objectid(A)

# Upload the grid as an R32F 3D texture (default CLAMP×3 wrap, LINEAR filter); `tex3d_r32f` does the
# Float32/contiguous conversion at the GL boundary.
upload_field(A::KeyedArray{<:Real,3}) = KeyedFieldGPU(tex3d_r32f(A))

bind_field!(g::KeyedFieldGPU, prog) = bind_sampler(prog, "fieldTex", 0, GL.GL_TEXTURE_3D, g.tex)

free_field!(g::KeyedFieldGPU) = GL.glDeleteTextures(1, Ref(g.tex))

# ================================================================================================
# Built-in colored source — methods on `KeyedArray{<:Colorant,3}`.
# ================================================================================================
# `sample4` returns the stored `(rgb, a)` verbatim — no colormap, no encoding assumption. Same uvw
# convention as the scalar source, RGBA texture instead of R32F. The luminance `sampleField` sibling
# (so MIP/avg compile) comes from `field_glsl_full`; the UI only offers DVR here.

uses_transfer_function(::KeyedArray{<:Colorant,3}) = false

function field_glsl(A::KeyedArray{<:Colorant,3})
    k = axiskeys(A)
    nx, ny, nz = size(A)
    ix = axis_index_glsl(k[1], "p.x")
    iy = axis_index_glsl(k[2], "p.y")
    iz = axis_index_glsl(k[3], "p.z")
    mincell = repr(Float64(minimum(_axis_min_cell(kk) for kk in k)))
    """
    uniform sampler3D fieldTex;
    uniform int interp;
    vec4 sample4(vec3 p){
        vec3 idx = vec3( $ix, $iy, $iz );
        vec3 dims = vec3($(repr(Float64(nx))), $(repr(Float64(ny))), $(repr(Float64(nz))));
        vec3 uvw = (idx + 0.5) / dims;
        if (interp == 0) uvw = (floor(uvw*dims) + 0.5) / dims;
        return texture(fieldTex, uvw);
    }
    float stepSize(vec3 p){ return $mincell; }
    """
end

region(A::KeyedArray{<:Colorant,3}) =
    (k = axiskeys(A); BoxRegion(SVector(Float64.(first.(k))...), SVector(Float64.(last.(k))...)))

value_range(A::KeyedArray{<:Colorant,3}) = (0.0, 1.0)   # no scalar to window; TF unused

fingerprint(A::KeyedArray{<:Colorant,3}) = objectid(A)

# Reuses KeyedFieldGPU's bind_field!/free_field! — only the upload differs (RGBA vs R32F).
upload_field(A::KeyedArray{<:Colorant,3}) = KeyedFieldGPU(tex3d_rgba(A))

# ================================================================================================
# GLSLField — ad-hoc / analytic source (no texture).
# ================================================================================================
struct GLSLField <: FieldSource
    glsl::String                       # defines stepSize + its native sampler (+ its own uniforms):
                                       # `sampleField` if mode==:scalar, `sample4` if mode==:rgba
    region::Region
    value_range::Tuple{Float64,Float64}
    mode::Symbol                       # :scalar (default) or :rgba — which sampler `glsl` provides
    bind                               # (prog)->nothing, run per frame; default `_->nothing`
end
function GLSLField(glsl; region, value_range, mode = :scalar, bind = _ -> nothing)
    mode in (:scalar, :rgba) || throw(ArgumentError("GLSLField mode must be :scalar or :rgba, got $(repr(mode))"))
    GLSLField(glsl, region, value_range, mode, bind)
end

# `glsl` supplies its native sampler; `field_glsl_full` appends the sibling per `uses_transfer_function`.
field_glsl(f::GLSLField) = f.glsl
uses_transfer_function(f::GLSLField) = f.mode === :scalar
region(f::GLSLField) = f.region
value_range(f::GLSLField) = f.value_range
fingerprint(f::GLSLField) = hash(f.glsl)

struct GLSLFieldGPU
    bind
end
upload_field(f::GLSLField) = GLSLFieldGPU(f.bind)
bind_field!(g::GLSLFieldGPU, prog) = g.bind(prog)
free_field!(::GLSLFieldGPU) = nothing

# ================================================================================================
# Tiled Cartesian source — render grids whose dimensions exceed the GPU's GL_MAX_3D_TEXTURE_SIZE at
# full resolution, by splitting the volume into a small grid of GL_TEXTURE_3D bricks.
#
# A plain KeyedArray{<:Real,3} is uploaded as ONE 3D texture (`tex3d_r32f`). OpenGL caps every
# dimension of a 3D texture at GL_MAX_3D_TEXTURE_SIZE (commonly 2048); any dimension above that
# leaves the texture *incomplete*, so every sample reads 0 and the volume renders black. Tiling
# splits each axis `d` into `k[d]` chunks so every brick dimension ≤ maxdim.
#
# THIS STEP defines only the pure, GPU-free tiling geometry: the type, how each brick's data is
# extracted from the source (with its shared boundary texel), and the index mapping the shader will
# mirror. The GL path (`field_glsl` / `upload_field` / `bind_field!`) arrives in later steps.
#
#   k[d] = tiles along axis d (1 = the axis is NOT tiled; it already fits)
#   L[d] = chunk = non-overlapping source indices owned by each brick
#   D[d] = per-brick texels along d = L[d] + 1 when tiled, else n[d]
#          (the +1 is a shared boundary texel so GL_LINEAR filtering blends seamlessly across bricks)
# ================================================================================================
struct TiledField <: FieldSource
    field
    k::NTuple{3,Int}   # tiles per axis
    L::NTuple{3,Int}   # chunk stride per axis
    D::NTuple{3,Int}   # per-brick texels per axis
end

function TiledField(field::KeyedArray{<:Real,3}; maxdim::Integer = 2048)
    n  = size(field)
    k  = ntuple(d -> n[d] > maxdim ? cld(n[d], maxdim) : 1, 3)
    L  = ntuple(d -> cld(n[d], k[d]), 3)
    D  = ntuple(d -> k[d] == 1 ? n[d] : L[d] + 1, 3)
    TiledField(field, k, L, D)
end

total_bricks(t::TiledField) = prod(t.k)
_axis_len(t::TiledField, d) = size(t.field, d)

# Map one axis' local brick texel `loc` (0..D[d]-1) to its 1-based global source index.
# Brick `b` (0..k[d]-1) owns the L[d] chunk starting at source index b*L[d] (0-based); texel loc == L[d]
# is the shared boundary: it is the first interior texel of the NEXT brick (so both bricks carry the
# same value there → seamless interpolation). The last brick clamps trailing texels to the final source
# index; those are never sampled by a valid field point (the BoxRegion bounds the march).
function _brick_axis_index(t::TiledField, d::Int, b::Int)
    nd = _axis_len(t, d)
    Dd = t.D[d]
    [clamp(b * t.L[d] + loc, 0, nd - 1) + 1 for loc in 0:Dd-1]
end

# Materialize the full D-sized brick `b` (0-based, matching the shader) as a plain Array{Float32,3}.
function brick_data(t::TiledField, b::NTuple{3,Int})
    A = parent(t.field)
    src = ntuple(d -> _brick_axis_index(t, d, b[d]), 3)
    B = Array{eltype(A)}(undef, t.D)
    for I in CartesianIndices(B)
        B[I] = A[ntuple(d -> src[d][I[d]], 3)...]
    end
    B
end

# Forward index map (the shader mirrors this exactly): a continuous source index `g` (0-based, e.g.
# `(coord - first) / step`) → `(brick b, local float l)`. `b` is clamped to [0, k-1], `l` to [0, D-1].
function _map_axis(t::TiledField, d::Int, g::Real)
    kd, Ld, Dd = t.k[d], t.L[d], t.D[d]
    b = clamp(floor(Int, g / Ld), 0, kd - 1)
    l = clamp(g - b * Ld, 0.0, Dd - 1.0)
    (b, l)
end

# ================================================================================================
# FieldSource interface for TiledField. Spatial extent and value range come from the inner grid
# (unchanged → the world BoxRegion, camera framing and transfer-function window are identical to the
# plain field). Only the GLSL/upload/bind parts differ (later steps).
# ================================================================================================
uses_transfer_function(t::TiledField) = uses_transfer_function(t.field)

region(t::TiledField) = region(t.field)

value_range(t::TiledField) = value_range(t.field)

# `objectid` alone already distinguishes fields (the renderer swaps whole field objects, never mutates
# in place); we add the tiling geometry so a single inner grid re-tiled differently gets a new key.
fingerprint(t::TiledField) = (:tiled, objectid(t.field), t.k, t.L, t.D)

# ================================================================================================
# GLSL generation for a tiled field.
#
# Contract for brick ordering (shared with upload/bind in the next step): bricks are numbered by a
# flat index `flat = b1 * (k2*k3) + b2 * k3 + b3`, in lexicographic (b1,b2,b3) order. Samplers are
# named `fieldTex_<flat>`. `_brick_coords` recovers (b1,b2,b3) from `flat`.
# ================================================================================================
function _brick_coords(t::TiledField, i::Int)
    b1 = div(i, t.k[2] * t.k[3]); r = i - b1 * (t.k[2] * t.k[3])
    b2 = div(r, t.k[3]);          b3 = r - b2 * t.k[3]
    (b1, b2, b3)
end

field_glsl(t::TiledField) = begin
    A  = t.field
    k  = axiskeys(A)
    coord = ("p.x", "p.y", "p.z")
    gsrc  = ntuple(d -> axis_index_glsl(k[d], coord[d]), 3)   # continuous source index (== (coord-first)/step)
    mincell = repr(Float64(minimum(_axis_min_cell(kk) for kk in k)))
    nB  = total_bricks(t)
    k2, k3 = t.k[2], t.k[3]

    ax(d) = begin
        Ld, Dd, kd = t.L[d], t.D[d], t.k[d]
        """
        float g$d = $((gsrc[d]));
        int   b$d = clamp(int(floor(g$d / float($Ld))), 0, $kd-1);
        float l$d = clamp(g$d - float(b$d) * float($Ld), 0.0, float($Dd-1.0));
        float u$d = (((interp == 0) ? floor(l$d) : l$d) + 0.5) / float($Dd);
        """
    end

    samplers = join(("uniform sampler3D fieldTex_$i;" for i in 0:nB-1), "\n")

    branches = IOBuffer()
    for i in 0:nB-1
        kw = i == 0 ? "if (bid == $i)" : "else if (bid == $i)"
        print(branches, "\t", kw, " return texture(fieldTex_$i, vec3(u1,u2,u3)).r;\n")
    end

    """
    $samplers
    uniform int interp;
    float sampleField(vec3 p){
    $(ax(1))
    $(ax(2))
    $(ax(3))
        int bid = b1 * ($(k2) * $k3) + b2 * $k3 + b3;
    $(String(take!(branches)))
        return 0.0;
    }
    float stepSize(vec3 p){ return $mincell; }
    """
end


struct TiledFieldGPU
    tex::Vector{UInt32}   # one 3D texture id per brick, flat order (matches `fieldTex_<flat>`)
end

upload_field(t::TiledField) =
    TiledFieldGPU([tex3d_r32f(brick_data(t, _brick_coords(t, i))) for i in 0:total_bricks(t)-1])

function bind_field!(g::TiledFieldGPU, prog)
    for (i, id) in enumerate(g.tex)
        bind_sampler(prog, "fieldTex_$(i - 1)", i - 1, GL.GL_TEXTURE_3D, id)
    end
    g
end

free_field!(g::TiledFieldGPU) = (for id in g.tex; GL.glDeleteTextures(1, Ref(id)); end)

# ================================================================================================
# Public constructor: `TiledFieldView(field)` returns an ordinary FieldView whose field is a
# TiledField, so it renders oversized grids at full resolution. It auto-queries the hardware limit
# when a GL context is current (falling back to a safe constant otherwise).
# ================================================================================================
function _query_3d_maxdim(; fallback::Integer = 2048)
    v = Ref{GL.GLint}(0)
    try
        GL.glGetIntegerv(GL.GL_MAX_3D_TEXTURE_SIZE, v)
        v[] > 0 ? Int(v[]) : fallback
    catch
        fallback
    end
end

TiledFieldView(field; maxdim::Integer = _query_3d_maxdim(), kwargs...) =
    FieldView(TiledField(field; maxdim = maxdim); kwargs...)

# idempotent: already a TiledField → just wrap it in a view.
TiledFieldView(t::TiledField; kwargs...) = FieldView(t; kwargs...)
