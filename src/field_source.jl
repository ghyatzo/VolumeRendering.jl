# FieldSource interface + built-in Cartesian `KeyedArray` source + `GLSLField`.
#
# A FieldSource supplies the GLSL that samples the scalar field (`sampleField` + `stepSize`), the
# spatial `region` it lives in, a `value_range` for the default TF window, a `fingerprint` for the
# render key, and a GPU lifecycle: `upload_field` (once → a handle holding GL texture ids),
# `bind_field!` (per frame), `free_field!` (teardown). See SPEC "FieldSource".
#
# `abstract type FieldSource end` is an OPTIONAL base for user structs. A bare
# `KeyedArray{<:Real,3}` satisfies the interface directly (methods defined on it below).

using AxisKeys, StaticArrays

abstract type FieldSource end

# ---- interface (generic functions; methods below overload these) --------------------------------
# field_glsl(field)::String                    — defines `float sampleField(vec3 p)` + `float stepSize(vec3 p)`
# region(field)::Region
# value_range(field)::Tuple{Float64,Float64}   — (min,max) for the default TF window
# fingerprint(field)                           — hashable, for render_key
# upload_field(field)::Handle                  — once; allocates GL textures (units 0..13)
# bind_field!(handle, prog)                    — per frame; binds the handle's textures/uniforms
# free_field!(handle)                          — delete the handle's GL resources

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

value_range(A::KeyedArray{<:Real,3}) = Float64.(extrema(A))

# Identity fingerprint: the field is swapped (a new object set on the view), not mutated in place, to
# change what is displayed — so `objectid` distinguishes distinct fields without hashing the data.
fingerprint(A::KeyedArray{<:Real,3}) = objectid(A)

# Upload the grid as an R32F 3D texture (default CLAMP×3 wrap, LINEAR filter); `tex3d_r32f` does the
# Float32/contiguous conversion at the GL boundary.
upload_field(A::KeyedArray{<:Real,3}) = KeyedFieldGPU(tex3d_r32f(A))

bind_field!(g::KeyedFieldGPU, prog) = bind_sampler(prog, "fieldTex", 0, GL.GL_TEXTURE_3D, g.tex)

free_field!(g::KeyedFieldGPU) = GL.glDeleteTextures(1, Ref(g.tex))

# ================================================================================================
# GLSLField — ad-hoc / analytic source (no texture).
# ================================================================================================
struct GLSLField <: FieldSource
    glsl::String                       # defines sampleField + stepSize (+ its own uniforms)
    region::Region
    value_range::Tuple{Float64,Float64}
    bind                               # (prog)->nothing, run per frame; default `_->nothing`
end
GLSLField(glsl; region, value_range, bind = _ -> nothing) = GLSLField(glsl, region, value_range, bind)

field_glsl(f::GLSLField) = f.glsl
region(f::GLSLField) = f.region
value_range(f::GLSLField) = f.value_range
fingerprint(f::GLSLField) = hash(f.glsl)

struct GLSLFieldGPU
    bind
end
upload_field(f::GLSLField) = GLSLFieldGPU(f.bind)
bind_field!(g::GLSLFieldGPU, prog) = g.bind(prog)
free_field!(::GLSLFieldGPU) = nothing
