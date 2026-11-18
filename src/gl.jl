glcheck() = (e = GL.glGetError(); e == GL.GL_NO_ERROR || error("GL error 0x$(string(e; base = 16))"); nothing)

gen(f) = (r = Ref{GL.GLuint}(0); f(1, r); r[])

# ── programs ──

function info_log(getiv, getlog, obj)
    len = Ref{GL.GLint}(0); getiv(obj, GL.GL_INFO_LOG_LENGTH, len)
    buf = Vector{UInt8}(undef, max(len[], 1)); getlog(obj, length(buf), C_NULL, buf)
    String(buf[1:end-1])
end

function compile_shader(kind, src::String)
    s = GL.glCreateShader(kind)
    GC.@preserve src GL.glShaderSource(s, 1, Ref(pointer(src)), Ref(GL.GLint(ncodeunits(src))))
    GL.glCompileShader(s)
    ok = Ref{GL.GLint}(0); GL.glGetShaderiv(s, GL.GL_COMPILE_STATUS, ok)
    ok[] == GL.GL_TRUE && return s
    log = info_log(GL.glGetShaderiv, GL.glGetShaderInfoLog, s)
    GL.glDeleteShader(s)
    error("shader compilation failed:\n$log")
end

# `locations`: every declared uniform (samplers included) → location, -1 when the linker dropped it (GL ignores writes
# to -1). `samplers[i]` reads texture unit i-1, set once at link.
struct Program
    id::GL.GLuint
    locations::Dict{Symbol,GL.GLint}
    samplers::Vector{Symbol}
end

function Program(vert::String, frag::String, uniforms::Vector{Symbol}, samplers::Vector{Symbol})
    vs = compile_shader(GL.GL_VERTEX_SHADER, vert)
    fs = try compile_shader(GL.GL_FRAGMENT_SHADER, frag) catch; GL.glDeleteShader(vs); rethrow() end
    p = GL.glCreateProgram()
    GL.glAttachShader(p, vs); GL.glAttachShader(p, fs); GL.glLinkProgram(p)
    GL.glDeleteShader(vs); GL.glDeleteShader(fs)
    ok = Ref{GL.GLint}(0); GL.glGetProgramiv(p, GL.GL_LINK_STATUS, ok)
    if ok[] != GL.GL_TRUE
        log = info_log(GL.glGetProgramiv, GL.glGetProgramInfoLog, p)
        GL.glDeleteProgram(p)
        error("program link failed:\n$log")
    end
    locations = Dict(n => GL.glGetUniformLocation(p, String(n)) for n in [uniforms; samplers])
    foreach(((unit, n),) -> GL.glProgramUniform1i(p, locations[n], unit - 1), enumerate(samplers))
    Program(p, locations, samplers)
end

gl_delete(p::Program) = GL.glDeleteProgram(p.id)

# ── textures ──

# One Julia texel type per GL format, used for upload, allocation and readback alike.
struct Depth32    # texel of a DEPTH_COMPONENT32F texture (a `Float32` texel is R32F)
    d::Float32
end

internal_format(::Type{Float32}) = GL.GL_R32F
internal_format(::Type{RGBA{N0f8}}) = GL.GL_RGBA8
internal_format(::Type{RGBA{Float16}}) = GL.GL_RGBA16F
internal_format(::Type{RGBA{Float32}}) = GL.GL_RGBA32F
internal_format(::Type{Depth32}) = GL.GL_DEPTH_COMPONENT32F

pixel_format(::Type{Float32}) = GL.GL_RED
pixel_format(::Type{<:RGBA}) = GL.GL_RGBA
pixel_format(::Type{Depth32}) = GL.GL_DEPTH_COMPONENT

pixel_type(::Type{<:Union{Float32,RGBA{Float32},Depth32}}) = GL.GL_FLOAT
pixel_type(::Type{RGBA{N0f8}}) = GL.GL_UNSIGNED_BYTE
pixel_type(::Type{RGBA{Float16}}) = GL.GL_HALF_FLOAT

struct GLTexture{T,N}
    id::GL.GLuint
    size::NTuple{N,Int}
end

gl_target(::GLTexture{<:Any,1}) = GL.GL_TEXTURE_1D
gl_target(::GLTexture{<:Any,2}) = GL.GL_TEXTURE_2D
gl_target(::GLTexture{<:Any,3}) = GL.GL_TEXTURE_3D

tex_image(target, ifmt, sz::NTuple{1,Int}, fmt, typ, data) = GL.glTexImage1D(target, 0, ifmt, sz..., 0, fmt, typ, data)
tex_image(target, ifmt, sz::NTuple{2,Int}, fmt, typ, data) = GL.glTexImage2D(target, 0, ifmt, sz..., 0, fmt, typ, data)
tex_image(target, ifmt, sz::NTuple{3,Int}, fmt, typ, data) = GL.glTexImage3D(target, 0, ifmt, sz..., 0, fmt, typ, data)

# `data`: the texels, or C_NULL to allocate uninitialized storage
function GLTexture{T}(size::NTuple{N,Int}, data::Union{Array{T,N},Ptr{Cvoid}}) where {T,N}
    tex = GLTexture{T,N}(gen(GL.glGenTextures), size)
    GL.glBindTexture(gl_target(tex), tex.id)
    GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)   # otherwise `data` is read as an offset into a bound buffer
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    foreach(p -> GL.glPixelStorei(p, 0), (GL.GL_UNPACK_ROW_LENGTH, GL.GL_UNPACK_SKIP_PIXELS, GL.GL_UNPACK_SKIP_ROWS,
                                          GL.GL_UNPACK_SKIP_IMAGES, GL.GL_UNPACK_IMAGE_HEIGHT))
    tex_image(gl_target(tex), internal_format(T), size, pixel_format(T), pixel_type(T), data)
    tex
end

# element type → texel type: Real → R32F, fixed-point colours → RGBA8, other colours → RGBA32F
texel_array(data::AbstractArray{<:Real,N}) where {N} = convert(Array{Float32,N}, data)
texel_array(data::AbstractArray{<:Colorant{<:FixedPoint},N}) where {N} = convert(Array{RGBA{N0f8},N}, data)
texel_array(data::AbstractArray{<:Colorant,N}) where {N} = convert(Array{RGBA{Float32},N}, data)

# dim 1 → S, 2 → T, 3 → R, no permutation
upload(data::AbstractArray) = (a = texel_array(data); GLTexture{eltype(a)}(size(a), a))

function read_texture(tex::GLTexture{T}) where {T}
    buf = Array{T}(undef, tex.size)
    GL.glBindTexture(gl_target(tex), tex.id)
    GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
    GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
    foreach(p -> GL.glPixelStorei(p, 0), (GL.GL_PACK_ROW_LENGTH, GL.GL_PACK_SKIP_PIXELS, GL.GL_PACK_SKIP_ROWS,
                                          GL.GL_PACK_SKIP_IMAGES, GL.GL_PACK_IMAGE_HEIGHT))
    GL.glGetTexImage(gl_target(tex), 0, pixel_format(T), pixel_type(T), buf)
    buf
end

gl_delete(t::GLTexture) = GL.glDeleteTextures(1, Ref(t.id))

# ── sampler objects: all sampling state; textures carry none ──

const WRAP = (clamp = GL.GL_CLAMP_TO_EDGE, repeat = GL.GL_REPEAT, mirror = GL.GL_MIRRORED_REPEAT)
const FILTER = (linear = GL.GL_LINEAR, nearest = GL.GL_NEAREST)

struct GLSampler
    id::GL.GLuint
end

function GLSampler(wrap::NTuple{N,Symbol}, filter::Symbol) where {N}
    s = GLSampler(gen(GL.glGenSamplers))
    GL.glSamplerParameteri(s.id, GL.GL_TEXTURE_MIN_FILTER, FILTER[filter])
    GL.glSamplerParameteri(s.id, GL.GL_TEXTURE_MAG_FILTER, FILTER[filter])
    foreach(((p, w),) -> GL.glSamplerParameteri(s.id, p, WRAP[w]),
            zip((GL.GL_TEXTURE_WRAP_S, GL.GL_TEXTURE_WRAP_T, GL.GL_TEXTURE_WRAP_R), wrap))
    s
end

gl_delete(s::GLSampler) = GL.glDeleteSamplers(1, Ref(s.id))

function bind_texture(unit::Integer, tex::GLTexture, sampler::GLSampler)
    GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
    GL.glBindTexture(gl_target(tex), tex.id)
    GL.glBindSampler(unit, sampler.id)
end

# ── instance buffers: one isbits record per instance, field i → attribute location i-1, divisor 1 ──

components(::Type{Float32}) = 1
components(::Type{SVector{N,Float32}}) where {N} = N
components(::Type{RGB{Float32}}) = 3

struct InstanceBuffer
    vao::GL.GLuint
    vbo::GL.GLuint
    n::Int
end

function InstanceBuffer(records::Vector{T}) where {T}
    buf = InstanceBuffer(gen(GL.glGenVertexArrays), gen(GL.glGenBuffers), length(records))
    GL.glBindVertexArray(buf.vao)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buf.vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, sizeof(records), records, GL.GL_STATIC_DRAW)
    foreach(1:fieldcount(T)) do i
        GL.glEnableVertexAttribArray(i - 1)
        GL.glVertexAttribPointer(i - 1, components(fieldtype(T, i)), GL.GL_FLOAT, GL.GL_FALSE, sizeof(T), Ptr{Cvoid}(fieldoffset(T, i)))
        GL.glVertexAttribDivisor(i - 1, 1)
    end
    GL.glBindVertexArray(0)   # keep later foreign attribute calls off this VAO
    buf
end

gl_delete(b::InstanceBuffer) = (GL.glDeleteVertexArrays(1, Ref(b.vao)); GL.glDeleteBuffers(1, Ref(b.vbo)))

# ── indexed triangle mesh buffers ──

struct MeshBuffer
    vao::GL.GLuint
    vbo::GL.GLuint
    ebo::GL.GLuint
    n::Int
end

function MeshBuffer(vertices::Matrix{Float32}, indices::Vector{UInt32})
    buf = MeshBuffer(gen(GL.glGenVertexArrays), gen(GL.glGenBuffers), gen(GL.glGenBuffers), length(indices))
    GL.glBindVertexArray(buf.vao)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buf.vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL.GL_STATIC_DRAW)
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, buf.ebo)
    GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), isempty(indices) ? C_NULL : indices, GL.GL_STATIC_DRAW)
    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 3 * sizeof(Float32), C_NULL)
    GL.glBindVertexArray(0)
    buf
end

gl_delete(b::MeshBuffer) = (GL.glDeleteVertexArrays(1, Ref(b.vao)); GL.glDeleteBuffers(1, Ref(b.vbo)); GL.glDeleteBuffers(1, Ref(b.ebo)))

# ── framebuffers ──

struct Framebuffer{T}
    fbo::GL.GLuint
    color::GLTexture{T,2}
    depth::Union{Nothing,GLTexture{Depth32,2}}
end

function Framebuffer{T}(size::NTuple{2,Int}; depth::Bool) where {T}
    color = GLTexture{T}(size, C_NULL)
    # the host displays the colour texture without a sampler object: give it a complete (non-mipmap) filter
    GL.glBindTexture(GL.GL_TEXTURE_2D, color.id)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    dtex = depth ? GLTexture{Depth32}(size, C_NULL) : nothing
    fb = Framebuffer{T}(gen(GL.glGenFramebuffers), color, dtex)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fb.fbo)
    GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, color.id, 0)
    depth && GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_2D, dtex.id, 0)
    status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
    status == GL.GL_FRAMEBUFFER_COMPLETE || error("incomplete framebuffer: status 0x$(string(status; base = 16))")
    fb
end

gl_delete(::Nothing) = nothing
gl_delete(fb::Framebuffer) = (GL.glDeleteFramebuffers(1, Ref(fb.fbo)); gl_delete(fb.color); gl_delete(fb.depth))

scene_framebuffer(w, h) = Framebuffer{RGBA{N0f8}}((w, h); depth = true)
# volume march output and the refinement mean: 16-bit float, since 8 bits would band faint layers before averaging
float_framebuffer(w, h) = Framebuffer{RGBA{Float16}}((w, h); depth = false)

# ── resource cache ──

# `hash` = objectid, `==` is `===`; holds `x` strongly so its id cannot be reused while the key lives
struct Identity{T}
    x::T
end
Base.:(==)(a::Identity, b::Identity) = a.x === b.x
Base.hash(a::Identity, h::UInt) = hash(objectid(a.x), h)

mutable struct ResourceCache
    const entries::Dict{Any,Any}   # key => Program | GLTexture | GLSampler | InstanceBuffer
    const empty_vao::GL.GLuint     # attribute-less draws still need a bound VAO in the core profile
    scene_fb::Framebuffer{RGBA{N0f8}}
    vol_fb::Framebuffer{RGBA{Float16}}
    accum_fb::Framebuffer{RGBA{Float16}}
end

ResourceCache((w, h), (vw, vh)) = ResourceCache(Dict{Any,Any}(), gen(GL.glGenVertexArrays),
                                                scene_framebuffer(w, h), float_framebuffer(vw, vh), float_framebuffer(w, h))

function fit_framebuffers!(cache::ResourceCache, (w, h), (vw, vh))
    cache.scene_fb.color.size == (w, h) || (gl_delete(cache.scene_fb); cache.scene_fb = scene_framebuffer(w, h))
    cache.vol_fb.color.size == (vw, vh) || (gl_delete(cache.vol_fb); cache.vol_fb = float_framebuffer(vw, vh))
    cache.accum_fb.color.size == (w, h) || (gl_delete(cache.accum_fb); cache.accum_fb = float_framebuffer(w, h))
    cache
end

# delete every entry whose key the frame plan does not list
function sweep!(cache::ResourceCache, needed)
    stale = filter(∉(needed), collect(keys(cache.entries)))
    foreach(k -> gl_delete(pop!(cache.entries, k)), stale)
    cache
end

function free!(cache::ResourceCache)
    foreach(gl_delete, values(cache.entries))
    empty!(cache.entries)
    foreach(gl_delete, (cache.scene_fb, cache.vol_fb, cache.accum_fb))
    GL.glDeleteVertexArrays(1, Ref(cache.empty_vao))
end
