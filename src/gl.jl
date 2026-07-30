# Thin OpenGL helpers (shader compilation, texture upload, FBO, uniforms) for the viewer.
# CImGui-free so the whole render path is testable in a headless GLFW context.

import ModernGL as GL

const SHADER_DIR = joinpath(@__DIR__, "shaders")

glcheck(tag) = (e = GL.glGetError(); e != 0 && error("GL error 0x$(string(e, base=16)) @ $tag"); nothing)

function compile_shader(kind, src::AbstractString)
    s = GL.glCreateShader(kind)
    bytes = Vector{UInt8}(codeunits(src)); push!(bytes, 0x00)
    GC.@preserve bytes GL.glShaderSource(s, 1, Ref(pointer(bytes)), Ref{GL.GLint}(length(bytes) - 1))
    GL.glCompileShader(s)
    ok = Ref{GL.GLint}(0); GL.glGetShaderiv(s, GL.GL_COMPILE_STATUS, ok)
    if ok[] == 0
        len = Ref{GL.GLint}(0); GL.glGetShaderiv(s, GL.GL_INFO_LOG_LENGTH, len)
        buf = Vector{UInt8}(undef, len[]); GL.glGetShaderInfoLog(s, len[], C_NULL, buf)
        error("shader compile failed:\n" * String(buf))
    end
    s
end

# Splice `//#include "name.glsl"` lines with either an in-memory `includes[name]` snippet or, when
# absent, `shaders/name.glsl` from disk (one level — shared snippets don't nest), so fragment/vertex
# shaders share `cameraRay`/`sampleField`/`tf`/`writeDepth` verbatim. The `#version` line stays in the
# including file; snippets carry only declarations + functions.
resolve_includes(src::AbstractString; includes::Dict{String,String} = Dict{String,String}()) = replace(src,
    r"(?m)^[ \t]*//#include[ \t]+\"([^\"]+)\"[ \t]*$" =>
        m -> (name = String(match(r"\"([^\"]+)\"", m).captures[1]);
              haskey(includes, name) ? includes[name] : read(joinpath(SHADER_DIR, name), String)))
_shader_src(file) = resolve_includes(read(joinpath(SHADER_DIR, file), String))

# Compile + link a program from vertex/fragment shader source strings, splicing any `//#include`s
# through `resolve_includes` with the given in-memory `includes` map (falling back to disk).
function link_program_src(vert_src::AbstractString, frag_src::AbstractString; includes::Dict{String,String} = Dict{String,String}())
    vs = compile_shader(GL.GL_VERTEX_SHADER,   resolve_includes(vert_src; includes))
    fs = compile_shader(GL.GL_FRAGMENT_SHADER, resolve_includes(frag_src; includes))
    p = GL.glCreateProgram()
    GL.glAttachShader(p, vs); GL.glAttachShader(p, fs); GL.glLinkProgram(p)
    ok = Ref{GL.GLint}(0); GL.glGetProgramiv(p, GL.GL_LINK_STATUS, ok)
    if ok[] == 0
        len = Ref{GL.GLint}(0); GL.glGetProgramiv(p, GL.GL_INFO_LOG_LENGTH, len)
        buf = Vector{UInt8}(undef, len[]); GL.glGetProgramInfoLog(p, len[], C_NULL, buf)
        error("program link failed:\n" * String(buf))
    end
    GL.glDeleteShader(vs); GL.glDeleteShader(fs)
    p
end

# Build a program from vertex/fragment shader source files under shaders/.
link_program(vert_file::AbstractString, frag_file::AbstractString) =
    link_program_src(read(joinpath(SHADER_DIR, vert_file), String), read(joinpath(SHADER_DIR, frag_file), String))

# `GL_LINEAR` (default) so the shader reads the field with one hardware-filtered `texture()` fetch (the
# 8-tap manual blend is gone). Wrap modes are per-axis: `wrap[1]→S, wrap[2]→T, wrap[3]→R`. Default is
# CLAMP_TO_EDGE on all three (correct for a generic Cartesian grid). The (w,h,d) axis order maps S,T,R.
function tex3d_r32f(data::AbstractArray{<:Real,3};
                    wrap = (GL.GL_CLAMP_TO_EDGE, GL.GL_CLAMP_TO_EDGE, GL.GL_CLAMP_TO_EDGE),
                    filter = GL.GL_LINEAR)
    arr = data isa Array{Float32,3} ? data : Array{Float32,3}(data)   # contiguous for glTexImage3D
    w, h, d = size(arr)
    id = Ref{GL.GLuint}(0); GL.glGenTextures(1, id); GL.glBindTexture(GL.GL_TEXTURE_3D, id[])
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, filter)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, filter)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_S, wrap[1])
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_T, wrap[2])
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_R, wrap[3])
    GL.glTexImage3D(GL.GL_TEXTURE_3D, 0, GL.GL_R32F, w, h, d, 0, GL.GL_RED, GL.GL_FLOAT, arr)
    id[]
end

function tex2d_r32f(data0::AbstractMatrix{<:Real}; filter = GL.GL_NEAREST)
    data = data0 isa Matrix{Float32} ? data0 : Matrix{Float32}(data0)
    w, h = size(data)
    id = Ref{GL.GLuint}(0); GL.glGenTextures(1, id); GL.glBindTexture(GL.GL_TEXTURE_2D, id[])
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, filter)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, filter)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_R32F, w, h, 0, GL.GL_RED, GL.GL_FLOAT, data)
    id[]
end

# A 256×1 RGBA32F transfer-function texture (LINEAR so lookups interpolate between entries).
function tex_tf()
    id = Ref{GL.GLuint}(0); GL.glGenTextures(1, id); GL.glBindTexture(GL.GL_TEXTURE_2D, id[])
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, 256, 1, 0, GL.GL_RGBA, GL.GL_FLOAT, C_NULL)
    id[]
end
# data: 4×256 Float32 (r,g,b,a per column).
update_tf!(id, data::Matrix{Float32}) = (GL.glBindTexture(GL.GL_TEXTURE_2D, id);
    GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, 256, 1, GL.GL_RGBA, GL.GL_FLOAT, data))

# Offscreen render target: an RGBA8 color texture + an optional depth attachment, resized on demand.
# `depth=:texture` makes depth a *sampleable* GL_DEPTH_COMPONENT32F texture (COMPARE_MODE=NONE) so a
# later pass can read the scene depth — the geometry G-buffer. `depth=:none` is color-only (the volume
# target). (Verified on this GL 4.1-over-Metal: depth-texture FBO is complete and samples as raw depth.)
mutable struct Framebuffer
    fbo::GL.GLuint
    tex::GL.GLuint          # RGBA8 color
    depth::GL.GLuint        # depth texture id, or 0 when depthmode == :none
    depthmode::Symbol       # :none | :texture
    w::Int
    h::Int
end
function Framebuffer(w, h; depth::Symbol = :none)
    fb = Framebuffer(0, 0, 0, depth, 0, 0)
    r = Ref{GL.GLuint}(0); GL.glGenFramebuffers(1, r); fb.fbo = r[]
    resize!(fb, w, h)
    fb
end
function Base.resize!(fb::Framebuffer, w, h)
    (w == fb.w && h == fb.h) && return fb
    fb.tex   != 0 && GL.glDeleteTextures(1, Ref(fb.tex))
    fb.depth != 0 && GL.glDeleteTextures(1, Ref(fb.depth))
    t = Ref{GL.GLuint}(0); GL.glGenTextures(1, t); GL.glBindTexture(GL.GL_TEXTURE_2D, t[])
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, C_NULL)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fb.fbo)
    GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, t[], 0)
    dtex = GL.GLuint(0)
    if fb.depthmode === :texture
        d = Ref{GL.GLuint}(0); GL.glGenTextures(1, d); GL.glBindTexture(GL.GL_TEXTURE_2D, d[])
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH_COMPONENT32F, w, h, 0, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, C_NULL)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_COMPARE_MODE, GL.GL_NONE)   # sample raw depth, not shadow
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_2D, d[], 0)
        dtex = d[]
    end
    @assert GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE "incomplete FBO"
    fb.tex = t[]; fb.depth = dtex; fb.w = w; fb.h = h
    fb
end

# Uniform setters (look up by name each call — simplest; the shader has few uniforms).
uni_i(p, name, v)     = GL.glUniform1i(GL.glGetUniformLocation(p, name), Int32(v))
uni_f(p, name, v)     = GL.glUniform1f(GL.glGetUniformLocation(p, name), Float32(v))
uni_2f(p, name, x, y) = GL.glUniform2f(GL.glGetUniformLocation(p, name), Float32(x), Float32(y))
uni_2i(p, name, x, y) = GL.glUniform2i(GL.glGetUniformLocation(p, name), Int32(x), Int32(y))
uni_3f(p, name, x, y, z) = GL.glUniform3f(GL.glGetUniformLocation(p, name), Float32(x), Float32(y), Float32(z))
# 4×4 matrix uniform. `m` is column-major (Julia/StaticArrays native), matching GL's expectation.
# A `Vector{Float32}` is uploaded as-is (already collected once per frame — no re-copy per overlay).
uni_m4(p, name, m) = GL.glUniformMatrix4fv(GL.glGetUniformLocation(p, name), 1, GL.GL_FALSE,
                                           collect(Float32, m))
uni_m4(p, name, m::Vector{Float32}) = GL.glUniformMatrix4fv(GL.glGetUniformLocation(p, name), 1, GL.GL_FALSE, m)

# Bind `texid` (of `target`) to texture unit `unit` and point sampler uniform `name` at it.
function bind_sampler(p, name, unit, target, texid)
    GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
    GL.glBindTexture(target, texid)
    uni_i(p, name, unit)
end
