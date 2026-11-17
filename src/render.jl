# Program assembly (§6.4), the frame plan and passes (§6.3), and the public render entry points (§8).

# ── uniforms: the §2 type table ──

glsl_type(::Bool) = "bool"
glsl_type(::Integer) = "int"
glsl_type(::Real) = "float"
glsl_type(::SVector{N,<:Integer}) where {N} = N in 2:4 ? "ivec$N" : nothing
glsl_type(::SVector{N,<:Real}) where {N} = N in 2:4 ? "vec$N" : nothing
glsl_type(::SMatrix{3,3,<:Real}) = "mat3"
glsl_type(::SMatrix{4,4,<:Real}) = "mat4"
glsl_type(_) = nothing

uniform_decl(name, x) =
    "uniform $(@something(glsl_type(x), throw(ArgumentError("uniform $name: unsupported value type $(typeof(x))")))) $name;\n"

# ccall converts each component with `convert` (Float32 / GLint)
const UNIFORM_F = (GL.glProgramUniform1f, GL.glProgramUniform2f, GL.glProgramUniform3f, GL.glProgramUniform4f)
const UNIFORM_I = (GL.glProgramUniform1i, GL.glProgramUniform2i, GL.glProgramUniform3i, GL.glProgramUniform4i)

set_uniform(p, loc, x::Integer) = GL.glProgramUniform1i(p, loc, x)
set_uniform(p, loc, x::Real) = GL.glProgramUniform1f(p, loc, x)
set_uniform(p, loc, x::SVector{N,<:Integer}) where {N} = UNIFORM_I[N](p, loc, x...)
set_uniform(p, loc, x::SVector{N,<:Real}) where {N} = UNIFORM_F[N](p, loc, x...)
set_uniform(p, loc, x::SMatrix{3,3,<:Real}) = GL.glProgramUniformMatrix3fv(p, loc, 1, GL.GL_FALSE, Vector{Float32}(vec(x)))
set_uniform(p, loc, x::SMatrix{4,4,<:Real}) = GL.glProgramUniformMatrix4fv(p, loc, 1, GL.GL_FALSE, Vector{Float32}(vec(x)))

# ── fixed GLSL ──

const PRELUDE = """
    #version 410 core
    uniform int vr_perspective;
    uniform float vr_fovscale, vr_aspect;
    uniform vec3 vr_eye, vr_right, vr_up, vr_fwd;
    uniform vec2 vr_viewport;
    uniform float vr_near;
    out vec4 vr_frag;

    // ray through the pixel at NDC `ndc`; the exact inverse of view_proj
    void vr_ray(vec2 ndc, out vec3 ro, out vec3 rd) {
        if (vr_perspective == 1) {
            ro = vr_eye;
            rd = normalize(vr_fwd + ndc.x * vr_fovscale * vr_aspect * vr_right + ndc.y * vr_fovscale * vr_up);
        } else {
            rd = vr_fwd;
            ro = vr_eye + ndc.x * vr_fovscale * vr_aspect * vr_right + ndc.y * vr_fovscale * vr_up;
        }
    }
    float vr_t(vec3 p) { return vr_perspective == 1 ? length(p - vr_eye) : dot(p - vr_eye, vr_fwd); }

    // pcg2d hash (Jarzynski & Olano, JCGT 2020) → [0, 1)
    float vr_hash12(uvec2 p) {
        uvec2 v = p * 1664525u + 1013904223u;
        v.x += v.y * 1664525u;  v.y += v.x * 1664525u;
        v ^= v >> 16u;
        v.x += v.y * 1664525u;  v.y += v.x * 1664525u;
        v ^= v >> 16u;
        return float(v.x) * (1.0 / 4294967296.0);
    }
    """

const RECT_VERT = """
    #version 410 core
    uniform vec4 vr_rect;   // NDC xmin, ymin, xmax, ymax
    void main() {
        gl_Position = vec4((gl_VertexID & 1) == 0 ? vr_rect.x : vr_rect.z, (gl_VertexID >> 1) == 0 ? vr_rect.y : vr_rect.w, 0.0, 1.0);
    }
    """

surface_body(t) = """
    void main() {
        vec3 ro, rd; vr_ray(gl_FragCoord.xy / vr_viewport * 2.0 - 1.0, ro, rd);
        vec2 h = vr_hit(ro, rd); if (h.y < h.x) discard;
        float t = $t;
        if (t <= 0.0) discard;
        vec3 p = ro + rd * t;
        vr_frag = vec4(vr_color(p).rgb * vr_opacity, vr_opacity);
        gl_FragDepth = vr_near / t;
    }
    """

# per-vertex payload of a Lines program: GLSL type and varying name
varying(::Vector{<:Vector{<:Color}}) = ("vec3", "vr_vrgb")
varying(::Vector{<:Vector{<:Real}}) = ("float", "vr_vval")

payload_attrs(::Nothing) = ""
payload_attrs(p) = let (T, name) = varying(p)
    "layout(location = 2) in $T vr_pa;  layout(location = 3) in $T vr_pb;\nout $T $name;\n"
end
payload_assign(::Nothing) = ""
payload_assign(p) = "$(varying(p)[2]) = end == 0 ? vr_pa : vr_pb;"

lines_vert(payload) = """
    #version 410 core
    layout(location = 0) in vec3 vr_a;  layout(location = 1) in vec3 vr_b;
    $(payload_attrs(payload))uniform mat4 vr_viewproj; uniform vec2 vr_viewport; uniform float vr_width_px;
    out vec3 vr_pos;  flat out vec2 vr_sa, vr_sb;
    void main() {
        int end = gl_VertexID >> 1; float side = (gl_VertexID & 1) == 0 ? -1.0 : 1.0;
        vec4 ca = vr_viewproj * vec4(vr_a, 1.0), cb = vr_viewproj * vec4(vr_b, 1.0);
        vec3 wa = vr_a, wb = vr_b;
        if (ca.w <= 0.0 && cb.w <= 0.0) { gl_Position = vec4(2.0, 2.0, 0.0, 1.0); return; }   // fully behind the eye
        // eye-plane clip of one behind endpoint at a relative epsilon of the other's w (scale-free): interpolated from the
        // front endpoint, and the clipped w set exactly — interpolating w cancels to garbage when the endpoint is far behind
        if (ca.w <= 0.0)      { float s = (1.0 - 1e-4) * cb.w / (cb.w - ca.w); ca = vec4(cb.xyz + (ca.xyz - cb.xyz) * s, 1e-4 * cb.w); wa = wb + (wa - wb) * s; }
        else if (cb.w <= 0.0) { float s = (1.0 - 1e-4) * ca.w / (ca.w - cb.w); cb = vec4(ca.xyz + (cb.xyz - ca.xyz) * s, 1e-4 * ca.w); wb = wa + (wb - wa) * s; }
        vec2 sa = (ca.xy / ca.w * 0.5 + 0.5) * vr_viewport, sb = (cb.xy / cb.w * 0.5 + 0.5) * vr_viewport;   // pixel coords
        vec2 d = sb - sa; float len = length(d);
        vec2 dir = len > 1e-6 ? d / len : vec2(1.0, 0.0);          // zero-length / end-on segment: no NaN, draws a dot
        vec2 nrm = vec2(-dir.y, dir.x); float hw = 0.5 * vr_width_px;
        vec2 sp = (end == 0 ? sa - dir * hw : sb + dir * hw) + nrm * side * hw;
        vec4 cur = end == 0 ? ca : cb;
        gl_Position = vec4((sp / vr_viewport * 2.0 - 1.0) * cur.w, 0.0, cur.w);
        vr_pos = end == 0 ? wa : wb;  vr_sa = sa;  vr_sb = sb;  $(payload_assign(payload))
    }
    """

const LINES_DECLS = "uniform float vr_width_px;\nin vec3 vr_pos;\nflat in vec2 vr_sa, vr_sb;\n"

const LINES_BODY = """
    void main() {
        vec2 q = gl_FragCoord.xy, ab = vr_sb - vr_sa; float l2 = dot(ab, ab);
        float s = l2 > 0.0 ? clamp(dot(q - vr_sa, ab) / l2, 0.0, 1.0) : 0.0;
        if (distance(q, vr_sa + s * ab) > 0.5 * vr_width_px) discard;        // capsule coverage
        float t = vr_t(vr_pos); if (t <= 0.0) discard;
        vr_frag = vec4(vr_color(vr_pos).rgb, 1.0);
        gl_FragDepth = vr_near / t;
    }
    """

# ── colour sources (§4.3) ──

# GLSL-bearing objects of a program, in declaration order (field before shape)
field_part(f) = (; object = f, textures = textures(f), uniforms = uniforms(f), glsl = field_glsl(f))
shape_part(s) = (; object = s, textures = (;), uniforms = uniforms(s), glsl = shape_glsl(s))

mapped_bridge(v) = "vec4 vr_color(vec3 p) { float u = vr_norm($v); return vec4(texture(vr_cmap, u).rgb, vr_alpha(u)); }\n"

rgb32(c::Color) = (x = RGB{Float32}(c); SVector(x.r, x.g, x.b))

# a colour source's part of a program: GLSL-bearing objects, colormap, source-specific declarations, the bridge,
# its uniforms, and the per-vertex payload vector (Lines) or nothing
source_parts(bridge; objects = [], colormap = nothing, decls = "", uniforms = (;), payload = nothing) =
    (; objects, colormap, decls, bridge, uniforms, payload)

vertex_parts(payload, bridge; colormap = nothing) = let (T, name) = varying(payload)
    source_parts(bridge; colormap, decls = "in $T $name;\n", payload)
end

color_source(c::Color) =
    source_parts("vec4 vr_color(vec3 p) { return vec4(vr_rgb, 1.0); }\n"; decls = "uniform vec3 vr_rgb;\n", uniforms = (; vr_rgb = rgb32(c)))
color_source(c::Vector{<:Vector{<:Color}}) = vertex_parts(c, "vec4 vr_color(vec3 p) { return vec4(vr_vrgb, 1.0); }\n")
color_source(m::Mapped) = mapped_source(m.source, m.colormap)
color_source(f) = source_parts("vec4 vr_color(vec3 p) { return value(p); }\n"; objects = [field_part(f)])   # :color field (validated)

mapped_source(vals::Vector{<:Vector{<:Real}}, colormap) = vertex_parts(vals, mapped_bridge("vr_vval"); colormap)
mapped_source(f, colormap) = source_parts(mapped_bridge("value(p)"); objects = [field_part(f)], colormap)

# ── colormap (§4.1) ──

colormap_lut(scheme::ColorScheme) = map(u -> RGBA{Float32}(get(scheme, u)), range(0.0, 1.0; length = 256))

colormap_decls(::Nothing) = ""
colormap_decls(::Colormap) = "uniform sampler1D vr_cmap;\nuniform float vr_cmap_lo, vr_cmap_hi;\n"

colormap_glsl(::Nothing) = ""
colormap_glsl(c::Colormap) = """
    float vr_norm(float v) { return clamp(($(c.log ? "log(max(v, 1e-30))" : "v") - vr_cmap_lo) / (vr_cmap_hi - vr_cmap_lo), 0.0, 1.0); }
    float vr_alpha(float u) { return 1.0; }
    """

colormap_uniforms(::Nothing) = (;)
colormap_uniforms(c::Colormap) = let (lo, hi) = map(x -> c.log ? log(x) : x, endpoints(c.window))
    (; vr_cmap_lo = lo, vr_cmap_hi = hi)
end

# ── draws ──

const MAX_SAMPLERS = 16   # GL_MAX_TEXTURE_IMAGE_UNITS on macOS GL 4.1

# one texture unit: sampler uniform name, texture cache key and its builder, sampler-object key (wrap, filter)
struct Unit
    name::Symbol
    key::Any
    make::Any
    sampler::Tuple{Tuple,Symbol}
end

texture_unit(name, t::Texture) = Unit(name, Identity(cache_key(t.data)), () -> upload(cache_key(t.data)), (t.wrap, t.filter))

colormap_units(::Nothing) = Unit[]
colormap_units(c::Colormap) = [Unit(:vr_cmap, (:colormap, c.scheme), () -> upload(colormap_lut(c.scheme)), ((:clamp,), :linear))]

# one draw call of the frame plan
struct Draw
    vert::String
    frag::String
    uniforms::NamedTuple   # every non-sampler uniform, current values
    units::Vector{Unit}    # samplers in unit order
    buffer::Union{Nothing,Pair}   # nothing: attribute-less 4-vertex strip; `key => builder` of the instance buffer: instanced strip
end

# the cache entries a draw needs: key => builder
requirements(d::Draw) = [
    (d.vert, d.frag) => () -> Program(d.vert, d.frag, collect(keys(d.uniforms)), map(u -> u.name, d.units));
    map(u -> u.key => u.make, d.units);
    map(u -> u.sampler => () -> GLSampler(u.sampler...), d.units);
    buffer_requirements(d.buffer)]

buffer_requirements(::Nothing) = []
buffer_requirements(b::Pair) = [b]

# fragment source, uniforms and units of one program (§6.4); validates everything before any GL call
function assemble_fragment(color, objects, engine_decls, body)
    check_declared_names(map(o -> o.object, objects))
    texs = mapreduce(o -> o.textures, merge, objects; init = (;))
    unis = mapreduce(o -> o.uniforms, merge, objects; init = (;))
    units = [map(texture_unit, keys(texs), values(texs))...; colormap_units(color.colormap)]
    length(units) <= MAX_SAMPLERS ||
        throw(ArgumentError("a program needs $(length(units)) samplers, more than GL_MAX_TEXTURE_IMAGE_UNITS = $MAX_SAMPLERS"))
    decls = join([map((n, t) -> "uniform sampler$(ndims(t.data))D $n;\n", keys(texs), values(texs))...;
                  map(uniform_decl, keys(unis), values(unis))...;
                  colormap_decls(color.colormap); color.decls; engine_decls])
    frag = join([PRELUDE, decls, map(o -> o.glsl, objects)..., colormap_glsl(color.colormap), color.bridge, body], "\n")
    (; frag, uniforms = merge(unis, colormap_uniforms(color.colormap), color.uniforms), units)
end

# §6.1/6.2 prelude uniforms
function frame_uniforms(cam::Camera, w, h)
    right, up, fwd = view_basis(cam)
    ortho = cam.projection === :ortho
    (; vr_perspective = ortho ? 0 : 1, vr_fovscale = fovscale(cam), vr_aspect = w / h,
       vr_eye = cam.eye, vr_right = right, vr_up = up, vr_fwd = fwd,
       vr_viewport = SVector(float(w), float(h)), vr_near = 1e-6 * (ortho ? cam.ortho_half : distance(cam)))
end

# §6.7: NDC rectangle of the projected aabb corners, ±1 px; full screen when a corner is behind the eye
function impostor_rect(shape, cam::Camera, w, h)
    P = view_proj(cam, w / h)
    clip = map(c -> P * SVector(c..., 1.0), Iterators.product(zip(aabb(shape)...)...))   # the 8 corners
    any(c -> c[4] <= 0, clip) && return SVector(-1.0, -1.0, 1.0, 1.0)
    ndc = map(c -> SVector(c[1], c[2]) / c[4], clip)
    px = SVector(2 / w, 2 / h)
    lo = clamp.(reduce((a, b) -> min.(a, b), ndc), -1, 1) - px
    hi = clamp.(reduce((a, b) -> max.(a, b), ndc), -1, 1) + px
    SVector(lo..., hi...)
end

# opaque variant (§6.7); the far/near variants of translucent surfaces only change `t`
function assemble(m::Surface, cam::Camera, w, h)
    color = color_source(m.color)
    (; frag, uniforms, units) = assemble_fragment(color, [color.objects; shape_part(m.shape)], "uniform float vr_opacity;\n",
                                                  surface_body("h.x > 0.0 ? h.x : h.y"))
    engine = (; vr_rect = impostor_rect(m.shape, cam, w, h), vr_opacity = m.opacity)
    Draw(RECT_VERT, frag, merge(frame_uniforms(cam, w, h), uniforms, engine), units, nothing)
end

function assemble(m::Lines, cam::Camera, w, h)
    color = color_source(m.color)
    (; frag, uniforms, units) = assemble_fragment(color, color.objects, LINES_DECLS, LINES_BODY)
    engine = (; vr_viewproj = view_proj(cam, w / h), vr_width_px = m.width_px)
    buffer = (Identity(m.polylines), payload_key(color.payload)) => () -> InstanceBuffer(segment_records(m.polylines, color.payload))
    Draw(lines_vert(color.payload), frag, merge(frame_uniforms(cam, w, h), uniforms, engine), units, buffer)
end

payload_key(::Nothing) = nothing
payload_key(p) = Identity(p)

# §6.7 vertex buffer: one record per segment
segment_pairs(v) = zip(view(v, 1:length(v)-1), view(v, 2:length(v)))
f32(p) = convert(SVector{3,Float32}, p)
vertex_attr(c::Color) = RGB{Float32}(c)
vertex_attr(x::Real) = convert(Float32, x)

segment_records(polylines, ::Nothing) = [(; a = f32(a), b = f32(b)) for l in polylines for (a, b) in segment_pairs(l)]
segment_records(polylines, payload) =
    [(; a = f32(a), b = f32(b), pa = vertex_attr(x), pb = vertex_attr(y))
     for (l, p) in zip(polylines, payload) for ((a, b), (x, y)) in zip(segment_pairs(l), segment_pairs(p))]

# ── passes (§6.3) ──

gl_enable(cap, on::Bool) = on ? GL.glEnable(cap) : GL.glDisable(cap)

# the full state a pass depends on; programs and VAOs are set by every draw
function pass_state!(w, h; depth_test::Bool, depth_write::Bool)
    GL.glViewport(0, 0, w, h)
    GL.glDisable(GL.GL_SCISSOR_TEST); GL.glDisable(GL.GL_CULL_FACE)
    gl_enable(GL.GL_DEPTH_TEST, depth_test); GL.glDepthFunc(GL.GL_GEQUAL); GL.glDepthMask(depth_write)
    GL.glColorMask(true, true, true, true); GL.glDepthRange(0, 1)
    GL.glDisable(GL.GL_BLEND); GL.glBlendFunc(GL.GL_ONE, GL.GL_ZERO); GL.glBlendColor(0, 0, 0, 0); GL.glBlendEquation(GL.GL_FUNC_ADD)
    GL.glDisable(GL.GL_STENCIL_TEST); GL.glDisable(GL.GL_SAMPLE_ALPHA_TO_COVERAGE); GL.glDisable(GL.GL_MULTISAMPLE)
    GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
end

function draw!(cache::ResourceCache, d::Draw)
    prog = cache.entries[(d.vert, d.frag)]
    GL.glUseProgram(prog.id)
    foreach(((n, x),) -> set_uniform(prog.id, prog.locations[n], x), pairs(d.uniforms))
    foreach(((i, u),) -> bind_texture(i - 1, cache.entries[u.key], cache.entries[u.sampler]), enumerate(d.units))
    draw_strip!(cache, d.buffer)
end

draw_strip!(cache, ::Nothing) = (GL.glBindVertexArray(cache.empty_vao); GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4))
draw_strip!(cache, (key, _)::Pair) =
    (b = cache.entries[key]; GL.glBindVertexArray(b.vao); GL.glDrawArraysInstanced(GL.GL_TRIANGLE_STRIP, 0, 4, b.n))

function opaque_pass!(cache::ResourceCache, draws, background, w, h)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, cache.scene_fb.fbo)
    pass_state!(w, h; depth_test = true, depth_write = true)
    bg = RGB{Float32}(background)
    GL.glClearColor(bg.r, bg.g, bg.b, 1); GL.glClearDepth(0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    foreach(d -> draw!(cache, d), draws)
end

# ── public entry points (§8) ──

refine_target(view::View) = any(jittered, view.marks) ? view.refine_frames : 1

volume_size(view::View, w, h) = let s = min(1, view.max_volume_px / max(w, h))
    (max(1, round(Int, w * s)), max(1, round(Int, h * s)))
end

function render!(view::View, w::Integer, h::Integer)
    w >= 1 && h >= 1 || throw(ArgumentError("render! needs w, h ≥ 1, got $w × $h"))
    validate(view); foreach(validate, view.marks)
    snap = snapshot(view, w, h)
    if !isequal(snap, view.last)
        view.last = snap; view.nsamp = 0
    elseif view.nsamp >= refine_target(view)
        return view.cache.scene_fb.color.id
    end
    all(m -> m isa Lines || (m isa Surface && m.opacity == 1), view.marks) ||
        error("Volume and translucent Surface rendering is not implemented yet")
    draws = map(m -> assemble(m, view.camera, w, h), view.marks)
    needs = Dict{Any,Any}(Iterators.flatten(map(requirements, draws)))

    prev_fbo = Ref{GL.GLint}(0); GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING, prev_fbo)
    vsize = volume_size(view, w, h)
    cache = @something(view.cache, view.cache = ResourceCache((w, h), vsize))
    sweep!(cache, keys(needs))   # before any upload: a swapped-out texture is freed before its replacement
    foreach(((k, make),) -> get!(make, cache.entries, k), needs)
    fit_framebuffers!(cache, (w, h), vsize)
    opaque_pass!(cache, draws, view.background, w, h)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, prev_fbo[])
    GL.glBindVertexArray(0)   # keep later foreign attribute calls off our VAOs
    # the host samples the displayed texture with its own filters, not with a sampler object left on a unit
    foreach(unit -> GL.glBindSampler(unit, 0), 0:maximum(d -> length(d.units), draws; init = 0) - 1)
    glcheck()
    view.nsamp += 1
    cache.scene_fb.color.id
end

# GL layout (x, y from the bottom) → standard image layout (row, col from the top)
image_layout(A::AbstractMatrix) = reverse(permutedims(A); dims = 1)

function render_image(view::View, w::Integer, h::Integer)
    foreach(_ -> render!(view, w, h), 1:refine_target(view))
    image_layout(map(c -> RGB{N0f8}(c), read_texture(view.cache.scene_fb.color)))
end

# internal, for tests: 0 = nothing drawn, else vr_near / t
depth_image(view::View, w::Integer, h::Integer) =
    (render!(view, w, h); image_layout(map(d -> d.d, read_texture(view.cache.scene_fb.depth))))

function free!(view::View)
    isnothing(view.cache) || free!(view.cache)
    view.cache = nothing; view.last = nothing; view.nsamp = 0
    nothing
end
