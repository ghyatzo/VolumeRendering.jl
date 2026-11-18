# ── uniforms ──

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

    // a 256-texel LUT holds f(i/255) at texel i: map u ∈ [0, 1] onto the texel centres
    vec4 vr_lut(sampler1D lut, float u) { return texture(lut, u * (255.0 / 256.0) + 0.5 / 256.0); }
    """

const RECT_VERT = """
    #version 410 core
    uniform vec4 vr_rect;   // NDC xmin, ymin, xmax, ymax
    void main() {
        gl_Position = vec4((gl_VertexID & 1) == 0 ? vr_rect.x : vr_rect.z, (gl_VertexID >> 1) == 0 ? vr_rect.y : vr_rect.w, 0.0, 1.0);
    }
    """

const FULLSCREEN = SVector(-1.0, -1.0, 1.0, 1.0)   # vr_rect of the Volume march and blits

# copies a framebuffer attachment onto the current target (composite, accumulate); linear upscaling of vol_fb
const BLIT_FRAG = """
    #version 410 core
    uniform sampler2D vr_src;
    uniform vec2 vr_viewport;
    out vec4 vr_frag;
    void main() { vr_frag = texture(vr_src, gl_FragCoord.xy / vr_viewport); }
    """

# hit selection per Surface variant; the near variant skips a Plane's single (coincident) hit
const SURFACE_HITS = (opaque = "h.x > 0.0 ? h.x : h.y", far = "h.y", near = "h.x == h.y ? -1.0 : h.x")

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

const VOLUME_DECLS = """
    uniform ivec2 vr_scene_size, vr_vol_size;
    uniform int vr_sample, vr_max_samples;
    uniform float vr_step_scale, vr_density;
    uniform sampler2D vr_scene_depth;
    """

# MIP and average have no intrinsic opacity: the layer alpha is the extinction LUT at the reduced value
mapped_layer(v) = "float u = vr_norm($v); float a = clamp(vr_alpha(u), 0.0, 1.0); vr_frag = vec4(vr_lut(vr_cmap, u).rgb * a, a);"

# per-mode parts of the march: accumulator declarations, per-sample update, output
const VOLUME_MODES = (
    dvr = (init = "vec3 col = vec3(0.0); float alpha = 0.0;",
           step = "vec4 c = vr_color(pos); float av = 1.0 - exp(-c.a * vr_density * dt); " *   # dt-correct opacity
                  "col += (1.0 - alpha) * av * c.rgb; alpha += (1.0 - alpha) * av; if (alpha > 0.995) break;",
           out = "vr_frag = vec4(col, alpha);"),
    mip = (init = "float vmax = -1e30;",   # fields may be negative
           step = "vmax = max(vmax, value(pos));",
           out = mapped_layer("vmax")),
    average = (init = "float vsum = 0.0, vlen = 0.0;",   # dt-weighted: the spacing varies
               step = "vsum += value(pos) * dt; vlen += dt;",
               out = mapped_layer("vsum / vlen")),
)

volume_body(mode::Symbol) = let (; init, step, out) = VOLUME_MODES[mode]
    """
    // the field's step at p, floored at dt_min; a comparison, not max(), so a NaN or non-positive step takes the floor too
    float vr_step(vec3 p, float dt_min) { float dt = vr_step_scale * stepSize(p); return dt > dt_min ? dt : dt_min; }
    void main() {
        vec3 ro, rd; vr_ray(gl_FragCoord.xy / vr_viewport * 2.0 - 1.0, ro, rd);
        vec2 tr = vr_hit(ro, rd);
        float tn = max(tr.x, 0.0), tfar = tr.y;   // the region may start behind the eye
        // every step is at least span / vr_max_samples, so the march crosses the region within vr_max_samples samples: a
        // ray that would need more is coarsened, not cut short. The span is before scene-depth clipping, so an opaque mark
        // in front does not change the sampling behind it
        float dt_min = (tfar - tn) / float(vr_max_samples);
        // clip at the nearest scene depth over this cell's full-res footprint, so sub-cell-thin lines are not missed
        ivec2 fc = ivec2(gl_FragCoord.xy);
        ivec2 b0 = fc * vr_scene_size / vr_vol_size, b1 = (fc + 1) * vr_scene_size / vr_vol_size;
        float d = 0.0;
        for (int y = b0.y; y < b1.y; y++)
            for (int x = b0.x; x < b1.x; x++)
                d = max(d, texelFetch(vr_scene_depth, ivec2(x, y), 0).r);
        if (d > 0.0) tfar = min(tfar, vr_near / d);
        if (tfar <= tn) { vr_frag = vec4(0.0); return; }   // missed or fully clipped
        // entry jitter: van der Corput over refinement frames, rotated per pixel
        float vdc = float(bitfieldReverse(uint(vr_sample))) * 2.3283064365386963e-10;
        float t = tn + fract(vdc + vr_hash12(uvec2(gl_FragCoord.xy))) * vr_step(ro + rd * tn, dt_min);
        if (t >= tfar) { vr_frag = vec4(0.0); return; }   // no sample: like a miss
        $init
        // the count bound only guards against float rounding: the step floor already ends the march within the budget
        for (int s = 0; s < vr_max_samples && t < tfar; s++) {
            vec3 pos = ro + rd * t;
            float dt = vr_step(pos, dt_min);
            $step
            t += dt;
        }
        $out
    }
    """
end

# per-vertex payload of a Lines program: its GLSL type; the endpoint values reach the fragment flat, as vr_va, vr_vb
payload_type(::Vector{<:Vector{<:Color}}) = "vec3"
payload_type(::Vector{<:Vector{<:Real}}) = "float"

payload_attrs(::Nothing) = ""
payload_attrs(p) = let T = payload_type(p)
    "layout(location = 2) in $T vr_pa;  layout(location = 3) in $T vr_pb;\nflat out $T vr_va, vr_vb;\n"
end
payload_assign(::Nothing) = ""
payload_assign(p) = "vr_va = mix(vr_pa, vr_pb, ua);  vr_vb = mix(vr_pa, vr_pb, ub);"

lines_vert(payload) = """
    #version 410 core
    layout(location = 0) in vec3 vr_a;  layout(location = 1) in vec3 vr_b;
    $(payload_attrs(payload))uniform mat4 vr_viewproj; uniform vec2 vr_viewport; uniform float vr_width_px;
    flat out vec2 vr_sa, vr_sb;  flat out vec4 vr_ha, vr_hb;
    void main() {
        int end = gl_VertexID >> 1; float side = (gl_VertexID & 1) == 0 ? -1.0 : 1.0;
        vec4 ca = vr_viewproj * vec4(vr_a, 1.0), cb = vr_viewproj * vec4(vr_b, 1.0);
        if (ca.w <= 0.0 && cb.w <= 0.0) { gl_Position = vec4(2.0, 2.0, 0.0, 1.0); return; }   // fully behind the eye
        // eye-plane clip of one behind endpoint at a relative epsilon of the other's w (scale-free): interpolated from the
        // front endpoint, and the clipped w set exactly — interpolating w cancels to garbage when the endpoint is far behind.
        // ua, ub: the drawn endpoints' parameters along the input segment, shared by the world position and the payload
        float ua = 0.0, ub = 1.0;
        if (ca.w <= 0.0)      { float s = (1.0 - 1e-4) * cb.w / (cb.w - ca.w); ca = vec4(cb.xyz + (ca.xyz - cb.xyz) * s, 1e-4 * cb.w); ua = 1.0 - s; }
        else if (cb.w <= 0.0) { float s = (1.0 - 1e-4) * ca.w / (ca.w - cb.w); cb = vec4(ca.xyz + (cb.xyz - ca.xyz) * s, 1e-4 * ca.w); ub = s; }
        vec3 wa = mix(vr_a, vr_b, ua), wb = mix(vr_a, vr_b, ub);
        vec2 sa = (ca.xy / ca.w * 0.5 + 0.5) * vr_viewport, sb = (cb.xy / cb.w * 0.5 + 0.5) * vr_viewport;   // pixel coords
        vec2 d = sb - sa; float len = length(d);
        vec2 dir = len > 1e-6 ? d / len : vec2(1.0, 0.0);          // zero-length / end-on segment: no NaN, draws a dot
        vec2 nrm = vec2(-dir.y, dir.x); float hw = 0.5 * vr_width_px;
        vec2 sp = (end == 0 ? sa - dir * hw : sb + dir * hw) + nrm * side * hw;
        gl_Position = vec4(sp / vr_viewport * 2.0 - 1.0, 0.0, 1.0);   // screen-space quad; every varying is flat
        vr_sa = sa;  vr_sb = sb;  vr_ha = vec4(wa, 1.0) / ca.w;  vr_hb = vec4(wb, 1.0) / cb.w;  $(payload_assign(payload))
    }
    """

const LINES_DECLS = "uniform float vr_width_px;\nflat in vec2 vr_sa, vr_sb;\nflat in vec4 vr_ha, vr_hb;\n"

const LINES_BODY = """
    void main() {
        vec2 q = gl_FragCoord.xy, ab = vr_sb - vr_sa; float l2 = dot(ab, ab);
        float s = l2 > 0.0 ? clamp(dot(q - vr_sa, ab) / l2, 0.0, 1.0) : 0.0;
        if (distance(q, vr_sa + s * ab) > 0.5 * vr_width_px) discard;        // capsule coverage
        // the segment point under the pixel, perspective-correct from the screen parameter (homogeneous endpoints)
        vec4 h = mix(vr_ha, vr_hb, s); vec3 p = h.xyz / h.w;
        float t = vr_t(p); if (t <= 0.0) discard;
        vr_frag = vec4(vr_color(p).rgb, 1.0);
        gl_FragDepth = vr_near / t;
    }
    """

const HEIGHT_VERT = """
    #version 410 core
    layout(location = 0) in vec3 vr_vert;
    uniform mat4 vr_viewproj;
    uniform vec3 vr_scale, vr_offset;
    uniform int vr_flip_depth;
    out vec3 vr_pos;
    out float vr_height;
    void main() {
        vr_height = vr_vert.z;
        vec3 p = vr_vert;
        if (vr_flip_depth == 1) p.z = -p.z;
        vr_pos = p * vr_scale + vr_offset;
        gl_Position = vr_viewproj * vec4(vr_pos, 1.0);
    }
    """

const HEIGHT_DECLS = """
    in vec3 vr_pos;
    in float vr_height;
    uniform int vr_has_clip, vr_shading;
    uniform vec3 vr_clip_lo, vr_clip_hi, vr_light;
    """

const HEIGHT_BODY = """
    void main() {
        vec3 normal = cross(dFdx(vr_pos), dFdy(vr_pos));
        if (vr_has_clip == 1 && (any(lessThan(vr_pos, vr_clip_lo)) || any(greaterThan(vr_pos, vr_clip_hi)))) discard;
        float t = vr_t(vr_pos); if (t <= 0.0) discard;
        vec3 col = vr_color(vr_pos).rgb;
        if (vr_shading == 1) {
            float nlen = length(normal);
            float diffuse = nlen > 0.0 ? abs(dot(normal / nlen, vr_light)) : 1.0;
            col *= 0.35 + 0.65 * diffuse;
        }
        vr_frag = vec4(col, 1.0);
        gl_FragDepth = vr_near / t;
    }
    """

# ── colour sources ──

# GLSL-bearing objects of a program, in declaration order (field before shape)
field_part(f) = (; object = f, textures = textures(f), uniforms = uniforms(f), glsl = field_glsl(f))
shape_part(s) = (; object = s, textures = (;), uniforms = uniforms(s), glsl = shape_glsl(s))

mapped_bridge(v) = "vec4 vr_color(vec3 p) { float u = vr_norm($v); return vec4(vr_lut(vr_cmap, u).rgb, vr_alpha(u)); }\n"

rgb32(c::Color) = (x = RGB{Float32}(c); SVector(x.r, x.g, x.b))

# a colour source's part of a program: GLSL-bearing objects, colormap, source-specific declarations, the bridge,
# its uniforms, and the per-vertex payload vector (Lines) or nothing
source_parts(bridge; objects = [], colormap = nothing, decls = "", uniforms = (;), payload = nothing) =
    (; objects, colormap, decls, bridge, uniforms, payload)

# the payload at the fragment's segment point p, interpolated by p's parameter along the segment: the same
# perspective-correct rule as p itself
const PAYLOAD_GLSL = """
    float vr_seg_u(vec3 p) {
        vec3 a = vr_ha.xyz / vr_ha.w, ab = vr_hb.xyz / vr_hb.w - a; float l2 = dot(ab, ab);
        return l2 > 0.0 ? clamp(dot(p - a, ab) / l2, 0.0, 1.0) : 0.0;
    }
    """
vertex_parts(payload, bridge; colormap = nothing) =
    source_parts(PAYLOAD_GLSL * bridge; colormap, decls = "flat in $(payload_type(payload)) vr_va, vr_vb;\n", payload)

color_source(c::Color) =
    source_parts("vec4 vr_color(vec3 p) { return vec4(vr_rgb, 1.0); }\n"; decls = "uniform vec3 vr_rgb;\n", uniforms = (; vr_rgb = rgb32(c)))
color_source(c::Vector{<:Vector{<:Color}}) = vertex_parts(c, "vec4 vr_color(vec3 p) { return vec4(mix(vr_va, vr_vb, vr_seg_u(p)), 1.0); }\n")
color_source(m::Mapped) = mapped_source(m.source, m.colormap)
color_source(c::Colormap) = source_parts(mapped_bridge("vr_height"); colormap = c)
color_source(f) = source_parts("vec4 vr_color(vec3 p) { return value(p); }\n"; objects = [field_part(f)])   # :color field (validated)

mapped_source(vals::Vector{<:Vector{<:Real}}, colormap) = vertex_parts(vals, mapped_bridge("mix(vr_va, vr_vb, vr_seg_u(p))"); colormap)
mapped_source(::Union{SurfaceMesh,KeyedArray{<:Real,2}}, colormap) = source_parts(mapped_bridge("vr_height"); colormap)
mapped_source(f, colormap) = source_parts(mapped_bridge("value(p)"); objects = [field_part(f)], colormap)

# ── colormap ──

colormap_lut(scheme::ColorScheme) = map(u -> RGBA{Float32}(get(scheme, u)), range(0.0, 1.0; length = 256))

colormap_decls(::Nothing) = ""
colormap_decls(::Colormap) = "uniform sampler1D vr_cmap;\nuniform float vr_cmap_lo, vr_cmap_hi;\n"

colormap_glsl(::Nothing, _) = ""
colormap_glsl(c::Colormap, extinction) = """
    float vr_norm(float v) { return clamp(($(c.log ? "log(max(v, 1e-30))" : "v") - vr_cmap_lo) / (vr_cmap_hi - vr_cmap_lo), 0.0, 1.0); }
    $(alpha_glsl(extinction))"""

# ── extinction: only a Volume with a Mapped colour has one ──

mapped_extinction(::Mapped, m::Volume) = m.extinction
mapped_extinction(_, ::Volume) = nothing

alpha_glsl(::Nothing) = "float vr_alpha(float u) { return 1.0; }\n"
alpha_glsl(_) = "float vr_alpha(float u) { return vr_lut(vr_extinction, u).r; }\n"

extinction_decls(::Nothing) = ""
extinction_decls(_) = "uniform sampler1D vr_extinction;\n"

# piecewise-linear σ(u) through the sorted points, constant beyond the end points
extinction_lut(pts) = map(range(0.0, 1.0; length = 256)) do u
    i = clamp(searchsortedlast(pts, u; by = first), 1, length(pts) - 1)
    (u0, s0), (u1, s1) = pts[i], pts[i+1]
    convert(Float32, u1 > u0 ? s0 + (s1 - s0) * clamp((u - u0) / (u1 - u0), 0, 1) : s1)
end

colormap_uniforms(::Nothing) = (;)
colormap_uniforms(c::Colormap) = let (lo, hi) = map(x -> c.log ? log(x) : x, endpoints(c.window))
    (; vr_cmap_lo = lo, vr_cmap_hi = hi)
end

# ── draws ──

# a framebuffer attachment of the view, bound like a texture; owned by the view, not a cache entry. Resolved at
# draw time: the framebuffers are (re)sized after the plan is built
struct Attachment
    get::Function   # ResourceCache → GLTexture
end

# one texture unit: sampler uniform name, texture (cache key => builder, or an Attachment), sampler-object key
struct Unit
    name::Symbol
    texture::Union{Pair,Attachment}
    sampler::Tuple{Tuple,Symbol}
end

texture_unit(name, t::Texture) = Unit(name, Identity(cache_key(t.data)) => () -> upload(cache_key(t.data)), (t.wrap, t.filter))
attachment_unit(name, get, filter) = Unit(name, Attachment(get), ((:clamp, :clamp), filter))

colormap_units(::Nothing) = Unit[]
colormap_units(c::Colormap) = [Unit(:vr_cmap, (:colormap, c.scheme) => () -> upload(colormap_lut(c.scheme)), ((:clamp,), :linear))]

extinction_units(::Nothing) = Unit[]
# the key holds a copy: it is compared by value, and an in-place edit must not change a stored key
extinction_units(e) = (pts = copy(e); [Unit(:vr_extinction, (:extinction, pts) => () -> upload(extinction_lut(pts)), ((:clamp,), :linear))])

# one draw call of the frame plan; marks assemble to `pass => Draw` steps, pass ∈ :opaque | :march | :composite | :translucent
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
    mapreduce(u -> texture_requirements(u.texture), vcat, d.units; init = []);
    map(u -> u.sampler => () -> GLSampler(u.sampler...), d.units);
    buffer_requirements(d.buffer)]

texture_requirements(t::Pair) = [t]
texture_requirements(::Attachment) = []

buffer_requirements(::Nothing) = []
buffer_requirements(b::Pair) = [b]

# fragment source, uniforms and units of one program; validates everything before any GL call
function assemble_fragment(color, objects, engine_decls, body; extinction = nothing, engine_units = Unit[])
    check_declared_names(map(o -> o.object, objects))
    texs = mapreduce(o -> o.textures, merge, objects; init = (;))
    unis = mapreduce(o -> o.uniforms, merge, objects; init = (;))
    units = [map(texture_unit, keys(texs), values(texs))...; colormap_units(color.colormap); extinction_units(extinction); engine_units]
    length(units) <= MAX_SAMPLERS ||
        throw(ArgumentError("a program needs $(length(units)) samplers, more than GL_MAX_TEXTURE_IMAGE_UNITS = $MAX_SAMPLERS"))
    decls = join([map((n, t) -> "uniform sampler$(ndims(t.data))D $n;\n", keys(texs), values(texs))...;
                  map(uniform_decl, keys(unis), values(unis))...;
                  colormap_decls(color.colormap); extinction_decls(extinction); color.decls; engine_decls])
    frag = join([PRELUDE, decls, map(o -> o.glsl, objects)..., colormap_glsl(color.colormap, extinction), color.bridge, body], "\n")
    (; frag, uniforms = merge(unis, colormap_uniforms(color.colormap), color.uniforms), units)
end

# prelude uniforms. Depth is reversed: surfaces and lines write gl_FragDepth = vr_near / t to a float buffer cleared to
# 0 (nothing drawn) and tested GEQUAL — ~2⁻²³ relative precision at every distance ≥ vr_near, independent of scene
# bounds; on a tie the later draw wins, so list order means "on top" in the opaque pass as in the translucent one
function frame_uniforms(cam::Camera, w, h)
    right, up, fwd = view_basis(cam)
    ortho = cam.projection === :ortho
    (; vr_perspective = ortho ? 0 : 1, vr_fovscale = fovscale(cam), vr_aspect = w / h,
       vr_eye = cam.eye, vr_right = right, vr_up = up, vr_fwd = fwd,
       vr_viewport = SVector(float(w), float(h)), vr_near = 1e-6 * (ortho ? cam.ortho_half : distance(cam)))
end

# NDC rectangle of the projected aabb corners, ±1 px; full screen when a corner is behind the eye
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

# `frame`: (; camera, w, h, vsize = volume march size, sample = refinement frame index); every method returns its steps

# opacity 1 → one opaque draw; else far hits, then near hits, blended
assemble(m::Surface, frame) = m.opacity == 1 ? [:opaque => surface_draw(m, :opaque, frame)] :
                                               [:translucent => surface_draw(m, :far, frame), :translucent => surface_draw(m, :near, frame)]

function surface_draw(m::Surface, variant, (; camera, w, h))
    color = color_source(m.color)
    (; frag, uniforms, units) = assemble_fragment(color, [color.objects; shape_part(m.shape)], "uniform float vr_opacity;\n",
                                                  surface_body(SURFACE_HITS[variant]))
    engine = (; vr_rect = impostor_rect(m.shape, camera, w, h), vr_opacity = m.opacity)
    Draw(RECT_VERT, frag, merge(frame_uniforms(camera, w, h), uniforms, engine), units, nothing)
end

function assemble(m::Lines, (; camera, w, h))
    color = color_source(m.color)
    (; frag, uniforms, units) = assemble_fragment(color, color.objects, LINES_DECLS, LINES_BODY)
    engine = (; vr_viewproj = view_proj(camera, w / h), vr_width_px = m.width_px)
    buffer = (Identity(m.polylines), payload_key(color.payload)) => () -> InstanceBuffer(segment_records(m.polylines, color.payload))
    [:opaque => Draw(lines_vert(color.payload), frag, merge(frame_uniforms(camera, w, h), uniforms, engine), units, buffer)]
end

function assemble(m::HeightSurface, (; camera, w, h))
    color = color_source(m.color)
    (; frag, uniforms, units) = assemble_fragment(color, color.objects, HEIGHT_DECLS, HEIGHT_BODY)
    right, up, fwd = view_basis(camera)
    clip_lo, clip_hi = isnothing(m.clip) ? (SVector(0.0, 0.0, 0.0), SVector(0.0, 0.0, 0.0)) : (aabb(m.clip).lo, aabb(m.clip).hi)
    engine = (; vr_viewproj = view_proj(camera, w / h), vr_scale = m.scale, vr_offset = m.offset,
                vr_flip_depth = m.flip_depth ? 1 : 0, vr_shading = m.shading ? 1 : 0,
                vr_has_clip = isnothing(m.clip) ? 0 : 1, vr_clip_lo = clip_lo, vr_clip_hi = clip_hi,
                vr_light = normalize(-fwd + 0.5 * right + 0.8 * up))
    buffer = Identity(m.mesh) => () -> MeshBuffer(m.mesh.vertices, m.mesh.indices)
    [:opaque => Draw(HEIGHT_VERT, frag, merge(frame_uniforms(camera, w, h), uniforms, engine), units, buffer)]
end

# march into vol_fb at vsize (same aspect w/h), then composite over the scene
function assemble(m::Volume, (; camera, w, h, vsize, sample))
    color = color_source(m.color)
    (; frag, uniforms, units) = assemble_fragment(color, [color.objects; shape_part(m.region)], VOLUME_DECLS, volume_body(m.mode);
                                                  extinction = mapped_extinction(m.color, m),
                                                  engine_units = [attachment_unit(:vr_scene_depth, c -> c.scene_fb.depth, :nearest)])
    engine = (; vr_rect = FULLSCREEN, vr_viewport = SVector(float.(vsize)...), vr_scene_size = SVector(w, h), vr_vol_size = SVector(vsize...),
                vr_sample = sample, vr_max_samples = m.max_samples, vr_step_scale = m.step_scale, vr_density = m.density)
    [:march => Draw(RECT_VERT, frag, merge(frame_uniforms(camera, w, h), uniforms, engine), units, nothing),
     :composite => blit(attachment_unit(:vr_src, c -> c.vol_fb.color, :linear), w, h)]
end

blit(src::Unit, w, h) = Draw(RECT_VERT, BLIT_FRAG, (; vr_rect = FULLSCREEN, vr_viewport = SVector(float(w), float(h))), [src], nothing)

payload_key(::Nothing) = nothing
payload_key(p) = Identity(p)

# Lines vertex buffer: one record per segment
segment_pairs(v) = zip(view(v, 1:length(v)-1), view(v, 2:length(v)))
f32(p) = convert(SVector{3,Float32}, p)
vertex_attr(c::Color) = RGB{Float32}(c)
vertex_attr(x::Real) = convert(Float32, x)

segment_records(polylines, ::Nothing) = [(; a = f32(a), b = f32(b)) for l in polylines for (a, b) in segment_pairs(l)]
segment_records(polylines, payload) =
    [(; a = f32(a), b = f32(b), pa = vertex_attr(x), pb = vertex_attr(y))
     for (l, p) in zip(polylines, payload) for ((a, b), (x, y)) in zip(segment_pairs(l), segment_pairs(p))]

# ── passes ──

gl_enable(cap, on::Bool) = on ? GL.glEnable(cap) : GL.glDisable(cap)

# the full state a pass depends on; programs and VAOs are set by every draw. `blend`: (src, dst) factors, or nothing
function pass_state!(w, h; depth_test::Bool, depth_write::Bool, blend = nothing, blend_alpha = 0)
    GL.glViewport(0, 0, w, h)
    GL.glDisable(GL.GL_SCISSOR_TEST); GL.glDisable(GL.GL_CULL_FACE)
    gl_enable(GL.GL_DEPTH_TEST, depth_test); GL.glDepthFunc(GL.GL_GEQUAL); GL.glDepthMask(depth_write)
    GL.glColorMask(true, true, true, true); GL.glDepthRange(0, 1)
    gl_enable(GL.GL_BLEND, !isnothing(blend)); GL.glBlendFunc(something(blend, (GL.GL_ONE, GL.GL_ZERO))...)
    GL.glBlendColor(0, 0, 0, blend_alpha); GL.glBlendEquation(GL.GL_FUNC_ADD)
    GL.glDisable(GL.GL_STENCIL_TEST); GL.glDisable(GL.GL_SAMPLE_ALPHA_TO_COVERAGE); GL.glDisable(GL.GL_MULTISAMPLE)
    GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
end

const OVER = (GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)   # premultiplied colour over the target

texture(cache::ResourceCache, (key, _)::Pair) = cache.entries[key]
texture(cache::ResourceCache, a::Attachment) = a.get(cache)

function draw!(cache::ResourceCache, d::Draw)
    prog = cache.entries[(d.vert, d.frag)]
    GL.glUseProgram(prog.id)
    foreach(((n, x),) -> set_uniform(prog.id, prog.locations[n], x), pairs(d.uniforms))
    foreach(((i, u),) -> bind_texture(i - 1, texture(cache, u.texture), cache.entries[u.sampler]), enumerate(d.units))
    draw_strip!(cache, d.buffer)
end

draw_strip!(cache, ::Nothing) = (GL.glBindVertexArray(cache.empty_vao); GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4))
draw_strip!(cache, (key, _)::Pair) = draw_buffer!(cache.entries[key])
draw_buffer!(b::InstanceBuffer) = (GL.glBindVertexArray(b.vao); GL.glDrawArraysInstanced(GL.GL_TRIANGLE_STRIP, 0, 4, b.n))
draw_buffer!(b::MeshBuffer) = b.n > 0 && (GL.glBindVertexArray(b.vao); GL.glDrawElements(GL.GL_TRIANGLES, b.n, GL.GL_UNSIGNED_INT, C_NULL))

function opaque_pass!(cache::ResourceCache, draws, background, w, h)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, cache.scene_fb.fbo)
    pass_state!(w, h; depth_test = true, depth_write = true)
    bg = RGB{Float32}(background)
    GL.glClearColor(bg.r, bg.g, bg.b, 1); GL.glClearDepth(0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    foreach(d -> draw!(cache, d), draws)
end

# one step of the translucent pass, in list order
translucent_step!(cache::ResourceCache, (pass, d)::Pair, w, h) = translucent_step!(Val(pass), cache, d, w, h)

function translucent_step!(::Val{:march}, cache, d, w, h)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, cache.vol_fb.fbo)
    pass_state!(cache.vol_fb.color.size...; depth_test = false, depth_write = false)
    GL.glClearColor(0, 0, 0, 0); GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    draw!(cache, d)
end
translucent_step!(::Val{:composite}, cache, d, w, h) = scene_step!(cache, d, w, h; depth_test = false)
translucent_step!(::Val{:translucent}, cache, d, w, h) = scene_step!(cache, d, w, h; depth_test = true)

function scene_step!(cache, d, w, h; depth_test)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, cache.scene_fb.fbo)
    pass_state!(w, h; depth_test, depth_write = false, blend = OVER)
    draw!(cache, d)
end

# fold the n-th frame into the running mean
function accumulate_pass!(cache::ResourceCache, d::Draw, n, w, h)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, cache.accum_fb.fbo)
    pass_state!(w, h; depth_test = false, depth_write = false,
                blend = n == 1 ? nothing : (GL.GL_CONSTANT_ALPHA, GL.GL_ONE_MINUS_CONSTANT_ALPHA), blend_alpha = 1 / n)
    draw!(cache, d)
end

# ── public entry points ──

refine_target(view::View) = any(jittered, view.marks) ? view.refine_frames : 1

volume_size(view::View, w, h) = let s = min(1, view.max_volume_px / max(w, h))
    (max(1, round(Int, w * s)), max(1, round(Int, h * s)))
end

# the running mean while refining, else the scene
displayed(view::View) = refine_target(view) > 1 ? view.cache.accum_fb.color : view.cache.scene_fb.color

"""    render!(view, w, h)

Render `view` at `w × h` device pixels into a GL texture owned by the view, and return the texture's id for the host
to display (rows bottom-up, as GL stores them). Needs a current OpenGL 4.1 core context; the host's framebuffer binding
is restored.

When nothing changed since the last call, returns the same texture without rendering; while volume refinement runs
(see [`View`](@ref)), each call adds one frame.
"""
function render!(view::View, w::Integer, h::Integer)
    w >= 1 && h >= 1 || throw(ArgumentError("render! needs w, h ≥ 1, got $w × $h"))
    validate(view); foreach(validate, view.marks)
    snap = snapshot(view, w, h)
    nmax = refine_target(view)
    if !isequal(snap, view.last)
        view.last = snap; view.nsamp = 0
    elseif view.nsamp >= nmax
        return displayed(view).id
    end
    vsize = volume_size(view, w, h)
    steps = mapreduce(m -> assemble(m, (; camera = view.camera, w, h, vsize, sample = view.nsamp)), vcat, view.marks; init = Pair{Symbol,Draw}[])
    accum = blit(attachment_unit(:vr_src, c -> c.scene_fb.color, :nearest), w, h)   # drawn only when refining
    draws = [last.(steps); nmax > 1 ? [accum] : Draw[]]
    needs = Dict{Any,Any}(Iterators.flatten(map(requirements, draws)))

    prev_fbo = Ref{GL.GLint}(0); GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING, prev_fbo)
    cache = @something(view.cache, view.cache = ResourceCache((w, h), vsize))
    sweep!(cache, keys(needs))   # before any upload: a swapped-out texture is freed before its replacement
    foreach(((k, make),) -> get!(make, cache.entries, k), needs)
    fit_framebuffers!(cache, (w, h), vsize)
    opaque_pass!(cache, map(last, filter(s -> first(s) === :opaque, steps)), view.background, w, h)
    foreach(s -> translucent_step!(cache, s, w, h), filter(s -> first(s) !== :opaque, steps))
    nmax > 1 && accumulate_pass!(cache, accum, view.nsamp + 1, w, h)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, prev_fbo[])
    GL.glBindVertexArray(0)   # keep later foreign attribute calls off our VAOs
    # the host samples the displayed texture with its own filters, not with a sampler object left on a unit
    foreach(unit -> GL.glBindSampler(unit, 0), 0:maximum(d -> length(d.units), draws; init = 0) - 1)
    glcheck()
    view.nsamp += 1
    displayed(view).id
end

# GL layout (x, y from the bottom) → standard image layout (row, col from the top)
image_layout(A::AbstractMatrix) = reverse(permutedims(A); dims = 1)

"""    render_image(view, w, h)::Matrix{RGB{N0f8}}

Render `view` at `w × h` pixels, refinement included, and read the image back: `h × w`, row 1 at the top. Needs a
current OpenGL 4.1 core context.
"""
function render_image(view::View, w::Integer, h::Integer)
    foreach(_ -> render!(view, w, h), 1:refine_target(view))
    image_layout(map(c -> RGB{N0f8}(c), read_texture(displayed(view))))
end

# internal, for tests: 0 = nothing drawn, else vr_near / t
depth_image(view::View, w::Integer, h::Integer) =
    (render!(view, w, h); image_layout(map(d -> d.d, read_texture(view.cache.scene_fb.depth))))

"""    free!(view)

Delete the view's GL resources, in its context; the next [`render!`](@ref) starts over. Optional: resources of marks
removed from a view are freed on its next rendered frame anyway.
"""
function free!(view::View)
    isnothing(view.cache) || free!(view.cache)
    view.cache = nothing; view.last = nothing; view.nsamp = 0
    nothing
end
