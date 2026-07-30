# Concrete overlays for the geometry G-buffer. Each is an INDEPENDENT unit (its own GL program + draw)
# that writes opaque color + depth in the shared `viewProj` convention, so `GL_LESS` composes them and
# the volume clips against their union. The protocol (`abstract type Overlay`, `enabled`, `refresh!`,
# `draw!`, `overlay_fingerprint`) lives in volume_renderer.jl; here we only add concrete types + methods.
#
# GL-resource construction is done LAZILY inside `refresh!` (guarded by a 0/`nothing` handle), so an
# overlay can be constructed with no current GL context; the first `refresh!` under the render context
# builds its program/VAO. `overlay_fingerprint` returns a hashable tuple of ALL image-affecting state,
# feeding the FieldView render-on-demand key.

using StaticArrays

# A fresh (attribute-less) VAO — core profile still requires one bound to draw the generated geometry.
_new_vao() = (r = Ref{GL.GLuint}(0); GL.glGenVertexArrays(1, r); r[])

_axis_int(a::Symbol) = a === :x ? 0 : a === :y ? 1 : a === :z ? 2 :
    error("SliceOverlay axis must be :x, :y, or :z; got $(repr(a))")

# ── SliceOverlay: world-space quad on {axis = pos}, sampling the field, TF-colored ──
# slice.frag injects the field snippet, so the program is PER-FIELD and rebuilt when the field's GLSL
# changes (tracked by `built_for`).
mutable struct SliceOverlay <: Overlay
    axis::Int                       # 0 = x, 1 = y, 2 = z
    pos::Float64
    enabled::Bool
    prog::UInt32
    vao::UInt32
    built_for::Union{Nothing,String}
end
SliceOverlay(axis::Symbol; pos = 0.0, enabled = true) =
    SliceOverlay(_axis_int(axis), Float64(pos), enabled, UInt32(0), UInt32(0), nothing)

enabled(o::SliceOverlay) = o.enabled

function refresh!(o::SliceOverlay, ctx::GeomContext)
    o.vao == 0 && (o.vao = _new_vao())
    fg = field_glsl(ctx.field)
    if fg != o.built_for
        o.prog != 0 && GL.glDeleteProgram(o.prog)
        vsrc = read(joinpath(SHADER_DIR, "slice.vert"), String)
        fsrc = read(joinpath(SHADER_DIR, "slice.frag"), String)
        o.prog = link_program_src(vsrc, fsrc; includes = Dict{String,String}("field" => fg))
        o.built_for = fg
    end
    o
end

function draw!(o::SliceOverlay, ctx::GeomContext)
    lo, hi = aabb(ctx.region)
    GL.glUseProgram(o.prog); GL.glBindVertexArray(o.vao)
    uni_m4(o.prog, "viewProj", ctx.viewproj)
    uni_i(o.prog, "sliceAxis", o.axis)
    uni_f(o.prog, "slicePos", o.pos)
    uni_3f(o.prog, "boxLo", lo...); uni_3f(o.prog, "boxHi", hi...)
    uni_i(o.prog, "interp", ctx.params.interp)
    bind_field!(ctx.gpu, o.prog)
    bind_tf_uniforms!(o.prog, ctx.tf)
    GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
end

overlay_fingerprint(o::SliceOverlay) = (:slice, o.axis, o.pos, o.enabled)

# ── SphereOverlay: analytic ray–sphere impostor over the fullscreen triangle ──
mutable struct SphereOverlay <: Overlay
    center::NTuple{3,Float64}
    radius::Float64
    color::NTuple{3,Float64}
    enabled::Bool
    prog::UInt32
    vao::UInt32
end
SphereOverlay(; center = (0.0, 0.0, 0.0), radius, color = (0.06, 0.06, 0.09), enabled = true) =
    SphereOverlay(NTuple{3,Float64}(center), Float64(radius), NTuple{3,Float64}(color),
                  enabled, UInt32(0), UInt32(0))

enabled(o::SphereOverlay) = o.enabled

function refresh!(o::SphereOverlay, ::GeomContext)
    o.vao == 0 && (o.vao = _new_vao())
    o.prog == 0 && (o.prog = link_program("volume.vert", "sphere.frag"))
    o
end

function draw!(o::SphereOverlay, ctx::GeomContext)
    GL.glUseProgram(o.prog); GL.glBindVertexArray(o.vao)
    set_camera_uniforms!(o.prog, ctx.cam, ctx.aspect)
    uni_m4(o.prog, "viewProj", ctx.viewproj)
    uni_3f(o.prog, "sphereCenter", o.center...)
    uni_f(o.prog, "sphereRadius", o.radius)
    uni_3f(o.prog, "sphereColor", o.color...)
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
end

overlay_fingerprint(o::SphereOverlay) = (:sphere, o.center, o.radius, o.color, o.enabled)

# ── LineOverlay: screen-space constant-pixel-width line between two world endpoints ──
mutable struct LineOverlay <: Overlay
    a::SVector{3,Float64}
    b::SVector{3,Float64}
    width_px::Float64
    color::NTuple{3,Float64}
    enabled::Bool
    prog::UInt32
    vao::UInt32
end
LineOverlay(a, b; width_px = 2.0, color = (0.8, 0.8, 0.85), enabled = true) =
    LineOverlay(SVector{3,Float64}(a), SVector{3,Float64}(b), Float64(width_px),
                NTuple{3,Float64}(color), enabled, UInt32(0), UInt32(0))

enabled(o::LineOverlay) = o.enabled

function refresh!(o::LineOverlay, ::GeomContext)
    o.vao == 0 && (o.vao = _new_vao())
    o.prog == 0 && (o.prog = link_program("line.vert", "line.frag"))
    o
end

function draw!(o::LineOverlay, ctx::GeomContext)
    GL.glUseProgram(o.prog); GL.glBindVertexArray(o.vao)
    uni_m4(o.prog, "viewProj", ctx.viewproj)
    uni_3f(o.prog, "axisA", o.a...); uni_3f(o.prog, "axisB", o.b...)
    uni_2f(o.prog, "viewportPx", ctx.vieww, ctx.viewh)
    uni_f(o.prog, "lineWidthPx", o.width_px)
    uni_3f(o.prog, "lineColor", o.color...)
    GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
end

overlay_fingerprint(o::LineOverlay) = (:line, o.a, o.b, o.width_px, o.color, o.enabled)

# ── AxesOverlay: three colored screen-space lines through the region center spanning the region ──
mutable struct AxesOverlay <: Overlay
    enabled::Bool
    width_px::Float64
    colors::NTuple{3,NTuple{3,Float64}}
    prog::UInt32
    vao::UInt32
end
AxesOverlay(; enabled = true, width_px = 2.0,
            colors = ((1.0, 0.3, 0.3), (0.3, 1.0, 0.3), (0.4, 0.6, 1.0))) =
    AxesOverlay(enabled, Float64(width_px),
                NTuple{3,NTuple{3,Float64}}(map(c -> NTuple{3,Float64}(c), colors)),
                UInt32(0), UInt32(0))

enabled(o::AxesOverlay) = o.enabled

function refresh!(o::AxesOverlay, ::GeomContext)
    o.vao == 0 && (o.vao = _new_vao())
    o.prog == 0 && (o.prog = link_program("line.vert", "line.frag"))
    o
end

const _AXIS_UNITS = (SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 1.0))

function draw!(o::AxesOverlay, ctx::GeomContext)
    center, radius = bounds(ctx.region)
    GL.glUseProgram(o.prog); GL.glBindVertexArray(o.vao)
    uni_m4(o.prog, "viewProj", ctx.viewproj)
    uni_2f(o.prog, "viewportPx", ctx.vieww, ctx.viewh)
    uni_f(o.prog, "lineWidthPx", o.width_px)
    for i in 1:3
        ê = _AXIS_UNITS[i]
        a = center - radius * ê
        b = center + radius * ê
        uni_3f(o.prog, "axisA", a...); uni_3f(o.prog, "axisB", b...)
        uni_3f(o.prog, "lineColor", o.colors[i]...)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
    end
end

overlay_fingerprint(o::AxesOverlay) = (:axes, o.enabled, o.width_px, o.colors)

# ── BoxOutlineOverlay: the 12 edges of an AABB as GL_LINES (via a GeometryRenderer) ──
mutable struct BoxOutlineOverlay <: Overlay
    region::Union{Nothing,Region}   # explicit region, else ctx.region at draw time
    color::NTuple{3,Float64}
    enabled::Bool
    gr::Union{Nothing,GeometryRenderer}
    last::Any                        # (lo, hi, color) the current vertices were built for
end
BoxOutlineOverlay(; region = nothing, color = (0.5, 0.5, 0.55), enabled = true) =
    BoxOutlineOverlay(region, NTuple{3,Float64}(color), enabled, nothing, nothing)

enabled(o::BoxOutlineOverlay) = o.enabled

# Interleaved (x,y,z, r,g,b) vertices for the 12 edges of [lo,hi] (drawn as GL_LINES).
function _boxoutline_verts(lo, hi, color)
    r, g, b = Float32.(color)
    xs = (lo[1], hi[1]); ys = (lo[2], hi[2]); zs = (lo[3], hi[3])
    corner(ix, iy, iz) = (Float32(xs[ix]), Float32(ys[iy]), Float32(zs[iz]))
    edges = (
        ((1, 1, 1), (2, 1, 1)), ((1, 2, 1), (2, 2, 1)),     # 4 edges ∥ x
        ((1, 1, 2), (2, 1, 2)), ((1, 2, 2), (2, 2, 2)),
        ((1, 1, 1), (1, 2, 1)), ((2, 1, 1), (2, 2, 1)),     # 4 edges ∥ y
        ((1, 1, 2), (1, 2, 2)), ((2, 1, 2), (2, 2, 2)),
        ((1, 1, 1), (1, 1, 2)), ((2, 1, 1), (2, 1, 2)),     # 4 edges ∥ z
        ((1, 2, 1), (1, 2, 2)), ((2, 2, 1), (2, 2, 2)),
    )
    out = Float32[]
    for (ca, cb) in edges
        append!(out, (corner(ca...)..., r, g, b, corner(cb...)..., r, g, b))
    end
    out
end

function refresh!(o::BoxOutlineOverlay, ctx::GeomContext)
    o.gr === nothing && (o.gr = GeometryRenderer())
    reg = o.region === nothing ? ctx.region : o.region
    lo, hi = aabb(reg)
    key = (lo, hi, o.color)
    if key != o.last
        upload!(o.gr, _boxoutline_verts(lo, hi, o.color))
        o.last = key
    end
    o
end

draw!(o::BoxOutlineOverlay, ctx::GeomContext) = (o.gr === nothing || draw!(o.gr, ctx.viewproj); o)

overlay_fingerprint(o::BoxOutlineOverlay) = (:boxoutline, o.region, o.color, o.enabled)

# ── StreamlinesOverlay: evenly-spaced field lines of a VectorField, colored by |V| ──
mutable struct StreamlinesOverlay <: Overlay
    vf::VectorField
    colormap::Symbol
    min_density::Int
    max_density::Int
    mag_window::Union{Nothing,Tuple}
    enabled::Bool
    gr::Union{Nothing,GeometryRenderer}
    data::Any                        # US.StreamlineData, or nothing until integrated
    dirty::Bool
end
StreamlinesOverlay(vf::VectorField; colormap = :viridis, min_density = 1, max_density = 2,
                   mag_window = nothing, enabled = true) =
    StreamlinesOverlay(vf, colormap, Int(min_density), Int(max_density), mag_window, enabled,
                       nothing, nothing, true)

enabled(o::StreamlinesOverlay) = o.enabled

function refresh!(o::StreamlinesOverlay, ::GeomContext)
    o.gr === nothing && (o.gr = GeometryRenderer())
    if o.dirty
        o.data = compute_streamlines(o.vf; min_density = o.min_density, max_density = o.max_density)
        mw = o.mag_window === nothing ? magnitude_window(o.data) : o.mag_window
        upload!(o.gr, streamline_vertices(o.data, o.colormap, mw...))
        o.dirty = false
    end
    o
end

draw!(o::StreamlinesOverlay, ctx::GeomContext) = (o.gr === nothing || draw!(o.gr, ctx.viewproj); o)

overlay_fingerprint(o::StreamlinesOverlay) =
    (:streamlines, objectid(o.vf), o.colormap, o.min_density, o.max_density, o.mag_window, o.enabled)

# ── GlyphsOverlay: arrow glyphs subsampled from the streamlines of a VectorField ──
mutable struct GlyphsOverlay <: Overlay
    vf::VectorField
    colormap::Symbol
    min_density::Int
    max_density::Int
    every::Int
    mag_window::Union{Nothing,Tuple}
    enabled::Bool
    gr::Union{Nothing,GeometryRenderer}
    data::Any
    dirty::Bool
end
GlyphsOverlay(vf::VectorField; colormap = :viridis, min_density = 1, max_density = 2, every = 12,
              mag_window = nothing, enabled = true) =
    GlyphsOverlay(vf, colormap, Int(min_density), Int(max_density), Int(every), mag_window, enabled,
                  nothing, nothing, true)

enabled(o::GlyphsOverlay) = o.enabled

function refresh!(o::GlyphsOverlay, ::GeomContext)
    o.gr === nothing && (o.gr = GeometryRenderer())
    if o.dirty
        o.data = compute_streamlines(o.vf; min_density = o.min_density, max_density = o.max_density)
        mw = o.mag_window === nothing ? magnitude_window(o.data) : o.mag_window
        upload!(o.gr, glyph_vertices(o.data, o.colormap, mw...; every = o.every))
        o.dirty = false
    end
    o
end

draw!(o::GlyphsOverlay, ctx::GeomContext) = (o.gr === nothing || draw!(o.gr, ctx.viewproj); o)

overlay_fingerprint(o::GlyphsOverlay) =
    (:glyphs, objectid(o.vf), o.colormap, o.min_density, o.max_density, o.every, o.mag_window, o.enabled)

# Hosts add overlays explicitly; the default stack is empty.
default_overlays(field) = Overlay[]
