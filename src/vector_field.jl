# A Cartesian vector field over a Region, plus evenly-spaced 3D streamlines / arrow glyphs of it via
# UniformStreamlines. Pure CPU (no GL) — the GL upload lives in geometry_renderer.jl; here we produce
# flat interleaved (x,y,z, r,g,b) vertex arrays ready to draw as GL_LINES.

using StaticArrays, LinearAlgebra, Statistics
using ColorSchemes
import UniformStreamlines as US

# A Cartesian vector field: `f(p::SVector{3,Float64})::SVector{3,Float64}` over `region`.
struct VectorField{F}
    f::F
    region::Region
end
VectorField(f; region) = VectorField(f, region)

# Evenly-spaced streamlines of the field over the region's AABB (unit-normalized integration, so
# `stepsize` sets spatial resolution and any near-singularity |V|→∞ never blows up the step). The
# box may be non-cubic, so the integration extents are per-axis.
function compute_streamlines(vf::VectorField; min_density=1, max_density=2)
    lo, hi = aabb(vf.region)
    US.evenstream([lo[1], hi[1]], [lo[2], hi[2]], [lo[3], hi[3]], vf.f;
                  min_density, max_density, stepsize = maximum(hi .- lo) / 100)
end

# Log10 color window spanning the streamlines' magnitude range (top ~2% down 3 decades).
function magnitude_window(data)
    mags = Float64[norm(data.field(c)) for c in eachcol(data.paths) if !isnan(c[1])]
    isempty(mags) && return (-3.0, 0.0)
    hi = log10(max(quantile!(mags, 0.98), 1e-30)); (hi - 3, hi)
end

_color(scheme, v, lo, hi) = let t = clamp((log10(max(v, 1e-30)) - lo) / (hi - lo), 0.0, 1.0)
    c = ColorSchemes.get(scheme, t); (Float32(c.r), Float32(c.g), Float32(c.b))
end

# Interleaved (x,y,z, r,g,b) GL_LINES vertices for the streamlines: each polyline (NaN-separated in
# `paths`) becomes consecutive point-pairs, colored per vertex by |V| through `colormap`/[lo,hi].
function streamline_vertices(data, colormap::Symbol, lo, hi)
    scheme = ColorSchemes.colorschemes[colormap]
    out = Float32[]
    P = data.paths
    push_vert!(j) = (p = @view P[:, j]; v = norm(data.field(p));
        append!(out, (Float32(p[1]), Float32(p[2]), Float32(p[3]), _color(scheme, v, lo, hi)...)))
    for j in 1:size(P, 2)-1
        (isnan(P[1, j]) || isnan(P[1, j+1])) && continue
        push_vert!(j); push_vert!(j+1)
    end
    out
end

# Interleaved GL_LINES vertices for arrow glyphs (shaft + a small 2-segment head), subsampled from
# the streamlines and colored by local speed.
function glyph_vertices(data, colormap::Symbol, lo, hi; every=12, headfrac=0.35)
    scheme = ColorSchemes.colorschemes[colormap]
    arr = US.streamarrows(data; every)
    out = Float32[]
    seg!(a, b, col) = append!(out, (Float32(a[1]), Float32(a[2]), Float32(a[3]), col...,
                                    Float32(b[1]), Float32(b[2]), Float32(b[3]), col...))
    up = SVector(0.0, 0.0, 1.0)
    for k in axes(arr.points, 2)
        base = SVector{3}(@view arr.points[:, k]); vec = SVector{3}(@view arr.vectors[:, k])
        tip = base + vec; L = norm(vec); L < 1e-9 && continue
        col = _color(scheme, arr.speeds[k], lo, hi)
        dir = vec / L
        side = cross(dir, up); ns = norm(side); side = ns > 1e-6 ? side/ns : SVector(1.0,0.0,0.0)
        seg!(base, tip, col)                                            # shaft
        seg!(tip, tip - headfrac*L*(dir + side), col)                  # head ┐
        seg!(tip, tip - headfrac*L*(dir - side), col)                  # head ┘
    end
    out
end
