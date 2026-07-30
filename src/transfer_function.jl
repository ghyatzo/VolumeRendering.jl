# Transfer-function model: a colormap (color vs normalized value) plus an editable
# piecewise-linear opacity curve. Baked to a 256×1 RGBA texture the shader samples.
# The interactive editor widget is not part of this package (this file stays UI-free).

using ColorSchemes, StaticArrays

# A small default list a host UI can offer. NOT a restriction: `colormap` accepts any
# ColorSchemes symbol (bake! looks it up in ColorSchemes.colorschemes directly).
const TF_COLORMAPS = (:viridis, :inferno, :turbo, :afmhot)

mutable struct TransferFunction
    colormap::Symbol
    opacity_pts::Vector{SVector{2,Float64}}   # (value∈[0,1], opacity∈[0,1]), kept sorted by value
    logscale::Bool
    lo::Float64; hi::Float64                   # value-normalization window (log10 units if logscale)
    tex::UInt32
    dirty::Bool
end

function TransferFunction(; colormap = :viridis, logscale = false, lo = 0.0, hi = 1.0)
    TransferFunction(colormap, [SVector(0.0, 0.0), SVector(1.0, 1.0)], logscale, lo, hi, tex_tf(), true)
end

# Piecewise-linear opacity at normalized value t∈[0,1] (clamped to the end points).
function opacity_at(tf::TransferFunction, t)
    pts = tf.opacity_pts
    t <= pts[1][1]   && return pts[1][2]
    t >= pts[end][1] && return pts[end][2]
    for k in 2:length(pts)
        if t <= pts[k][1]
            x0, y0 = pts[k-1]; x1, y1 = pts[k]
            return y0 + (t - x0) / (x1 - x0 + eps()) * (y1 - y0)
        end
    end
    pts[end][2]
end

# Keep control points sorted and clamped after an edit.
function normalize_points!(tf::TransferFunction)
    sort!(tf.opacity_pts; by = p -> p[1])
    tf.opacity_pts .= (p -> SVector(clamp(p[1], 0.0, 1.0), clamp(p[2], 0.0, 1.0))).(tf.opacity_pts)
    tf.dirty = true
    tf
end

# Rebuild the RGBA texture: color from the colormap, alpha from the opacity curve.
function bake!(tf::TransferFunction)
    scheme = ColorSchemes.colorschemes[tf.colormap]
    data = Matrix{Float32}(undef, 4, 256)
    for i in 1:256
        t = (i - 1) / 255
        c = get(scheme, t)
        data[:, i] .= (Float32(c.r), Float32(c.g), Float32(c.b), Float32(opacity_at(tf, t)))
    end
    update_tf!(tf.tex, data)
    tf.dirty = false
    tf
end

ensure_baked!(tf::TransferFunction) = tf.dirty ? bake!(tf) : tf

# Reset the value-normalization window to span the field's value range `(vmin, vmax)`.
# Default (linear): `[vmin, vmax]`, robust to negative fields. Log is opt-in: if the caller
# sets `tf.logscale = true`, use 3 decades below a positive max: `[log10(vmax)-3, log10(vmax)]`.
# `lo`/`hi` are live shader uniforms (not baked).
function default_window!(tf::TransferFunction, (vmin, vmax))
    if tf.logscale
        tf.hi = log10(max(vmax, floatmin(Float32))); tf.lo = tf.hi - 3
    else
        tf.lo = vmin; tf.hi = vmax
    end
    tf
end
