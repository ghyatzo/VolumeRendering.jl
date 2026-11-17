# Pure helpers returning data for `Lines` (§10). Angles are radians; Unitful angles work too, since they only reach sin/cos.

# a count → evenly spaced with the poles left out (an odd count includes the equator); explicit angles pass through
parallel_lats(n::Integer) = range(-π/2, π/2, n + 2)[2:end-1]
parallel_lats(lats) = lats
# a count → evenly spaced over the full turn; a meridian is a half great circle pole to pole
meridian_lons(n::Integer) = range(; start = 0.0, step = 2π / n, length = n)
meridian_lons(lons) = lons

unit_point(lat, lon) = SVector(cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat))

# rings at fixed latitude, then meridians (half the segments: half a circle)
graticule(sphere::Sphere; parallels = 3, meridians = 6, segments = 64) = on_sphere(vcat(
    [[unit_point(lat, lon) for lon in range(0, 2π, segments + 1)] for lat in parallel_lats(parallels)],
    [[unit_point(lat, lon) for lat in range(-π/2, π/2, segments ÷ 2 + 1)] for lon in meridian_lons(meridians)]), sphere)

# Curves of directions from the centre, scaled out until the chord of the widest step touches the sphere at its midpoint
# (a chord between directions σ apart dips to cos(σ/2) of its ends' radius): no segment dips inside the sphere.
function on_sphere(directions::AbstractVector{<:AbstractVector{<:AbstractVector}}, sphere::Sphere)
    dirs = map(c -> map(u -> normalize(SVector{3,Float64}(u)), c), directions)
    σmax = maximum((atan(norm(cross(a, b)), dot(a, b)) for d in dirs for (a, b) in zip(d, @view d[2:end])); init = 0.0)
    r = sphere.radius / cos(σmax / 2)
    map(d -> map(u -> sphere.center + r * u, d), dirs)
end

function box_edges(box::Box)
    (; lo, hi) = aabb(box)
    corner(i) = SVector(ntuple(k -> (lo[k], hi[k])[i[k]], 3))
    # each edge: the corners differing in axis k only
    [[corner(setindex(i, 1, k)), corner(setindex(i, 2, k))] for k in 1:3 for i in Iterators.product(1:2, 1:2, 1:2) if i[k] == 1]
end

function axis_lines(center, radius)
    units = (SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 1.0))
    (; polylines = [[center - radius * e, center + radius * e] for e in units],
       colors = [fill(c, 2) for c in (RGB(1.0, 0.0, 0.0), RGB(0.0, 1.0, 0.0), RGB(0.0, 0.0, 1.0))])
end

# ── vector fields (UniformStreamlines) ──

struct VectorField{F}
    f::F          # f(p::SVector{3,Float64})::SVector{3,Float64}
    region::Box
end
VectorField(f; region) = VectorField(f, region)

struct Streamlines
    polylines::Vector{Vector{SVector{3,Float64}}}
    values::Vector{Vector{Float64}}   # |V| per vertex
    data                              # the UniformStreamlines result, consumed by `glyphs`
end

# evenly spaced streamlines over the region; UniformStreamlines integrates the unit-normalized field, so the step
# sets the spatial resolution
function streamlines(vf::VectorField; min_density = 1, max_density = 2)
    (; lo, hi) = aabb(vf.region)
    data = US.evenstream([lo.x, hi.x], [lo.y, hi.y], [lo.z, hi.z], vf.f; min_density, max_density, stepsize = maximum(hi - lo) / 100)
    polylines = split_at_nan(SVector{3,Float64}.(eachcol(data.paths)))
    Streamlines(polylines, map(l -> norm.(vf.f.(l)), polylines), data)
end

# `paths` holds the streamlines separated by NaN columns
function split_at_nan(pts)
    cuts = [0; findall(p -> any(isnan, p), pts); length(pts) + 1]
    filter(!isempty, [pts[a+1:b-1] for (a, b) in zip(cuts, @view cuts[2:end])])
end

# arrows subsampled from the integrated streamlines: shaft + one head stroke, then the other head stroke
function glyphs(s::Streamlines; every = 12, headfrac = 0.35)
    arr = US.streamarrows(s.data; every)
    arrows = map(axes(arr.points, 2)) do k
        base, v = SVector{3,Float64}(view(arr.points, :, k)), SVector{3,Float64}(view(arr.vectors, :, k))
        tip, L = base + v, norm(v)
        dir = v / L
        c = cross(dir, SVector(0.0, 0.0, 1.0))
        side = norm(c) > 1e-6 ? normalize(c) : SVector(1.0, 0.0, 0.0)   # dir ∥ z: any perpendicular
        head(sgn) = tip - headfrac * L * (dir + sgn * side)
        (; polylines = [[base, tip, head(1)], [tip, head(-1)]], values = [fill(arr.speeds[k], 3), fill(arr.speeds[k], 2)])
    end
    (; polylines = Vector{SVector{3,Float64}}[l for a in arrows for l in a.polylines],
       values = Vector{Float64}[v for a in arrows for v in a.values])
end
