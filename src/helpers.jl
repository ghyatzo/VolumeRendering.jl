# a count → evenly spaced with the poles left out (an odd count includes the equator); explicit angles pass through
# (angles only reach sin/cos, so Unitful ones work as they are)
parallel_lats(n::Integer) = range(-π/2, π/2, n + 2)[2:end-1]
parallel_lats(lats) = lats
# a count → evenly spaced over the full turn; a meridian is a half great circle pole to pole
meridian_lons(n::Integer) = range(; start = 0.0, step = 2π / n, length = n)
meridian_lons(lons) = lons

unit_point(lat, lon) = SVector(cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat))

"""    graticule(sphere::Sphere; parallels = 3, meridians = 6, segments = 64)

Polylines of a latitude/longitude grid over `sphere`, for [`Lines`](@ref): latitude is measured from the xy plane,
longitude about z from +x. `parallels` and `meridians` are counts (evenly spaced; parallels leave out the poles) or
explicit angles in radians (Unitful angles work too). Full circles have `segments` segments. The curves are placed
by [`on_sphere`](@ref).
"""
graticule(sphere::Sphere; parallels = 3, meridians = 6, segments = 64) = on_sphere(vcat(   # parallels, then meridians (half circles)
    [[unit_point(lat, lon) for lon in range(0, 2π, segments + 1)] for lat in parallel_lats(parallels)],
    [[unit_point(lat, lon) for lat in range(-π/2, π/2, segments ÷ 2 + 1)] for lon in meridian_lons(meridians)]), sphere)

"""    on_sphere(curves, sphere::Sphere)

Polylines on `sphere` from `curves` of directions from its centre (vectors of any length): all scaled out by one
factor, just enough that no segment dips inside the sphere. To draw them over an opaque sphere, put them on a
slightly larger one, or list them after it: on an exact depth tie the later mark wins.
"""
function on_sphere(directions::AbstractVector{<:AbstractVector{<:AbstractVector}}, sphere::Sphere)
    dirs = map(c -> map(u -> normalize(SVector{3,Float64}(u)), c), directions)
    # a chord between directions σ apart dips to cos(σ/2) of its ends' radius: the widest step's chord touches
    σmax = maximum((atan(norm(cross(a, b)), dot(a, b)) for d in dirs for (a, b) in zip(d, @view d[2:end])); init = 0.0)
    r = sphere.radius / cos(σmax / 2)
    map(d -> map(u -> sphere.center + r * u, d), dirs)
end

"""    box_edges(box::Box)

The 12 edges of `box`, as two-point polylines for [`Lines`](@ref).
"""
function box_edges(box::Box)
    (; lo, hi) = aabb(box)
    corner(i) = SVector(ntuple(k -> (lo[k], hi[k])[i[k]], 3))
    # each edge: the corners differing in axis k only
    [[corner(setindex(i, 1, k)), corner(setindex(i, 2, k))] for k in 1:3 for i in Iterators.product(1:2, 1:2, 1:2) if i[k] == 1]
end

"""    axis_lines(center, radius)

Coordinate axes through `center`, `radius` long each way: `(; polylines, colors)`, x/y/z in red/green/blue, for
`Lines(ax.polylines; color = ax.colors)`.
"""
function axis_lines(center, radius)
    units = (SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 1.0))
    (; polylines = [[center - radius * e, center + radius * e] for e in units],
       colors = [fill(c, 2) for c in (RGB(1.0, 0.0, 0.0), RGB(0.0, 1.0, 0.0), RGB(0.0, 0.0, 1.0))])
end

"""    VectorField(f; region::Box)

A vector field over `region`, for [`streamlines`](@ref): `f(p::SVector{3,Float64})` returns the vector at `p`.
"""
struct VectorField{F}
    f::F
    region::Box
end
VectorField(f; region) = VectorField(f, region)

"""    Streamlines

The result of [`streamlines`](@ref): `polylines`, the field magnitude per vertex as `values` (nested like
`polylines`, for a [`Mapped`](@ref) colour), and the integration `data` that [`glyphs`](@ref) subsamples.
"""
struct Streamlines
    polylines::Vector{Vector{SVector{3,Float64}}}
    values::Vector{Vector{Float64}}
    data
end

"""    streamlines(vf::VectorField; min_density = 1, max_density = 2)::Streamlines

Evenly spaced streamlines of `vf` over its region, by UniformStreamlines.jl's `evenstream`: `min_density` sets how
densely seeds are tried, `max_density` how close streamlines may pass.
"""
function streamlines(vf::VectorField; min_density = 1, max_density = 2)
    (; lo, hi) = aabb(vf.region)
    # UniformStreamlines integrates the unit-normalized field, so the step sets the spatial resolution
    data = US.evenstream([lo.x, hi.x], [lo.y, hi.y], [lo.z, hi.z], vf.f; min_density, max_density, stepsize = maximum(hi - lo) / 100)
    polylines = split_at_nan(SVector{3,Float64}.(eachcol(data.paths)))
    Streamlines(polylines, map(l -> norm.(vf.f.(l)), polylines), data)
end

# `paths` holds the streamlines separated by NaN columns
function split_at_nan(pts)
    cuts = [0; findall(p -> any(isnan, p), pts); length(pts) + 1]
    filter(!isempty, [pts[a+1:b-1] for (a, b) in zip(cuts, @view cuts[2:end])])
end

"""    glyphs(s::Streamlines; every = 12, headfrac = 0.35)

Arrows along the streamlines `s`, one per `every` integration steps, pointing along the field; `headfrac` is the head
size relative to the arrow length. Returns `(; polylines, values)`, the field magnitude per vertex, for
[`Lines`](@ref). Reuses the integration in `s`: a new `every` needs no new [`streamlines`](@ref).
"""
function glyphs(s::Streamlines; every = 12, headfrac = 0.35)
    arr = US.streamarrows(s.data; every)
    arrows = map(axes(arr.points, 2)) do k
        base, v = SVector{3,Float64}(view(arr.points, :, k)), SVector{3,Float64}(view(arr.vectors, :, k))
        tip, L = base + v, norm(v)
        dir = v / L
        c = cross(dir, SVector(0.0, 0.0, 1.0))
        side = norm(c) > 1e-6 ? normalize(c) : SVector(1.0, 0.0, 0.0)   # dir ∥ z: any perpendicular
        head(sgn) = tip - headfrac * L * (dir + sgn * side)
        # shaft + one head stroke, then the other head stroke
        (; polylines = [[base, tip, head(1)], [tip, head(-1)]], values = [fill(arr.speeds[k], 3), fill(arr.speeds[k], 2)])
    end
    (; polylines = Vector{SVector{3,Float64}}[l for a in arrows for l in a.polylines],
       values = Vector{Float64}[v for a in arrows for v in a.values])
end
