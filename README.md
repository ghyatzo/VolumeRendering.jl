# VolumeRendering.jl

**An embeddable [Dear ImGui](https://github.com/JuliaImGui/CImGui.jl) widget for real-time GPU volume rendering of 3D fields, together with surfaces and lines.**

VolumeRendering turns 3D fields into an interactive, ray-marched image on the GPU, and hands it to you as a single ImGui widget you can drop into your own application — the way you'd drop a plot into a window. A scene is a list of marks: volumes, surfaces (spheres, boxes, slice planes, your own shapes) and lines (streamlines, glyphs, outlines, any polylines), all depth-composited together.

The recommended way to use `VolumeRendering.jl` is as a Dear ImGui widget, but it can be integrated into other environments that have OpenGL (4.1 core), can render a texture, and pass mouse events.

Simple example, see [demo.jl](examples/demo.jl):

https://github.com/user-attachments/assets/0ff46ebc-9a73-4ef7-b758-9f330e348ea5

In [PlasmaScope.jl](https://github.com/aplavin/PlasmaScope.jl) that uses `VolumeRendering.jl` to display GRHMD snapshots:

https://github.com/user-attachments/assets/fb981f3e-8402-4861-85fb-c061f06532b5

# Quick start

```julia
using VolumeRendering, AxisKeys, ColorSchemes

# your data: a 3D `KeyedArray` whose axis keys give the world coordinates
A = KeyedArray(...; x = xs, y = ys, z = zs)

grid = GridField(A)
vol  = Volume(Mapped(grid, Colormap(ColorSchemes.viridis, value_range(grid))); region = Box(grid))
view = View([vol]; camera = Camera(; bounds(vol.region)...))

# ...then, inside your Dear ImGui frame (with CImGui loaded):
VolumeRendering.ShowView(view)
```

The host's GL context must be OpenGL 4.1 core: pass `opengl_version = v"4.1"` to `CImGui.render`, see [demo.jl](examples/demo.jl).

`ShowView` fills the current content region with the rendered view and turns mouse/keyboard into camera motion: drag to orbit, right/middle drag to pan, wheel to zoom; with `view.flymode = true`, drag looks around and WASD/Space/C fly. There is no window or render-loop boilerplate of our own — you drive the ImGui frame like you already do for any other widget.

Everything is explicit: a colour is a `Colormap` (a `ColorScheme` object and a value window) applied to a field through `Mapped`, a volume always has a `region`, a view always has a `camera`. Marks, fields and views are plain data; nothing touches the GPU until the first render. With a current OpenGL context, `render_image(view, w, h)` renders a view into an image matrix without any UI. Other hosts call `render!(view, w, h)`, which returns the GL texture to display, and drive the camera with `orbit!`, `pan!`, `zoom!(cam, factor, view.initial_camera)`, `fly!`; `reset!(view)` restores the initial camera, `free!(view)` releases GL resources.

# Scalar fields

## Rectilinear grid

`GridField` wraps a `KeyedArray`; its dim names choose the coordinates — `(x, y, z)` for a Cartesian grid, `(lon, lat)` (radians, or Unitful angles) for a field on directions from the origin, e.g. to colour a sphere:

```julia
xs = ys = zs = -1:0.02:1
grid = GridField(KeyedArray([exp(-(x^2 + y^2 + z^2) / 0.1) for x in xs, y in ys, z in zs]; x = xs, y = ys, z = zs);
                 interpolation = :linear)   # or :nearest
Box(grid)          # the box of the keys, a natural Volume region
value_range(grid)  # data range, to build a Colormap window from
```

An RGBA-valued grid is a colour field (rgb = emission, alpha = extinction) and is used directly as a volume's colour: `Volume(grid; region = Box(grid))`, see [milkyway.jl](examples/milkyway.jl).

For a non-uniform axis, define one method — `VolumeRendering.axis_index_glsl(k::MyAxis, coord)` returning the GLSL expression that maps a world coordinate to a fractional cell index — and your axis type works inside a `KeyedArray` like any range.

## Analytic

`GLSLField` is a field defined directly in GLSL, optionally with textures and uniforms:

```julia
f = GLSLField("""
        float value(vec3 p) { return exp(-dot(p, p) * k); }
        float stepSize(vec3 p) { return 0.05; }
        """;
    uniforms = (; k = 0.7))
vol = Volume(Mapped(f, Colormap(ColorSchemes.magma, value_range(f))); region = Sphere(radius = 3.0))   # value_range defaults to 0..1
```

`GLSLField(...; kind = :color)` with `vec4 value(vec3 p)` is an analytic RGBA field.

## Custom field

Any type is a field when it implements the field interface — extend the functions qualified, `VolumeRendering.field_glsl(f::MyField) = ...` etc. (likewise for shapes):

```julia
field_glsl(f)    # GLSL defining `float value(vec3 p)` (or `vec4` for a colour field) and `float stepSize(vec3 p)`
value_kind(f)    # :scalar | :color
textures(f)      # optional, NamedTuple of name => Texture(array; wrap, filter); declared as samplers for you
uniforms(f)      # optional, NamedTuple of name => value (numbers, SVectors, SMatrices); declared for you
value_range(f)   # optional, for building a colormap window
VolumeRendering.controls!(f)   # optional, ImGui widgets for the field's own settings
```

Store the arrays inside `f` and return the same objects from `textures(f)` each time: they are uploaded once and cached by identity.

# Custom shapes

Volume regions and surfaces are shapes: `Sphere`, `Box`, `Plane` (the parallelogram `Plane(origin, u, v)`; `Plane(:z, 0.0, box)` is the slice of `box` at `z = 0`), or any type implementing

```julia
shape_glsl(s)   # GLSL defining `vec2 vr_hit(vec3 ro, vec3 rd)`: entry and exit ray parameters (tfar < tnear on a miss)
uniforms(s)     # optional, as for fields
bounds(s)       # (; center, radius), an enclosing sphere: `Camera(; bounds(s)...)` frames it
aabb(s)         # (; lo, hi), an axis-aligned bounding box
```

Shapes must be convex.

# Marks

A view draws a list of marks:

```julia
using StaticArrays, ColorTypes, IntervalSets

box = Box(grid)
colored = Mapped(grid, Colormap(ColorSchemes.afmhot, 1e-3..1.0; log = true))
view = View([
    Volume(colored; region = box, mode = :dvr),                 # :dvr | :mip | :average
    Surface(Plane(:z, 0.0, box); color = colored),              # a slice coloured by the same field and colormap
    Surface(Sphere(radius = 0.2); color = RGB(0.3, 0.3, 0.35)), # opaque; `opacity < 1` for translucent
    Lines(box_edges(box)),
    Lines([[SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 0.5)]]; width_px = 2.0, color = RGB(1.0, 0.5, 0.0)),
]; camera = Camera(; bounds(box)...))
```

A mark's `color` is a constant colour, per-vertex colours (`Lines`), a colour field, or a `Mapped` — a scalar field or per-vertex values through a `Colormap`. Helpers return polylines for `Lines`: `box_edges(box)`, `graticule(sphere; parallels, meridians)` (each a count or explicit angles, radians or Unitful), and `on_sphere(curves, sphere)` for curves given as directions from a sphere's centre (both scaled out just enough that no segment dips inside the sphere; draw them on a slightly larger sphere than an opaque one to avoid depth ties). Opaque marks are depth-tested; translucent ones (volumes, surfaces with `opacity < 1`) are composited in list order. Coloured x/y/z axes: `ax = axis_lines(center, radius); Lines(ax.polylines; color = ax.colors)`.

# Vector fields

A `VectorField` wraps a Julia function `p -> v` over a box. `streamlines` integrates evenly spaced field lines, `glyphs` subsamples them into arrows; both are drawn as `Lines`, coloured by magnitude:

```julia
vf = VectorField(p -> SVector(-p[2], p[1], 0.2); region = Box(-1..1, -1..1, -1..1))
s = streamlines(vf)
g = glyphs(s; every = 8)
speed = Colormap(ColorSchemes.viridis, value_range(s.values))
marks = [Lines(s.polylines; color = Mapped(s.values, speed)),
         Lines(g.polylines; color = Mapped(g.values, speed))]
```

# Updating

Marks, fields, colormaps, `Mapped`, the camera and the view are mutable and read every frame: setting a property is the whole update protocol (`cmap.window = 0.1..2.0`, `vol.mode = :mip`, `grid.data = A2`, `mapped.source = other_field`), and the view re-renders only when something changed. Bulk arrays (grid data, textures, polylines, per-vertex colours/values) are tracked by identity — assign new arrays, never mutate them in place. Shared objects are shared by reference — mutate them (`mapped.source = f`) to update every mark using them; to show or hide a mark, add or remove it from `view.marks`.

# UI

Right-click the view to open its controls: projection, camera reset, fly mode, render quality, and per-mark, per-field and per-colormap settings (volume mode, density, extinction curve editor, opacity, line width, interpolation, colour scheme, log scale and window). `VolumeRendering.ShowControls(view)` emits the same panel into any ImGui window.
