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

# Concepts

A `View` draws a list of marks through a camera:

- `Volume(color; region)` ray-marches a field through a shape;
- `Surface(shape; color, opacity)` draws the surface of a shape;
- `Lines(polylines; color, width_px)` draws polylines.

Fields are functions of 3D position evaluated on the GPU. `GridField` samples a `KeyedArray` whose dim names choose the coordinates: `(x, y, z)`, or `(lon, lat)` for a field on directions from the origin, e.g. to colour a sphere. `GLSLField` is written directly in GLSL. A grid of colours is a colour field (rgb = emission, alpha = extinction). Shapes are `Sphere`, `Box` and `Plane`.

A mark's `color` is a constant colour, per-vertex colours (`Lines`), a colour field, or a `Mapped`: a scalar field or per-vertex values through a `Colormap` (a `ColorScheme` and a value window).

Helpers return polylines for `Lines`: `box_edges`, `graticule`, `on_sphere`, `axis_lines`, and `streamlines` and `glyphs` of a `VectorField`. Each of these has a docstring with the details.

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

Opaque marks are depth-tested; translucent ones (volumes, surfaces with `opacity < 1`) are composited in list order.

# Updating

Marks, fields, colormaps, `Mapped`, the camera and the view are plain mutable data, read every frame; nothing touches the GPU until the first render. Setting a property is the whole update protocol (`cmap.window = 0.1..2.0`, `vol.mode = :mip`, `grid.data = A2`, `mapped.source = other_field`), and the view re-renders only when something changed. Bulk arrays (grid data, textures, polylines, per-vertex colours/values) are tracked by identity — assign new arrays, never mutate them in place. Shared objects are shared by reference — mutate them (`mapped.source = f`) to update every mark using them. To show or hide a mark, add or remove it from `view.marks`.

# Extending

A field is any type that supplies its GLSL: implement `field_glsl` — defining `float value(vec3 p)` (or `vec4` for a colour field) and `float stepSize(vec3 p)` — and `value_kind`; optionally `textures`, `uniforms`, `value_range`, `controls!`. A shape is any convex type implementing `shape_glsl` (its ray hit), `bounds` and `aabb`. See their docstrings for the details. For a one-off field, `GLSLField` needs no new type:

```julia
f = GLSLField("""
        float value(vec3 p) { return exp(-dot(p, p) * k); }
        float stepSize(vec3 p) { return 0.05; }
        """; uniforms = (; k = 0.7))
```

# Examples

- [demo.jl](examples/demo.jl): a volume in a complete Dear ImGui program.
- [milkyway.jl](examples/milkyway.jl): an RGBA colour volume reshaped live by sliders; it needs the OpenSpace Milky Way volume, see the file's header for the download.
