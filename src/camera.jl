"""    Camera(eye, lookat, up, projection, fov, ortho_half)
    Camera(; center = SVector(0.0, 0.0, 0.0), radius)

A camera at `eye` looking at `lookat`, with `up` the world up direction. `projection` is `:perspective` (vertical
field of view `fov`, in degrees) or `:ortho` (view half-height `ortho_half`, in world units).

The keyword form frames the sphere `(center, radius)` from an oblique default pose; `Camera(; bounds(shape)...)` frames
a shape. Mutable: assign fields or use [`orbit!`](@ref), [`flylook!`](@ref), [`pan!`](@ref), [`zoom!`](@ref),
[`fly!`](@ref).
"""
mutable struct Camera
    eye::SVector{3,Float64}
    lookat::SVector{3,Float64}
    up::SVector{3,Float64}
    projection::Symbol
    fov::Float64
    ortho_half::Float64
end

Camera(; center = SVector(0.0, 0.0, 0.0), radius) =
    Camera(center + normalize(SVector(1.0, 0.4, 0.7)) * 3radius, center, SVector(0.0, 0.0, 1.0), :perspective, 45.0, 1.1radius)

distance(cam::Camera) = norm(cam.eye - cam.lookat)

# orthonormal (right, up, forward), forward = eye → lookat
function view_basis(cam::Camera)
    fwd = normalize(cam.lookat - cam.eye)
    r = cross(fwd, cam.up); nr = norm(r)
    right = nr > 1e-9 ? r / nr : SVector(1.0, 0.0, 0.0)
    (right, cross(right, fwd), fwd)
end

# perspective: tan(fov/2); ortho: half-height
fovscale(cam::Camera) = cam.projection === :ortho ? cam.ortho_half : tand(cam.fov / 2)

# Rotate P about pivot Q by yaw (about world up) then pitch (about camera-right); the polar angle from `up` is clamped
# so the view never flips over the pole. Orbit and fly-look are both this, with eye and lookat swapped.
function rotate_about(P, Q, up, dyaw, dpitch)
    up = normalize(up)
    v = Rotations.AngleAxis(dyaw, up...) * (P - Q)
    ax = cross(v, up); na = norm(ax)
    if na > 1e-9
        ang = acos(clamp(dot(normalize(v), up), -1.0, 1.0))
        v = Rotations.AngleAxis(ang - clamp(ang - dpitch, 0.02, π - 0.02), (ax / na)...) * v
    end
    Q + v
end

"""    orbit!(cam, dyaw, dpitch)

Move the eye around `lookat`: `dyaw` radians about the world up, then `dpitch` radians towards the world up, stopping
short of the poles. Returns `cam`.
"""
orbit!(cam::Camera, dyaw, dpitch) = (cam.eye = rotate_about(cam.eye, cam.lookat, cam.up, dyaw, dpitch); cam)

"""    flylook!(cam, dyaw, dpitch)

Turn the view in place: like [`orbit!`](@ref), but `lookat` moves around the eye. Returns `cam`.
"""
flylook!(cam::Camera, dyaw, dpitch) = (cam.lookat = rotate_about(cam.lookat, cam.eye, cam.up, dyaw, dpitch); cam)

"""    zoom!(cam, factor, reference::Camera)

Scale the eye–lookat distance (perspective) or `ortho_half` (ortho) by `factor`, clamped to `1e-4 … 1e4` times that of
`reference` — so the limits follow the scene's scale. Returns `cam`.
"""
function zoom!(cam::Camera, factor, reference::Camera)
    if cam.projection === :ortho
        cam.ortho_half = clamp(cam.ortho_half * factor, 1e-4 * reference.ortho_half, 1e4 * reference.ortho_half)
    else
        v = cam.eye - cam.lookat
        cam.eye = cam.lookat + normalize(v) * clamp(norm(v) * factor, 1e-4 * distance(reference), 1e4 * distance(reference))
    end
    cam
end

"""    pan!(cam, dx, dy)

Move eye and lookat together in the view plane by `dx`, `dy` view half-heights (at `lookat`), in screen directions:
the scene follows `dx > 0` to the right and `dy > 0` downwards. Returns `cam`.
"""
function pan!(cam::Camera, dx, dy)
    right, up, _ = view_basis(cam)
    scale = cam.projection === :ortho ? cam.ortho_half : distance(cam) * tand(cam.fov / 2)
    shift = (-dx * right + dy * up) * scale
    cam.eye += shift; cam.lookat += shift
    cam
end

"""    fly!(cam, dforward, dright, dup, speed)

Move eye and lookat together by `speed` world units along the view's forward, right and up directions, weighted by
`dforward`, `dright`, `dup`. Returns `cam`.
"""
function fly!(cam::Camera, dforward, dright, dup, speed)
    right, up, fwd = view_basis(cam)
    shift = (dforward * fwd + dright * right + dup * up) * speed
    cam.eye += shift; cam.lookat += shift
    cam
end

"""    reset!(view::View)
    reset!(cam::Camera, default::Camera)

Restore `view.camera` to the camera the view was constructed with, or copy every field of `default` into `cam`. In
place: hosts may hold a reference to the camera.
"""
reset!(cam::Camera, default::Camera) =
    (foreach(n -> setproperty!(cam, n, getproperty(default, n)), fieldnames(Camera)); cam)

"""    view_proj(cam, aspect)

The 4×4 world → clip-space matrix for a view of width/height ratio `aspect`. It has no near/far planes: the z row is
zero, and `w` is the depth along the view direction (perspective) or 1 (ortho).
"""
function view_proj(cam::Camera, aspect)
    # eye space: +x right, +y up, +z = −forward
    right, up, fwd = view_basis(cam); e = cam.eye; fs = fovscale(cam)
    V = @SMatrix [ right[1] right[2] right[3] -dot(right, e);
                   up[1]    up[2]    up[3]    -dot(up, e);
                  -fwd[1]  -fwd[2]  -fwd[3]    dot(fwd, e);
                   0.0      0.0      0.0       1.0 ]
    P = cam.projection === :ortho ?
        @SMatrix([1/(fs*aspect) 0.0 0.0 0.0; 0.0 1/fs 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 1.0]) :
        @SMatrix([1/(fs*aspect) 0.0 0.0 0.0; 0.0 1/fs 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 -1.0 0.0])
    P * V
end
