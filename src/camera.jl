# Unified eye/lookat/up camera (Makie Camera3D model). Both orbit and fly reduce to one
# primitive — rotate a point about a pivot — so there is no azimuth/elevation bookkeeping and
# no conversion when switching modes. Pure math (no CImGui); input wiring lives in ui.jl.

using StaticArrays, LinearAlgebra
import Rotations

mutable struct Camera
    eye::SVector{3,Float64}
    lookat::SVector{3,Float64}
    up::SVector{3,Float64}       # world up (z by default)
    projection::Symbol           # :perspective | :ortho
    fov::Float64                 # field of view [deg] (perspective)
    ortho_half::Float64          # half-width (orthographic)
end

# Default framing for a scene bounding sphere `(center, radius)`: look at the center from an
# oblique angle.
function Camera(; center = SVector(0.0, 0.0, 0.0), radius::Real)
    dir = normalize(SVector(1.0, 0.4, 0.7))
    Camera(center + dir * 3radius, center, SVector(0.0, 0.0, 1.0),
           :perspective, 45.0, 1.1radius)
end

distance(cam::Camera) = norm(cam.eye - cam.lookat)

# Orthonormal camera basis (right, up, forward), forward = eye → lookat.
function view_basis(cam::Camera)
    fwd = normalize(cam.lookat - cam.eye)
    r = cross(fwd, cam.up); nr = norm(r)
    right = nr > 1e-9 ? r / nr : SVector(1.0, 0.0, 0.0)
    (right, cross(right, fwd), fwd)
end

# perspective: tan(fov/2); orthographic: half-width. Feeds the shader's `fovscale` uniform.
fovscale(cam::Camera) = cam.projection === :ortho ? cam.ortho_half : tand(cam.fov / 2)

# Rotate point P about pivot Q by yaw (about world up) then pitch (about camera-right),
# clamping the polar angle from `up` so the view never flips over the pole.
function _rotate_about(P, Q, up, dyaw, dpitch)
    up = normalize(up)
    v = P - Q
    v = Rotations.AngleAxis(dyaw, up[1], up[2], up[3]) * v
    ax = cross(v, up); na = norm(ax)
    if na > 1e-9
        right = ax / na
        ang = acos(clamp(dot(normalize(v), up), -1.0, 1.0))
        desired = clamp(ang - dpitch, 0.02, π - 0.02)
        v = Rotations.AngleAxis(ang - desired, right[1], right[2], right[3]) * v
    end
    Q + v
end

orbit!(cam::Camera, dyaw, dpitch)   = (cam.eye    = _rotate_about(cam.eye, cam.lookat, cam.up, dyaw, dpitch); cam)
flylook!(cam::Camera, dyaw, dpitch) = (cam.lookat = _rotate_about(cam.lookat, cam.eye, cam.up, dyaw, dpitch); cam)

function zoom!(cam::Camera, factor)
    if cam.projection === :ortho
        cam.ortho_half = clamp(cam.ortho_half * factor, 0.5, 1e5)
    else
        v = cam.eye - cam.lookat
        cam.eye = cam.lookat + normalize(v) * clamp(norm(v) * factor, 1.0, 1e6)
    end
    cam
end

# Pan target+eye together in the view plane; screen-fraction dx,dy scaled by the view size.
function pan!(cam::Camera, dx, dy)
    right, up, _ = view_basis(cam)
    scale = cam.projection === :ortho ? cam.ortho_half : distance(cam) * tand(cam.fov / 2)
    shift = (-dx * right + dy * up) * scale
    cam.eye += shift; cam.lookat += shift
    cam
end

# Fly: translate eye+lookat together along the camera frame (WASD/QE).
function fly!(cam::Camera, dforward, dright, dup, speed)
    right, up, fwd = view_basis(cam)
    shift = (dforward * fwd + dright * right + dup * up) * speed
    cam.eye += shift; cam.lookat += shift
    cam
end

reset!(cam::Camera, default::Camera) = (cam.eye = default.eye; cam.lookat = default.lookat;
    cam.up = default.up; cam.fov = default.fov; cam.ortho_half = default.ortho_half; cam)

# View·projection matrix, constructed to match the fragment shader's ray generation exactly (so a
# world point's projected depth is consistent between the ray-march and the line/point geometry
# pass). near/far bracket the content's bounding sphere `(center, radius)` along the view axis —
# anchored to the CONTENT, not `cam.lookat`, so geometry is never clipped away when the eye flies in
# close while `lookat` aims far past it. Eye space: +X right, +Y up, +Z = −forward (GL convention).
function view_proj(cam::Camera, center, radius, aspect)
    right, up, fwd = view_basis(cam); e = cam.eye; fs = fovscale(cam)
    depth = dot(fwd, center - e)                         # content-center depth along the view axis
    near = max(0.01 * radius, depth - radius)
    far  = max(depth + radius, near + 0.01 * radius)     # keep a valid frustum if content is behind the eye
    V = @SMatrix [ right[1] right[2] right[3] -dot(right, e);
                   up[1]    up[2]    up[3]    -dot(up, e);
                  -fwd[1]  -fwd[2]  -fwd[3]    dot(fwd, e);
                   0.0      0.0      0.0       1.0 ]
    P = if cam.projection === :perspective
        @SMatrix [ 1/(fs*aspect) 0.0  0.0                    0.0;
                   0.0           1/fs 0.0                    0.0;
                   0.0           0.0 -(far+near)/(far-near) -2far*near/(far-near);
                   0.0           0.0 -1.0                    0.0 ]
    else
        @SMatrix [ 1/(fs*aspect) 0.0  0.0            0.0;
                   0.0           1/fs 0.0            0.0;
                   0.0           0.0 -2/(far-near) -(far+near)/(far-near);
                   0.0           0.0  0.0            1.0 ]
    end
    P * V
end
