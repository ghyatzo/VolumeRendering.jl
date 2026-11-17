# Eye/lookat/up camera. Orbit and fly both reduce to one primitive — rotate a point about a pivot — so there is no
# azimuth/elevation bookkeeping and no conversion when switching modes.

mutable struct Camera
    eye::SVector{3,Float64}
    lookat::SVector{3,Float64}
    up::SVector{3,Float64}       # world up
    projection::Symbol           # :perspective | :ortho
    fov::Float64                 # field of view [deg] (perspective)
    ortho_half::Float64          # half-height of the view (ortho)
end

# oblique default pose framing the sphere (center, radius)
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
# so the view never flips over the pole.
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

orbit!(cam::Camera, dyaw, dpitch) = (cam.eye = rotate_about(cam.eye, cam.lookat, cam.up, dyaw, dpitch); cam)
flylook!(cam::Camera, dyaw, dpitch) = (cam.lookat = rotate_about(cam.lookat, cam.eye, cam.up, dyaw, dpitch); cam)

# clamped relative to the reference camera, so the limits scale with the scene
function zoom!(cam::Camera, factor, reference::Camera)
    if cam.projection === :ortho
        cam.ortho_half = clamp(cam.ortho_half * factor, 1e-4 * reference.ortho_half, 1e4 * reference.ortho_half)
    else
        v = cam.eye - cam.lookat
        cam.eye = cam.lookat + normalize(v) * clamp(norm(v) * factor, 1e-4 * distance(reference), 1e4 * distance(reference))
    end
    cam
end

# move eye and lookat together in the view plane; (dx, dy) are fractions of the view half-size
function pan!(cam::Camera, dx, dy)
    right, up, _ = view_basis(cam)
    scale = cam.projection === :ortho ? cam.ortho_half : distance(cam) * tand(cam.fov / 2)
    shift = (-dx * right + dy * up) * scale
    cam.eye += shift; cam.lookat += shift
    cam
end

function fly!(cam::Camera, dforward, dright, dup, speed)
    right, up, fwd = view_basis(cam)
    shift = (dforward * fwd + dright * right + dup * up) * speed
    cam.eye += shift; cam.lookat += shift
    cam
end

reset!(cam::Camera, default::Camera) =
    (foreach(n -> setproperty!(cam, n, getproperty(default, n)), fieldnames(Camera)); cam)

# No near/far: the z row is zero (depth comes from gl_FragDepth, §6.2); w = depth along forward (perspective) or 1.
# Eye space: +x right, +y up, +z = −forward.
function view_proj(cam::Camera, aspect)
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
