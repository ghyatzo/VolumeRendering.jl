// Shared camera ray generation (volume march + analytic impostors). `vuv` ∈ [-1,1]² is the pixel
// NDC. Perspective fans rays from the eye; ortho offsets parallel rays. This is the exact inverse of
// camera.jl `view_proj`, so impostors register with the volume by construction.
uniform int   perspective;   // 1 = perspective, 0 = ortho
uniform float fovscale;      // perspective: tan(fov/2); ortho: half-width
uniform float aspect;
uniform vec3  camEye, camRight, camUp, camFwd;

void cameraRay(vec2 vuv, out vec3 ro, out vec3 rd) {
    if (perspective == 1) {
        ro = camEye;
        rd = normalize(camFwd + vuv.x * fovscale * aspect * camRight + vuv.y * fovscale * camUp);
    } else {
        rd = normalize(camFwd);
        ro = camEye + (vuv.x * fovscale * aspect) * camRight + (vuv.y * fovscale) * camUp;
    }
}
