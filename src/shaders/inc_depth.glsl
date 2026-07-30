// Write a world point's depth in the ONE convention every geometry pass shares (`viewProj` = the
// camera.jl `view_proj` matrix). Impostors call this; rasterized geometry (quads, lines) gets the
// same value automatically from gl_Position — so all overlays live in one depth space and the volume
// can unproject it back with `invViewProj`.
uniform mat4 viewProj;

void writeDepth(vec3 worldPos) {
    vec4 clip = viewProj * vec4(worldPos, 1.0);
    float ndcZ = clip.z / clip.w;
    // Clip the impostor exactly like rasterized geometry: discard hits behind the eye or outside the
    // near/far planes rather than clamping, so impostors and real geometry fail identically at the frustum.
    if (clip.w <= 0.0 || ndcZ < -1.0 || ndcZ > 1.0) discard;
    gl_FragDepth = 0.5 * ndcZ + 0.5;
}
