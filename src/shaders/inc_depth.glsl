// Write a world point's depth in the ONE convention every geometry pass shares (`viewProj` = the
// camera.jl `view_proj` matrix). Impostors call this; rasterized geometry (quads, lines) gets the
// same value automatically from gl_Position — so all overlays live in one depth space and the volume
// can unproject it back with `invViewProj`.
uniform mat4 viewProj;

void writeDepth(vec3 worldPos) {
    vec4 clip = viewProj * vec4(worldPos, 1.0);
    gl_FragDepth = clamp(0.5 * clip.z / clip.w + 0.5, 0.0, 1.0);
}
