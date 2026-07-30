#version 410 core
// Fullscreen triangle: 3 vertices cover the viewport, no vertex buffer needed.
// vuv spans [-1,1]^2 across the screen and drives the per-pixel camera ray.
out vec2 vuv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    vuv = p * 2.0 - 1.0;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
