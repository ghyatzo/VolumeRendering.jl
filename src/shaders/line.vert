#version 410 core
// Spin axis as a screen-space-expanded quad: constant pixel width regardless of distance/zoom. The
// two endpoints are each emitted twice (±side); the perpendicular offset is computed in pixel space
// then converted to a clip-space shift, so width is exact in pixels. Depth = the endpoint's depth.
uniform mat4  viewProj;
uniform vec3  axisA, axisB;   // world endpoints
uniform vec2  viewportPx;     // geometry-pass pixel size
uniform float lineWidthPx;

void main() {
    int  endsel = gl_VertexID >> 1;                        // 0 = A, 1 = B
    float side  = (gl_VertexID & 1) == 0 ? -1.0 : 1.0;
    vec4 ca = viewProj * vec4(axisA, 1.0);
    vec4 cb = viewProj * vec4(axisB, 1.0);
    vec4 cur = endsel == 0 ? ca : cb;
    vec2 sa = ca.xy / ca.w, sb = cb.xy / cb.w;             // endpoints in NDC
    vec2 dir = normalize((sb - sa) * viewportPx);          // pixel-space line direction
    vec2 nrm = vec2(-dir.y, dir.x) * side;                 // pixel-space perpendicular
    vec2 offNdc = nrm * (lineWidthPx * 0.5) * 2.0 / viewportPx;   // px → NDC (NDC spans 2 over viewport)
    gl_Position = vec4(cur.xy + offNdc * cur.w, cur.z, cur.w);
}
