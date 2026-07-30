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
    // Near-plane clip BEFORE the screen-space divide: an endpoint behind the camera (w<0) would give a
    // corrupted screen position AND depth (cur.z/cur.w), letting the occluded half paint over geometry.
    // Move any behind-plane endpoint onto the near plane (z_ndc = -1, i.e. clip.z + clip.w = 0) so both
    // quad endpoints have valid w>0. Segments fully behind the near plane are culled off-screen.
    float da = ca.z + ca.w;                                // signed distance to near plane (>=0 = in front)
    float db = cb.z + cb.w;
    if (da < 0.0 && db < 0.0) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
    if (da < 0.0)      { ca = mix(ca, cb, da / (da - db)); }
    else if (db < 0.0) { cb = mix(cb, ca, db / (db - da)); }
    vec4 cur = endsel == 0 ? ca : cb;
    vec2 sa = ca.xy / ca.w, sb = cb.xy / cb.w;             // endpoints in NDC
    vec2 dir = normalize((sb - sa) * viewportPx);          // pixel-space line direction
    vec2 nrm = vec2(-dir.y, dir.x) * side;                 // pixel-space perpendicular
    vec2 offNdc = nrm * (lineWidthPx * 0.5) * 2.0 / viewportPx;   // px → NDC (NDC spans 2 over viewport)
    gl_Position = vec4(cur.xy + offNdc * cur.w, cur.z, cur.w);
}
