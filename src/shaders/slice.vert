#version 410 core
// Slice sheet as a world-space quad on the plane {sliceAxis = slicePos}, generated from gl_VertexID
// (a 4-vertex triangle strip — no VBO). The quad covers the axis-aligned box [boxLo, boxHi] over the
// two non-sliceAxis dims. Depth is rasterized automatically.
uniform mat4  viewProj;
uniform int   sliceAxis;   // 0 = x, 1 = y, 2 = z
uniform float slicePos;
uniform vec3  boxLo, boxHi;
out vec3 worldPos;

void main() {
    vec2 q = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));  // corners of [0,1]²
    vec3 p;
    if (sliceAxis == 0)
        p = vec3(slicePos, mix(boxLo.y, boxHi.y, q.x), mix(boxLo.z, boxHi.z, q.y));
    else if (sliceAxis == 1)
        p = vec3(mix(boxLo.x, boxHi.x, q.x), slicePos, mix(boxLo.z, boxHi.z, q.y));
    else
        p = vec3(mix(boxLo.x, boxHi.x, q.x), mix(boxLo.y, boxHi.y, q.y), slicePos);
    worldPos = p;
    gl_Position = viewProj * vec4(p, 1.0);
}
