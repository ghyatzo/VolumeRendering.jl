#version 410 core
// Line/point geometry (streamlines, arrow glyphs) for the vector-field overlay. `viewProj` is the
// same matrix the volume ray-march uses for its gl_FragDepth, so depths are consistent and the
// depth test occludes lines correctly against the volume.
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 col;
uniform mat4 viewProj;
out vec3 vcol;
void main() {
    vcol = col;
    gl_Position = viewProj * vec4(pos, 1.0);
}
