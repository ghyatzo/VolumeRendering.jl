#version 410 core
// Flat, hard-edged screen-space line. Color is a uniform so LineOverlay/AxesOverlay can tint each line.
uniform vec3 lineColor;
out vec4 frag;
void main() { frag = vec4(lineColor, 1.0); }
