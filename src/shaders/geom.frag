#version 410 core
in vec3 vcol;
out vec4 frag;
void main() { frag = vec4(vcol, 1.0); }
