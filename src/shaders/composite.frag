#version 410 core
// Composite the low-res volume over the full-res geometry: drawn into the geometry FBO with
// premultiplied-alpha blending (GL_ONE, GL_ONE_MINUS_SRC_ALPHA), so the volume texture's premultiplied
// RGBA gives out = volRGB + (1-volA)·geomRGB. The volume's GL_LINEAR sampler upscales it.
in vec2 vuv;
out vec4 frag;
uniform sampler2D volTex;
void main() { frag = texture(volTex, vuv * 0.5 + 0.5); }
