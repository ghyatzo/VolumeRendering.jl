#version 410 core
// Opaque cross-section sheet: color the field via the same TF as the volume; fills the whole
// [boxLo, boxHi] quad (outside the data → tf(0)). Rasterized depth (from slice.vert) handles occlusion.
in vec3 worldPos;
out vec4 frag;

//#include "field"
//#include "inc_tf.glsl"

void main() {
    frag = vec4(tf(sampleField(worldPos)).rgb, 1.0);
}
