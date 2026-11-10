#version 410 core
// Color the nearest region boundary inside the volume's bounding box; clipped surfaces write no depth.
in vec2 vuv;
out vec4 frag;

//#include "inc_camera.glsl"
//#include "inc_depth.glsl"
//#include "inc_tf.glsl"
//#include "field"
//#include "region"

uniform vec3 fieldLo, fieldHi;

void main() {
    vec3 ro, rd;
    cameraRay(vuv, ro, rd);
    vec2 tr = intersectRegion(ro, rd);
    if (tr.y <= tr.x) discard;             // ray misses the region
    vec3 t0 = (fieldLo - ro) / rd, t1 = (fieldHi - ro) / rd;
    vec3 tmin = min(t0, t1), tmax = max(t0, t1);
    float clipNear = max(max(tmin.x, tmin.y), tmin.z);
    float clipFar = min(min(tmax.x, tmax.y), tmax.z);
    float t = tr.x;
    // The back surface remains visible through a clipped front surface.
    if (t <= 0.0 || t < clipNear || t > clipFar) t = tr.y;
    if (t <= 0.0 || t < clipNear || t > clipFar) discard;
    vec3 p = ro + rd * t;
    frag = vec4(sample4(p).rgb, 1.0);
    writeDepth(p);
}
