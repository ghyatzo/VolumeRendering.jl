#version 410 core
// Sphere as an analytic impostor over the fullscreen triangle: ray–sphere at sphereCenter, flat
// sphereColor, writing the hit's depth so it occludes / is occluded like any geometry.
in vec2 vuv;
out vec4 frag;

//#include "inc_camera.glsl"
//#include "inc_depth.glsl"

uniform vec3  sphereCenter;
uniform float sphereRadius;
uniform vec3  sphereColor;

void main() {
    vec3 ro, rd;
    cameraRay(vuv, ro, rd);
    // |ro + t rd - center|² = sphereRadius², rd normalized ⇒ t² + 2b t + c = 0.
    vec3 oc = ro - sphereCenter;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - sphereRadius * sphereRadius;
    float disc = b * b - c;
    if (disc < 0.0) discard;
    float s = sqrt(disc);
    float t = -b - s;                 // near root
    if (t < 0.0) t = -b + s;          // eye inside the sphere → far root
    if (t < 0.0) discard;             // sphere entirely behind the eye
    frag = vec4(sphereColor, 1.0);
    writeDepth(ro + rd * t);
}
