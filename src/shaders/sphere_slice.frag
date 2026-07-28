#version 410 core
// RadialSliceOverlay: spherical "slice" — analytic ray/sphere impostor over the fullscreen
// triangle, sampling the field on the hit point of the sphere of radius `sphereRadius` centered at
// `sphereCenter`. Two independent, composable passes:
//   showSurface == 1  → fragment colored through the transfer function (tf(v).rgb), like slice.frag.
//   showIso    == 1  → a Gaussian iso-contour band centered on isoValue, width isoSigma (in field
//                       units), painted in isoColor and composited over the surface color via
//                       per-fragment `mix`. Both may be on simultaneously (surface + contour on top).
// Always opaque (writes depth via inc_depth.glsl), so it occludes / is occluded like any geometry.
in vec2 vuv;
out vec4 frag;

//#include "inc_camera.glsl"
//#include "inc_depth.glsl"
//#include "inc_tf.glsl"
//#include "field"

uniform vec3  sphereCenter;
uniform float sphereRadius;
uniform int   showSurface, showIso;
uniform float isoValue, isoSigma;
uniform vec3  isoColor;

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
    if (t < 0.0) discard;            // sphere entirely behind the eye
    vec3 p = ro + rd * t;
    float v = sampleField(p);
    float a = (showIso == 1)
        ? exp(-(v - isoValue) * (v - isoValue) / (2.0 * isoSigma * isoSigma))
        : 0.0;
    vec3 col;
    if (showSurface == 1) {
        // TF-colored shell; the iso contour (if on) is composited on top via per-fragment mix, so
        // off-band regions still show the shell and the band tints toward isoColor.
        col = mix(tf(v).rgb, isoColor, a);
    } else if (showIso == 1) {
        if (a < 0.01) discard;         // iso-only: keep just the band, cull the rest of the sphere
        col = isoColor;
    } else {
        discard;                      // both flags off: nothing to draw
    }
    frag = vec4(col, 1.0);
    writeDepth(p);
}