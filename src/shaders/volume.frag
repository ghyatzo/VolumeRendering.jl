#version 410 core
//
// Volume ray-marcher for a generic scalar field, sampled through the injected `field` snippet
// (sampleField/stepSize) inside the region defined by the injected `region` snippet (intersectRegion).
// Overlays (slices, spheres, axes, streamlines) are NOT drawn here — they are the full-res geometry
// G-buffer. This pass clips the march at the nearest geometry depth and outputs PREMULTIPLIED RGBA,
// which the composite blends over the geometry.
//
in vec2 vuv;
out vec4 frag;

//#include "inc_camera.glsl"
//#include "field"
//#include "region"
//#include "inc_tf.glsl"

uniform int   mode;          // 0 = emission-absorption (DVR), 1 = MIP, 2 = average
uniform float opacityScale;  // DVR extinction per unit length (σ = tfAlpha·opacityScale)
uniform int   steps;         // hard iteration cap for the adaptive march (not a fixed sample count)
uniform float stepScale;     // adaptive step: dt = stepScale·stepSize(pos) — the quality knob
uniform mat4  invViewProj;   // unproject sampled scene depth → world (matches viewProj)
uniform sampler2D geomDepthTex;   // full-res geometry depth; clip the march at it
uniform ivec2 geomSize;      // full-res geometry-pass size (gw,gh)
uniform ivec2 volSize;       // this pass's low-res size (vw,vh)

void main() {
    vec3 ro, rd;
    cameraRay(vuv, ro, rd);

    // Intersect the region (near/far ray parameters) via the injected `region` snippet.
    vec2 tr = intersectRegion(ro, rd);
    float tn   = tr.x;
    float tfar = tr.y;

    // Clip the far end at the nearest opaque geometry. Min-reduce the geometry depth over this low-res
    // cell's FULL-res footprint (not one tap) so sub-pixel-thin lines are never missed — otherwise a
    // thin line falls between cell centers, the volume marches the whole box, and the line is dashed /
    // over-occluded. Total taps ≈ the full-res depth size, so this is ~one depth read. d==1 → no
    // geometry → full box.
    ivec2 fc = ivec2(gl_FragCoord.xy);
    ivec2 b0 = fc * geomSize / volSize, b1 = (fc + 1) * geomSize / volSize;
    float d = 1.0;
    for (int y = b0.y; y < b1.y; y++)
        for (int x = b0.x; x < b1.x; x++)
            d = min(d, texelFetch(geomDepthTex, ivec2(x, y), 0).r);
    if (d < 1.0) {
        vec4 w = invViewProj * vec4(vuv, 2.0 * d - 1.0, 1.0);
        tfar = min(tfar, length(w.xyz / w.w - ro));
    }

    vec3 col = vec3(0.0);        // premultiplied accumulated color (DVR)
    float alpha = 0.0;
    float vmax = 0.0, vsum = 0.0, vlen = 0.0;

    if (tfar > tn) {
        // Adaptive marching: the field's local cell size is reported by stepSize(pos), so the sample
        // spacing tracks it: dt = stepScale·stepSize(pos). One law resolves fine regions and stops
        // over-sampling coarse ones; `steps` is only a cap. stepSize already encapsulates any floor, so
        // rays never take vanishing steps.
        // Deterministic per-pixel jitter offsets the first sample by a sub-step so the sampling lattice
        // shows up as (stable, frame-invariant) fine noise instead of coherent banding / ringing.
        float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
        float t = tn + jitter * (stepScale * stepSize(ro + rd * tn));
        for (int s = 0; s < steps && t < tfar; s++) {
            vec3 pos = ro + rd * t;
            float dt = stepScale * stepSize(pos);
            float v = sampleField(pos);
            if (mode == 0) {                               // emission-absorption (DVR)
                vec4 c = tf(v);
                float av = 1.0 - exp(-c.a * opacityScale * dt);   // dt-correct opacity → step-count invariant
                col   += (1.0 - alpha) * av * c.rgb;
                alpha += (1.0 - alpha) * av;
                if (alpha > 0.995) break;
            } else if (mode == 1) {                        // MIP
                vmax = max(vmax, v);
            } else {                                       // average
                vsum += v * dt; vlen += dt;                // dt-weighted spatial mean (spacing varies)
            }
            t += dt;
        }
    }

    if (tfar <= tn) { frag = vec4(0.0); return; }   // ray missed the box (or fully clipped) → transparent

    // Premultiplied output. MIP/avg have no intrinsic opacity → the layer alpha is the TF opacity of
    // the reduced value, so aids show through faint regions and are hidden behind bright ones.
    if (mode == 1) {
        vec4 c = tf(vmax);       frag = vec4(c.rgb * c.a, c.a);
    } else if (mode == 2) {
        float mv = vlen > 0.0 ? vsum / vlen : 0.0;
        vec4 c = tf(mv);         frag = vec4(c.rgb * c.a, c.a);
    } else {
        frag = vec4(col, alpha); // DVR: col already premultiplied
    }
}
