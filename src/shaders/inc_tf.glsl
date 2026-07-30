// Transfer-function lookup shared by the volume march and the slice sheet: normalize a value into
// [0,1] over the [lo,hi] window (log10 units if logScale==1) and sample the 256×1 RGBA TF texture.
uniform sampler2D tfTex;
uniform int   logScale;
uniform float lo, hi;

float normVal(float v) {
    float t = (logScale == 1) ? (log(max(v, 1e-30)) / 2.302585093 - lo) / (hi - lo)
                              : (v - lo) / (hi - lo);
    return clamp(t, 0.0, 1.0);
}
vec4 tf(float v) { return texture(tfTex, vec2(normVal(v), 0.5)); }
