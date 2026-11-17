# Limitations

- Line depth is constant across the line's width: lines lying exactly on a sloped surface can lose pixels at their edges — place them slightly above.
- Lines are hard-edged (no anti-aliasing); width is in device pixels.
- Glyphs can contain NaN arrows where UniformStreamlines' tracer stalls in place (e.g. at a field source): its tangent there is 0/0.
- Translucent marks composite in list order, unsorted; a translucent surface inside an earlier volume appears in front of it.
- Overlapping volumes are marched separately, not jointly.
- Shapes must be convex.
- Coplanar opaque faces resolve by list order (later on top), not by geometry.
- Objects closer than ~1e-6 × the camera distance tie in depth.
- Fields and shapes in one program share one GLSL namespace: helper function names can collide (not checked).
- At most 16 samplers per program, including the renderer's colormap/extinction/scene-depth ones.
- Arrays held by marks and fields are tracked by identity: assign a new array to change data; in-place edits are not seen.
- Marks removed from a view free their GPU resources on the next frame; re-adding re-uploads.
- A `(lon, lat)` grid used as a Volume source needs a region that excludes the origin.
- Grid axes other than plain `Real` ranges need an `axis_index_glsl` method; angle keys are radians.
- No `(lon, lat, r)` grids and no point marks yet.
