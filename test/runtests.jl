# NOTE: this suite uses a single shared GLFW/OpenGL context for all render checks (rather than the
# usual per-@testitem isolation) — every test needs a live GL context, and one shared hidden context
# is both faster and avoids multiple concurrent contexts. All GL work happens inside `withctx`.

using Test
using VolumeRendering, AxisKeys, StaticArrays
using ColorTypes, FixedPointNumbers, Unitful
using LinearAlgebra: norm, normalize
import GLFW, ModernGL as GL

# ── headless GL context + pixel readback helpers ──
function withctx(f; w = 64, h = 64)
    GLFW.Init()
    GLFW.WindowHint(GLFW.VISIBLE, false)
    GLFW.WindowHint(GLFW.CONTEXT_VERSION_MAJOR, 4); GLFW.WindowHint(GLFW.CONTEXT_VERSION_MINOR, 1)
    GLFW.WindowHint(GLFW.OPENGL_PROFILE, GLFW.OPENGL_CORE_PROFILE)
    GLFW.WindowHint(GLFW.OPENGL_FORWARD_COMPAT, true)
    win = GLFW.CreateWindow(w, h, "VolumeRendering tests"); GLFW.MakeContextCurrent(win)
    try f() finally GLFW.DestroyWindow(win); GLFW.Terminate() end
end
function renderbuf(view, w)
    tex = render!(view, w, w)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, view.vr.geom_fb.fbo)
    buf = Array{UInt8}(undef, 4, w, w)
    GL.glReadPixels(0, 0, w, w, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, buf); GL.glFinish()
    (tex, buf)
end
# Geometry depth only (the composite pass leaves it untouched), so overlay coverage can be checked
# without the volume's contribution mixed in. 1 == nothing drawn.
function depthbuf(view, w)
    render!(view, w, w)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, view.vr.geom_fb.fbo)
    buf = Array{Float32}(undef, w, w)
    GL.glReadPixels(0, 0, w, w, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, buf); GL.glFinish()
    buf
end
litfrac(b) = count(>(4), @view b[1:3, :, :]) / (3 * size(b, 2) * size(b, 3))

const W = 256
gaussian(xs, ys, zs; c = (0.0, 0.0, 0.0), s = 0.2) =
    Float64[exp(-((x-c[1])^2 + (y-c[2])^2 + (z-c[3])^2) / s) for x in xs, y in ys, z in zs]

# a custom (non-AbstractRange) uniform axis, to exercise the axis_index_glsl extension seam
struct UniformAxis <: AbstractVector{Float64}; a::Float64; d::Float64; n::Int; end
Base.size(k::UniformAxis) = (k.n,)
Base.getindex(k::UniformAxis, i::Int) = k.a + k.d * (i - 1)
VolumeRendering.axis_index_glsl(k::UniformAxis, c) = "((" * c * ") - " * repr(k.a) * ") * " * repr(inv(k.d))

@testset "VolumeRendering" begin
    @testset "construction needs no GL context" begin
        xs = ys = zs = -1.0:0.5:1.0
        @test FieldView(KeyedArray(gaussian(xs, ys, zs); x = xs, y = ys, z = zs)).tf.tex == 0
        @test FieldView(nothing; region = SphereRegion(radius = 1.0), overlays = [SphereOverlay(radius = 0.5)]).tf.tex == 0
    end

    withctx() do
        xs = ys = zs = -1.0:0.05:1.0

        @testset "KeyedArray volume renders" begin
            v = FieldView(KeyedArray(gaussian(xs, ys, zs); x = xs, y = ys, z = zs))
            _, b = renderbuf(v, W)
            @test GL.glGetError() == 0
            @test litfrac(b) > 0.02
        end

        @testset "coordinate mapping (x>0 field → image right)" begin
            step = Float64[x > 0 ? 1.0 : 0.0 for x in xs, y in ys, z in zs]
            v = FieldView(KeyedArray(step; x = xs, y = ys, z = zs))
            v.params.mode = 1                                   # MIP
            v.camera.projection = :ortho
            v.camera.eye = SVector(0.0, 0.0, 3.0); v.camera.lookat = SVector(0.0, 0.0, 0.0)
            v.camera.up = SVector(0.0, 1.0, 0.0); v.camera.ortho_half = 1.2
            _, b = renderbuf(v, W)
            lum = sum(Int.(b[1:3, :, :]); dims = 1)[1, :, :]
            left = sum(@view lum[1:W÷2, :]); right = sum(@view lum[W÷2+1:W, :])
            @test right > 3 * max(left, 1)                      # +x maps to camera-right
        end

        @testset "RGBA color field: intrinsic color + placement" begin
            # red where x>0, green where x<0; the field carries its own color (no colormap).
            grid = [RGBA{N0f8}(x > 0 ? 1.0 : 0.0, x > 0 ? 0.0 : 1.0, 0.0, 0.5) for x in xs, y in ys, z in zs]
            A = KeyedArray(grid; x = xs, y = ys, z = zs)
            @test uses_transfer_function(A) == false
            v = FieldView(A)
            v.camera.projection = :ortho
            v.camera.eye = SVector(0.0, 0.0, 3.0); v.camera.lookat = SVector(0.0, 0.0, 0.0)
            v.camera.up = SVector(0.0, 1.0, 0.0); v.camera.ortho_half = 1.2
            _, b = renderbuf(v, W)
            @test GL.glGetError() == 0
            @test litfrac(b) > 0.02
            R, G = Int.(b[1, :, :]), Int.(b[2, :, :])
            rR, rG = sum(@view R[W÷2+1:W, :]), sum(@view G[W÷2+1:W, :])   # +x half (camera-right)
            lR, lG = sum(@view R[1:W÷2, :]),   sum(@view G[1:W÷2, :])     # -x half
            @test rR > 3 * max(rG, 1)     # right half is red
            @test lG > 3 * max(lR, 1)     # left half is green
        end

        @testset "GLSLField (analytic) + SphereRegion" begin
            f = GLSLField("""
                float sampleField(vec3 p){ return exp(-dot(p,p)*0.7); }
                float stepSize(vec3 p){ return 0.05; }""";
                region = SphereRegion(radius = 3.0), value_range = (0.0, 1.0))
            _, b = renderbuf(FieldView(f), W)
            @test GL.glGetError() == 0
            @test litfrac(b) > 0.02
        end

        @testset "custom axis type == equivalent range axis (pixel-identical)" begin
            blob = gaussian(xs, ys, zs)
            _, bref = renderbuf(FieldView(KeyedArray(blob; x = xs, y = ys, z = zs)), W)
            ax = UniformAxis(-1.0, 0.05, length(xs))
            _, bcus = renderbuf(FieldView(KeyedArray(blob; x = ax, y = ys, z = zs)), W)
            @test bref == bcus
        end

        @testset "all overlays render without GL error" begin
            A = KeyedArray(gaussian(xs, ys, zs); x = xs, y = ys, z = zs)
            vf = VectorField(p -> SVector(-p[2], p[1], 0.2); region = BoxRegion((-1.,-1.,-1.), (1.,1.,1.)))
            ovs = Overlay[SliceOverlay(:z), BoxOutlineOverlay(), SphereOverlay(radius = 0.4),
                          SphereOutlineOverlay(region = SphereRegion(radius = 0.6)),
                          SurfaceSliceOverlay(SphereRegion(radius = 0.4)),
                          SpherePolylinesOverlay([[SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0)]];
                                                 region = SphereRegion(radius = 0.6)),
                          AxesOverlay(), StreamlinesOverlay(vf), GlyphsOverlay(vf; every = 8)]
            _, b = renderbuf(FieldView(A; overlays = ovs), W)
            @test GL.glGetError() == 0
            @test litfrac(b) > 0.05
        end

        @testset "surface slice" begin
            A = KeyedArray(gaussian(xs, ys, zs); x = xs, y = ys, z = zs)
            topdown!(v) = (v.camera.projection = :ortho;
                           v.camera.eye = SVector(0.0, 0.0, 3.0); v.camera.lookat = SVector(0.0, 0.0, 0.0);
                           v.camera.up = SVector(0.0, 1.0, 0.0); v.camera.ortho_half = 1.2; v)

            # Face-on from +z, the box's near boundary IS the plane z = 1, so the two slice overlays
            # must agree pixel for pixel.
            _, bplane = renderbuf(topdown!(FieldView(A; overlays = [SliceOverlay(:z; pos = 1.0)])), W)
            _, bsurf = renderbuf(topdown!(FieldView(A;
                overlays = [SurfaceSliceOverlay(BoxRegion((-1., -1., -1.), (1., 1., 1.)))])), W)
            @test bsurf == bplane

            # Silhouette seen from outside: the same pixels as sphere.frag's independent root solve.
            c = SVector(0.1, 0.0, 0.0)
            dsurf = depthbuf(FieldView(A; overlays = [SurfaceSliceOverlay(SphereRegion(; center = c, radius = 0.5))]), W)
            dimp = depthbuf(FieldView(A; overlays = [SphereOverlay(; center = c, radius = 0.5)]), W)
            @test (dsurf .< 1) == (dimp .< 1)
            @test any(<(1), dsurf) && !all(<(1), dsurf)      # a silhouette, so the above isn't vacuous

            # Eye inside → the far surface, wrapping the whole view. Only works while intersectRegion
            # keeps tnear's sign.
            v = FieldView(A; overlays = [SurfaceSliceOverlay(SphereRegion(radius = 0.5))])
            v.camera.eye = SVector(0.1, 0.0, 0.0); v.camera.lookat = SVector(0.0, 0.0, 0.0)
            @test all(<(1), depthbuf(v, W))
        end

        @testset "sphere outline graticule" begin
            @test VolumeRendering._parallel_lats(3) ≈ [-π/4, 0, π/4]
            @test isempty(VolumeRendering._parallel_lats(0))
            @test VolumeRendering._meridian_lons(6) ≈ [0, π/3, 2π/3, π, 4π/3, 5π/3]
            @test isempty(VolumeRendering._meridian_lons(0))
            @test VolumeRendering._parallel_lats([0.1, 0.2]) == [0.1, 0.2]     # explicit angles pass through

            # equator only, 8 segments ⇒ one closed ring of 9 directions on z = 0
            (eq,) = VolumeRendering._graticule([0.0], 0, 8)
            @test length(eq) == 9
            @test eq[1] ≈ eq[end] ≈ SVector(1.0, 0.0, 0.0)
            @test getindex.(eq, 3) == zeros(9)

            # a meridian is half a circle, so it gets half the segments: pole to pole in 9 points
            (mer,) = VolumeRendering._graticule(0, [0.0], 16)
            @test length(mer) == 9
            @test mer[1] ≈ SVector(0.0, 0.0, -1.0) && mer[end] ≈ SVector(0.0, 0.0, 1.0)

            # angles only ever reach sin/cos, so Unitful angles work with no conversion
            @test VolumeRendering._graticule([30u"°"], 0, 8) ≈ VolumeRendering._graticule([π/6], 0, 8)
        end

        @testset "curves circumscribing a sphere" begin
            midpoints(pl) = [(a + b) / 2 for (a, b) in zip(pl, @view pl[2:end])]
            x, y, o = SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 0.0)

            # a 90° step: both ends out to √2, the chord's midpoint touching the unit sphere
            @test VolumeRendering._circumscribed(o, 1.0, [[x, y]]) ≈ [[SVector(√2, 0, 0), SVector(0, √2, 0)]]
            # only directions count
            @test VolumeRendering._circumscribed(o, 1.0, [[2x, 5y]]) ≈ VolumeRendering._circumscribed(o, 1.0, [[x, y]])

            # the widest step sets the scale for all curves: its midpoint touches, the others' are outside
            c = VolumeRendering._circumscribed(o, 1.0, [[x, y], [x, normalize(x + y)]])
            @test norm.(midpoints(c[1])) ≈ [1.0]
            @test norm.(midpoints(c[2])) ≈ [cos(π/8) / cos(π/4)]

            # around a center: the equator of a radius-2 sphere at (1,0,0), in 8 steps of 45°
            ring = only(VolumeRendering._circumscribed(x, 2.0, VolumeRendering._graticule([0.0], 0, 8)))
            @test norm.(ring .- Ref(x)) ≈ fill(2.164784400584788, 9)         # 2 / cos(π/8)
            @test norm.(midpoints(ring) .- Ref(x)) ≈ fill(2.0, 8)

            # a parallel's step is shorter than its longitude step: σ with cos σ = sin²φ + cos²φ·cos(45°)
            par = only(VolumeRendering._circumscribed(o, 1.0, VolumeRendering._graticule([π/6], 0, 8)))
            @test norm.(par) ≈ fill(1.0598996098693099, 9)
            @test norm.(midpoints(par)) ≈ fill(1.0, 8)
        end

        @testset "sphere outline stays visible on a sphere of the same radius" begin
            # the meridian facing the camera, alone vs drawn on its sphere: not a pixel of it may be hidden
            # (with chords inside the sphere, their middles would be)
            outline = SphereOutlineOverlay(region = SphereRegion(radius = 1.0), parallels = 0, meridians = [0.0],
                                           segments = 16, color = RGB(0.0, 1.0, 0.0))
            # seen from +x, screen-right is +y; shifting the camera by half a pixel (1.3/256) puts the
            # vertical meridian on the center of a pixel column
            view(ovs) = (v = FieldView(nothing; region = SphereRegion(radius = 1.2), overlays = ovs);
                         v.camera.projection = :ortho; v.camera.eye = SVector(3.0, -0.005078125, 0.0);
                         v.camera.lookat = SVector(0.0, -0.005078125, 0.0); v.camera.ortho_half = 1.3; v)
            green(b) = count(@views (b[2, :, :] .== 255) .& (b[1, :, :] .== 0))
            alone = green(renderbuf(view([outline]), W)[2])
            @test alone ∈ 200:202      # one column, 2·1.0196/2.6 of the height: 2 / cos(π/16) tall
            @test green(renderbuf(view([SphereOverlay(radius = 1.0, color = RGB(0.2, 0.2, 0.2)), outline]), W)[2]) == alone
        end

        @testset "box outline: 12 axis-aligned edges" begin
            edges = VolumeRendering._boxoutline_polylines(SVector(0.0, 0.0, 0.0), SVector(2.0, 4.0, 6.0))
            @test all(==(2) ∘ length, edges)
            @test sort([norm(b - a) for (a, b) in edges]) == [2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6]
        end

        @testset "polylines" begin
            # consecutive point pairs, each vertex (x,y,z, r,g,b); single points draw nothing
            @test VolumeRendering._polylines_verts(
                    [[SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(1.0, 2.0, 0.0)], [SVector(5.0, 5.0, 5.0)]],
                    RGB(1.0, 0.5, 0.0)) ==
                  Float32[0, 0, 0, 1, 0.5, 0,  1, 0, 0, 1, 0.5, 0,
                          1, 0, 0, 1, 0.5, 0,  1, 2, 0, 1, 0.5, 0]

            # a horizontal line across x ∈ [-1, 1], seen face-on: 1 px tall at its row, spanning 2/2.4 of
            # the width. y = 1.2·(160.5/128 - 1) is the center of pixel row 160 (0-based).
            o = PolylinesOverlay([[SVector(-1.0, 0.3046875, 0.0), SVector(0.0, 0.3046875, 0.0), SVector(1.0, 0.3046875, 0.0)]])
            v = FieldView(nothing; region = SphereRegion(radius = 1.0), overlays = [o])
            v.camera.projection = :ortho
            v.camera.eye = SVector(0.0, 0.0, 3.0); v.camera.lookat = SVector(0.0, 0.0, 0.0)
            v.camera.up = SVector(0.0, 1.0, 0.0); v.camera.ortho_half = 1.2
            litrows(d) = unique(getindex.(findall(<(1), d), 2))
            d = depthbuf(v, W)
            @test litrows(d) == [161]
            @test count(<(1), d) ∈ 212:214

            # an in-place edit is seen too: the line moves to the center of row 96
            o.polylines[1] .= [SVector(-1.0, -0.2953125, 0.0), SVector(0.0, -0.2953125, 0.0), SVector(1.0, -0.2953125, 0.0)]
            @test litrows(depthbuf(v, W)) == [97]
        end

        @testset "no volume: overlays alone" begin
            v = FieldView(nothing; region = SphereRegion(radius = 1.0),
                          overlays = [SphereOverlay(radius = 0.5, color = RGB(1.0, 0.0, 0.0))])
            v.camera.projection = :ortho
            v.camera.eye = SVector(0.0, 0.0, 3.0); v.camera.lookat = SVector(0.0, 0.0, 0.0)
            v.camera.up = SVector(0.0, 1.0, 0.0); v.camera.ortho_half = 1.2
            tex, b = renderbuf(v, W)
            @test GL.glGetError() == 0
            @test b[:, W÷2, W÷2] == [255, 0, 0, 255]               # the sphere, uncomposited
            @test b[:, 1, 1] == [0, 0, 0, 255]                     # background
            @test count(>(0), @view b[1, :, :]) / W^2 ≈ π * 0.5^2 / 2.4^2  rtol = 0.02
            @test render!(v, W, W) == tex && v.vr.nsamp == 1       # unchanged ⇒ cached, no refinement frames

            # a field switched in and out: the volume appears, then the overlay-only image returns exactly
            v.field = KeyedArray(gaussian(xs, ys, zs); x = xs, y = ys, z = zs)
            _, bvol = renderbuf(v, W)
            @test litfrac(bvol) > litfrac(b) + 0.02
            v.field = nothing
            @test renderbuf(v, W)[2] == b
        end

        @testset "overlay fingerprint tracks state, not GL resources" begin
            o = BoxOutlineOverlay(color = RGB(1.0, 0.0, 0.0))
            fp = VolumeRendering.overlay_fingerprint(o)
            o.gl.key = (1, 2, 3)                                  # GL cache: not part of the image
            @test VolumeRendering.overlay_fingerprint(o) == fp
            o.color = RGB(0.0, 1.0, 0.0)
            @test VolumeRendering.overlay_fingerprint(o) != fp
        end

        @testset "field switch re-uploads (same-shape, different data)" begin
            # A1/A2 have identical shape+axes → identical field_glsl → the volume program is NOT
            # rebuilt on switch; the texture MUST still be re-uploaded so the image reflects A2.
            A1 = KeyedArray(gaussian(xs, ys, zs; c = (-0.5, 0.0, 0.0), s = 0.1); x = xs, y = ys, z = zs)
            A2 = KeyedArray(gaussian(xs, ys, zs; c = ( 0.5, 0.0, 0.0), s = 0.1); x = xs, y = ys, z = zs)
            @test field_glsl(A1) == field_glsl(A2)     # same GLSL: only the data differs
            v = FieldView(A1); (_, b1) = renderbuf(v, W)
            v.field = A2;      (_, b2) = renderbuf(v, W)
            @test b1 != b2                             # image reflects the new data (no stale texture)
        end

        @testset "multi-view isolation (two views, one context)" begin
            A  = KeyedArray(gaussian(xs, ys, zs); x = xs, y = ys, z = zs)
            A2 = KeyedArray(gaussian(xs, ys, zs; c = (0.5, 0.0, 0.0), s = 0.1); x = xs, y = ys, z = zs)
            va = FieldView(A); vb = FieldView(A2)
            _, ba = renderbuf(va, W)
            renderbuf(vb, W)                                    # render B in between
            va.last_key[] = nothing                             # force A to re-render (same inputs)
            _, ba2 = renderbuf(va, W)
            @test ba == ba2                                     # B did not corrupt A
            @test va.vr.geom_fb.tex != vb.vr.geom_fb.tex        # separate GL resources
        end
    end
end

import Aqua
import CompatHelperLocal as CHL
@testset "_" begin
    Aqua.test_all(VolumeRendering; ambiguities=false)
    Aqua.test_ambiguities(VolumeRendering)
    CHL.@check()
end
