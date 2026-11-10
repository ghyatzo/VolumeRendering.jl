@testset "surface slices clipped to grid bounds" begin
    w = 128
    xy = -12.0:3.0:12.0
    z = -5.0:5.0:5.0
    field = KeyedArray(ones(Float32, length(xy), length(xy), length(z)), (xy, xy, z))
    pixels = range(-12 + 12 / w, 12 - 12 / w; length = w)
    radius2 = [x^2 + y^2 for x in pixels, y in pixels]
    ring = (75 .<= radius2) .& (radius2 .< 100)
    band = [x^2 + z^2 < 100 && abs(z) <= 5 for x in pixels, z in pixels]

    function surfaceview(source; eye = (0, 0, 30), lookat = (0, 0, 0), up = (0, 1, 0), center = (0, 0, 0))
        view = FieldView(source; overlays = [SurfaceSliceOverlay(SphereRegion(; center, radius = 10))])
        view.params.refine_samples = 1
        view.camera.projection = :ortho
        view.camera.eye = eye
        view.camera.lookat = lookat
        view.camera.up = up
        view.camera.ortho_half = 12
        view
    end

    @testset "equatorial band from above and from the side" begin
        @test (depthbuf(surfaceview(field), w) .< 1) == ring
        side = surfaceview(field; eye = (0, 30, 0), up = (0, 0, 1))
        @test (depthbuf(side, w) .< 1) == band
        inside = surfaceview(field; eye = (0, 0, 0), lookat = (0, 1, 0), up = (0, 0, 1))
        @test (depthbuf(inside, w) .< 1) == band
        @test GL.glGetError() == GL.GL_NO_ERROR
    end

    @testset "back surface through the clipped front" begin
        view = surfaceview(field; center = (0, 0, 2))
        depth = depthbuf(view, w)
        visible = (51 .<= radius2) .& (radius2 .< 100)
        @test (depth .< 1) == visible
        vp = view_proj(view.camera, bounds(view.region)..., 1.0)
        expected = ones(Float32, w, w)
        for I in CartesianIndices(expected)
            visible[I] || continue
            height = sqrt(100 - radius2[I])
            zhit = height <= 3 ? 2 + height : 2 - height
            clip = vp * SVector(pixels[I[1]], pixels[I[2]], zhit, 1.0)
            expected[I] = (clip[3] / clip[4] + 1) / 2
        end
        @test maximum(abs.(depth .- expected)) <= 2e-6
        @test GL.glGetError() == GL.GL_NO_ERROR
    end

    @testset "reversed axes and field changes" begin
        view = surfaceview(field)
        @test (depthbuf(view, w) .< 1) == ring
        reversed = KeyedArray(parent(field), (reverse(xy), reverse(xy), reverse(z)))
        view.field = reversed
        view.region = region(reversed)
        @test (depthbuf(view, w) .< 1) == ring

        small = KeyedArray(ones(Float32, 3, 3, 3), (-2:2:2, -2:2:2, -2:2:2))
        view.field = small
        view.region = region(small)
        @test all(==(1), depthbuf(view, w))
        @test GL.glGetError() == GL.GL_NO_ERROR
    end
end
