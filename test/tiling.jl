@testset "tiled fields" begin
    @testset "brick geometry and shared texels" begin
        for dims in ((8, 2, 2), (7, 2, 2), (9, 2, 2), (3, 8, 2), (3, 2, 8), (9, 7, 3))
            data = reshape(Float32.(1:prod(dims)), dims)
            field = KeyedArray(data, map(n -> 1:n, dims))
            tiled = TiledField(field; maxdim = 4)
            @test all(<=(4), tiled.D)
            for I in CartesianIndices(data)
                mapped = ntuple(d -> VolumeRendering._map_axis(tiled, d, I[d] - 1), 3)
                brick = ntuple(d -> mapped[d][1], 3)
                local_index = ntuple(d -> Int(mapped[d][2]) + 1, 3)
                @test VolumeRendering.brick_data(tiled, brick)[local_index...] == data[I]
            end
            for d in 1:3
                tiled.k[d] > 1 || continue
                a = VolumeRendering.brick_data(tiled, (0, 0, 0))
                b = VolumeRendering.brick_data(tiled, ntuple(i -> Int(i == d), 3))
                @test selectdim(a, d, tiled.D[d]) == selectdim(b, d, 1)
            end
        end
        field = KeyedArray(ones(Float32, 2, 2, 2), (1:2, 1:2, 1:2))
        @test_throws "maxdim must be at least 2" TiledField(field; maxdim = 1)
        empty_field = KeyedArray(zeros(Float32, 0, 2, 2), (1:0, 1:2, 1:2))
        @test_throws "axes must not be empty" TiledField(empty_field)

        many = KeyedArray(ones(Float32, 5, 5, 5), (1:5, 1:5, 1:5))
        view = TiledFieldView(many; maxdim = 2)
        for _ in 1:2
            @test_throws "only 14 field texture units" render!(view, 16, 16)
        end
    end

    @testset "brick allocations" begin
        dims = (64, 48, 8)
        data = reshape(Float32.(1:prod(dims)), dims)
        tiled = TiledField(KeyedArray(data, map(n -> 1:n, dims)); maxdim = 32)
        brick = VolumeRendering.brick_data(tiled, (0, 0, 0))
        @test brick == data[1:tiled.D[1], 1:tiled.D[2], 1:tiled.D[3]]
        @test (@allocated VolumeRendering.brick_data(tiled, (0, 0, 0))) <= 2sizeof(brick) + 16_384
    end

    @testset "surface clipping with tiled fields" begin
        w = 128
        xy = -12.0:3.0:12.0
        z = -5.0:5.0:5.0
        field = KeyedArray(ones(Float32, length(xy), length(xy), length(z)), (xy, xy, z))
        pixels = range(-12 + 12 / w, 12 - 12 / w; length = w)
        ring = [75 <= x^2 + y^2 < 100 for x in pixels, y in pixels]
        band = [x^2 + z^2 < 100 && abs(z) <= 5 for x in pixels, z in pixels]
        for (eye, lookat, up, expected) in (
            ((0, 0, 30), (0, 0, 0), (0, 1, 0), ring),
            ((0, 30, 0), (0, 0, 0), (0, 0, 1), band),
            ((0, 0, 0), (0, 1, 0), (0, 0, 1), band),
        )
            view = TiledFieldView(field; maxdim = 6,
                                  overlays = [SurfaceSliceOverlay(SphereRegion(radius = 10))])
            view.params.refine_samples = 1
            view.camera.projection = :ortho
            view.camera.eye = eye
            view.camera.lookat = lookat
            view.camera.up = up
            view.camera.ortho_half = 12
            @test (depthbuf(view, w) .< 1) == expected
        end
        @test GL.glGetError() == GL.GL_NO_ERROR
    end

    @testset "sampling matches a single texture" begin
        field = KeyedArray([Float32(isodd(i) + j / 70 + k / 30)
                            for i in 1:9, j in 1:7, k in 1:3],
                           (range(-1, 1; length = 9), range(-1, 1; length = 7), -1:1))
        for interp in (0, 1), kind in (:volume, :plane, :surface)
            overlays() = kind === :volume ? Overlay[] :
                         kind === :plane ? Overlay[SliceOverlay(:z; pos = 0.35)] :
                         Overlay[SurfaceSliceOverlay(SphereRegion(radius = 0.7))]
            function makeview(source)
                view = FieldView(source; overlays = overlays())
                view.params.refine_samples = 1
                view.params.mode = 1
                view.params.interp = interp
                view.camera.projection = :ortho
                view.camera.eye = SVector(0.0, 0.0, 3.0)
                view.camera.lookat = SVector(0.0, 0.0, 0.0)
                view.camera.up = SVector(0.0, 1.0, 0.0)
                view.camera.ortho_half = 1.2
                view
            end
            _, plain = renderbuf(makeview(field), 96)
            _, tiled = renderbuf(makeview(TiledField(field; maxdim = 6)), 96)
            @test maximum(abs.(Int.(plain) .- Int.(tiled))) <= 1
            @test GL.glGetError() == GL.GL_NO_ERROR
        end
    end
end
