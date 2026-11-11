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

    @testset "Cartesian ray budgets" begin
        field = KeyedArray(zeros(Float32, 9, 3, 3), (0:0.25:2, 0:1:2, 0:1:2))
        tiled = TiledField(field; maxdim = 6)
        colored = KeyedArray(fill(RGBA{Float32}(1, 0, 0, 0.1), size(field)), axiskeys(field))
        _, radius = VolumeRendering.bounds(region(field))
        steps = ceil(Int, 2radius / 0.125) + 1
        for source in (field, tiled, colored)
            @test VolumeRendering._march_steps(source, radius, 0.5) == steps
            @test VolumeRendering._march_steps(source, radius, 0.25) > steps
            @test VolumeRendering._march_steps(source, 2radius, 0.5) > steps
            @test_throws "ray sample budget must fit in a positive 32-bit integer" VolumeRendering._march_steps(source, radius, 1e-12)
        end
        custom = GLSLField("float sampleField(vec3 p){ return 1.0; } float stepSize(vec3 p){ return 0.1; }";
                           region = region(field), value_range = (0, 1))
        @test VolumeRendering._march_steps(custom, radius, 0.5) == 2048
        for source in (field, tiled, colored, custom)
            view = FieldView(source)
            for scale in (0.0, -1.0, Inf, NaN, 1e-300, 1e300)
                view.params.step_scale = scale
                @test_throws "step_scale must be finite and positive in Float32" render!(view, 8, 8)
                @test view.vr === nothing
            end
        end
        for spacing in (0.0, 1e-300, 1e300)
            invalid = KeyedArray(zeros(Float32, 3, 3, 3),
                                 ntuple(_ -> range(0; step = spacing, length = 3), 3))
            @test_throws "Cartesian sample spacing must be finite and positive in Float32" VolumeRendering._march_steps(invalid, radius, 0.5)
        end
    end

    @testset "long rays reach the far side" begin
        for direction in (-1, 1), n in (1537, 4097)
            xs = range(-1, 1; length = n)
            field = KeyedArray([Float32(direction * x < -0.75) for x in xs, y in -1:1, z in -1:1],
                               (xs, -1:1, -1:1))
            source = n > 2048 ? TiledField(field; maxdim = 2048) : field
            view = FieldView(source; colormap = :grays)
            view.params.refine_samples = 1
            view.camera.projection = :ortho
            view.camera.eye = (3direction, 0, 0)
            view.camera.lookat = (0, 0, 0)
            view.camera.up = (0, 0, 1)
            view.camera.ortho_half = 0.5
            for mode in 0:2, step_scale in (0.5, 0.25)
                view.params.mode = mode
                view.params.step_scale = step_scale
                _, pixels = renderbuf(view, 8)
                @test all(>(0), pixels[1:3, :, :])
                @test GL.glGetError() == GL.GL_NO_ERROR
            end
        end
    end

    @testset "rays crossing tiles match a single texture" begin
        dims = (9, 7, 9)
        field = KeyedArray([Float32(0.1 + 0.3sin(i)^2 + 0.3cos(j)^2 + 0.3sin(k)^2)
                            for i in 1:dims[1], j in 1:dims[2], k in 1:dims[3]],
                           map(n -> range(-1, 1; length = n), dims))
        tiled = TiledField(field; maxdim = 6)
        @test tiled.k == (2, 2, 2)
        for eye in ((3, 0, 0), (-3, 0, 0), (0, 3, 0), (0, -3, 0),
                    (0, 0, 3), (0, 0, -3), (3, 3, 3), (-3, -3, -3)),
            mode in 0:2, interp in 0:1
            views = map((field, tiled)) do source
                view = FieldView(source)
                view.params.refine_samples = 1
                view.params.mode = mode
                view.params.interp = interp
                view.camera.projection = :ortho
                view.camera.eye = eye
                view.camera.lookat = (0, 0, 0)
                view.camera.up = abs(eye[3]) > abs(eye[2]) ? (0, 1, 0) : (0, 0, 1)
                view.camera.ortho_half = 0.8
                view
            end
            _, plain = renderbuf(views[1], 32)
            _, bricks = renderbuf(views[2], 32)
            @test maximum(abs.(Int.(plain) .- Int.(bricks))) <= 1
            @test GL.glGetError() == GL.GL_NO_ERROR
        end
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
