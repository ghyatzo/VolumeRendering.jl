using TestItemRunner
@run_package_tests


@testmodule GLContext begin
    import GLFW, ModernGL as GL
    import VolumeRendering as VR

    # one hidden 4.1 core context per process, current for every GL test item, never destroyed before exit
    const window = let
        GLFW.WindowHint(GLFW.VISIBLE, false)
        GLFW.WindowHint(GLFW.CONTEXT_VERSION_MAJOR, 4); GLFW.WindowHint(GLFW.CONTEXT_VERSION_MINOR, 1)
        GLFW.WindowHint(GLFW.OPENGL_PROFILE, GLFW.OPENGL_CORE_PROFILE)
        GLFW.WindowHint(GLFW.OPENGL_FORWARD_COMPAT, true)
        w = GLFW.CreateWindow(64, 64, "VolumeRendering tests")
        GLFW.MakeContextCurrent(w)
        w
    end

    const FULLSCREEN_VERT = """
        #version 410 core
        void main() { gl_Position = vec4(vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0), 0.0, 1.0); }
        """
    const empty_vao = VR.gen(GL.glGenVertexArrays)

    # looking down -z from z = `eye` with screen-up +y: world +x → image right, +y → image up
    topcam(half = 1.0; projection = :ortho, eye = 3.0) =
        VR.Camera(VR.SVector(0.0, 0.0, eye), VR.SVector(0.0, 0.0, 0.0), VR.SVector(0.0, 1.0, 0.0), projection, 90.0, half)
    # world (x, y) of the centre of image pixel (row, col) under `topcam(half)` with a square w×w image
    pixel_xy(half, w, row, col) = (half * ((col - 0.5) / w * 2 - 1), half * (1 - (row - 0.5) / w * 2))

    # draw `count` vertices (× `instances`) with `prog` into `fb`; `setup()` runs with the program current
    function draw(setup, prog, fb; count = 3, instances = nothing, vao = empty_vao, mode = GL.GL_TRIANGLES)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fb.fbo)
        GL.glViewport(0, 0, fb.color.size...)
        GL.glDisable(GL.GL_DEPTH_TEST); GL.glDisable(GL.GL_BLEND)
        GL.glClearColor(0, 0, 0, 0); GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glUseProgram(prog.id)
        GL.glBindVertexArray(vao)
        setup()
        isnothing(instances) ? GL.glDrawArrays(mode, 0, count) : GL.glDrawArraysInstanced(mode, 0, count, instances)
        GL.glBindVertexArray(0); GL.glUseProgram(0)
        VR.glcheck()
    end
end


@testitem "Identity" begin
    import VolumeRendering as VR
    A = [1.0, 2.0]
    @test VR.Identity(A) == VR.Identity(A)
    @test isequal(VR.Identity(A), VR.Identity(A))
    @test hash(VR.Identity(A)) == hash(VR.Identity(A))
    @test VR.Identity(A) != VR.Identity(copy(A))
    @test length(Dict(VR.Identity(A) => 1, VR.Identity(A) => 2, VR.Identity(copy(A)) => 3)) == 2
end

@testitem "program link" setup = [GLContext] begin
    import ModernGL as GL, VolumeRendering as VR
    frag = """
        #version 410 core
        uniform sampler2D a; uniform sampler1D b; uniform float k; uniform float unused;
        out vec4 frag;
        void main() { frag = vec4(texture(a, vec2(0.5)).r + texture(b, 0.5).r + k); }
        """
    GL.glUseProgram(0)
    p = VR.Program(GLContext.FULLSCREEN_VERT, frag, [:k, :unused], [:b, :a])
    VR.glcheck()
    current = Ref{GL.GLint}(-1); GL.glGetIntegerv(GL.GL_CURRENT_PROGRAM, current)
    @test current[] == 0                       # linking leaves the current program alone
    @test p.locations[:k] >= 0
    @test p.locations[:unused] == -1           # dropped by the linker
    unit(n) = (r = Ref{GL.GLint}(-1); GL.glGetUniformiv(p.id, p.locations[n], r); r[])
    @test (unit(:b), unit(:a)) == (0, 1)       # declaration order of `samplers`
    @test_throws "shader compilation failed" VR.Program(GLContext.FULLSCREEN_VERT, "#version 410 core\nvoid main() { x; }", Symbol[], Symbol[])
    @test_throws "program link failed" VR.Program(GLContext.FULLSCREEN_VERT, "#version 410 core\nvoid f();\nvoid main() { f(); }", Symbol[], Symbol[])
    VR.glcheck()
    VR.gl_delete(p)
end

@testitem "texture upload → readback" setup = [GLContext] begin
    import ModernGL as GL, VolumeRendering as VR
    using ColorTypes, FixedPointNumbers
    # hostile host unpack/pack state: uploads and readback must set their own
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 8); GL.glPixelStorei(GL.GL_UNPACK_ROW_LENGTH, 2)
    GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 8); GL.glPixelStorei(GL.GL_PACK_SKIP_PIXELS, 1)
    internal(t) = (r = Ref{GL.GLint}(0); GL.glBindTexture(VR.gl_target(t), t.id);
                   GL.glGetTexLevelParameteriv(VR.gl_target(t), 0, GL.GL_TEXTURE_INTERNAL_FORMAT, r); r[])
    sizes = [(5,), (3, 5), (3, 5, 7)]
    cases = [   # input eltype, internal format, texel type
        (Float32,       GL.GL_R32F,    Float32),
        (Float64,       GL.GL_R32F,    Float32),
        (Int,           GL.GL_R32F,    Float32),
        (RGBA{N0f8},    GL.GL_RGBA8,   RGBA{N0f8}),
        (RGB{N0f8},     GL.GL_RGBA8,   RGBA{N0f8}),
        (RGBA{Float32}, GL.GL_RGBA32F, RGBA{Float32}),
        (RGB{Float64},  GL.GL_RGBA32F, RGBA{Float32}),
    ]
    rnd(::Type{T}, sz) where {T<:Real} = T <: Integer ? rand(-100:100, sz) : T.(randn(sz))
    rnd(::Type{C}, sz) where {C<:Colorant} = map(_ -> C(ntuple(_ -> rand(eltype(C)), length(C))...), CartesianIndices(sz))
    for sz in sizes, (T, fmt, TT) in cases
        A = rnd(T, sz)
        t = VR.upload(A)
        VR.glcheck()
        @test t isa VR.GLTexture{TT,length(sz)}
        @test t.size == sz
        @test internal(t) == fmt
        B = VR.read_texture(t)
        VR.glcheck()
        @test B isa Array{TT,length(sz)}
        @test B == TT.(A)
        VR.gl_delete(t)
    end
    # a texture already in its texel type is uploaded without a conversion copy
    A = rand(Float32, 4, 4)
    @test VR.texel_array(A) === A
    # non-Array inputs are materialized in their own index order
    C = rand(RGBA{N0f8}, 4, 5, 6)
    for X in (view(A, 2:3, :), PermutedDimsArray(A, (2, 1)), view(C, :, 2:4, 1:2:5), PermutedDimsArray(C, (3, 1, 2)))
        t = VR.upload(X)
        @test VR.read_texture(t) == X
        VR.gl_delete(t)
    end
    VR.glcheck()
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4); GL.glPixelStorei(GL.GL_UNPACK_ROW_LENGTH, 0)
    GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 4); GL.glPixelStorei(GL.GL_PACK_SKIP_PIXELS, 0)
end

@testitem "sampler filter and wrap" setup = [GLContext] begin
    import ModernGL as GL, VolumeRendering as VR
    using ColorTypes
    # each of the 4 output pixels samples the texture at coords[i]; R channel = sampled value
    sampled(frag_body, tex, sampler, coords) = begin
        frag = """
            #version 410 core
            $frag_body
            out vec4 frag;
            void main() { frag = vec4(sample_at(int(gl_FragCoord.x)), 0.0, 0.0, 1.0); }
            """
        p = VR.Program(GLContext.FULLSCREEN_VERT, frag, [:coords], [:tex])
        fb = VR.float_framebuffer(4, 1)
        GLContext.draw(p, fb) do
            VR.bind_texture(0, tex, sampler)
            GL.glUniform3fv(p.locations[:coords], 4, Float32.(reduce(vcat, coords)))
        end
        out = map(c -> c.r, vec(VR.read_texture(fb.color)))
        VR.gl_delete(p); VR.gl_delete(fb)
        out
    end

    # 1-D [0, 1]: texel centres at u = 0.25 and 0.75
    tex1 = VR.upload([0.0, 1.0])
    body1 = "uniform sampler1D tex; uniform vec3 coords[4]; float sample_at(int i) { return texture(tex, coords[i].x).r; }"
    coords1 = [[0.45, 0, 0], [1.0, 0, 0], [1.6, 0, 0], [-0.3, 0, 0]]
    for (wrap, filter, expected) in [
            (:clamp,  :linear,  [0.4, 1.0, 1.0, 0.0]),
            (:repeat, :linear,  [0.4, 0.5, 0.7, 0.9]),
            (:mirror, :linear,  [0.4, 1.0, 0.3, 0.1]),
            (:clamp,  :nearest, [0.0, 1.0, 1.0, 0.0]),
        ]
        s = VR.GLSampler((wrap,), filter)
        @test sampled(body1, tex1, s, coords1) ≈ expected atol = 0.01   # same texture, only the sampler differs
        VR.gl_delete(s)
    end

    # 3-D 2×2×2, value = (i-1) + 2(j-1) + 4(k-1): wrap is per axis S, T, R = array dims 1, 2, 3
    tex3 = VR.upload([(i - 1) + 2(j - 1) + 4(k - 1) for i in 1:2, j in 1:2, k in 1:2])
    body3 = "uniform sampler3D tex; uniform vec3 coords[4]; float sample_at(int i) { return texture(tex, coords[i]).r; }"
    s3 = VR.GLSampler((:clamp, :repeat, :mirror), :nearest)
    @test sampled(body3, tex3, s3, [[1.1, 0.25, 0.25], [0.25, 1.1, 0.25], [0.25, 0.25, 1.6], [0.25, 0.25, 1.1]]) ≈ [1, 0, 0, 4]
    VR.glcheck()
    foreach(VR.gl_delete, (tex1, tex3, s3))
end

@testitem "framebuffers" setup = [GLContext] begin
    import ModernGL as GL, VolumeRendering as VR
    using ColorTypes, FixedPointNumbers
    internal(t) = (r = Ref{GL.GLint}(0); GL.glBindTexture(GL.GL_TEXTURE_2D, t.id);
                   GL.glGetTexLevelParameteriv(GL.GL_TEXTURE_2D, 0, GL.GL_TEXTURE_INTERNAL_FORMAT, r); r[])
    scene = VR.scene_framebuffer(7, 5)
    flt = VR.float_framebuffer(7, 5)
    VR.glcheck()
    @test (internal(scene.color), internal(scene.depth), internal(flt.color)) == (GL.GL_RGBA8, GL.GL_DEPTH_COMPONENT32F, GL.GL_RGBA16F)
    @test isnothing(flt.depth)
    @test scene.color.size == scene.depth.size == flt.color.size == (7, 5)

    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, scene.fbo)
    GL.glColorMask(true, true, true, true); GL.glDepthMask(true)
    GL.glClearColor(1, 0.5, 0, 1); GL.glClearDepth(0.25); GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    @test all(==(RGBA{N0f8}(1, 0.5, 0, 1)), VR.read_texture(scene.color))
    @test all(==(VR.Depth32(0.25)), VR.read_texture(scene.depth))
    # the colour texture is complete with no sampler object bound (as when the host displays it)
    p = VR.Program(GLContext.FULLSCREEN_VERT,
                   "#version 410 core\nuniform sampler2D tex; out vec4 frag; void main() { frag = texture(tex, vec2(0.5)); }",
                   Symbol[], [:tex])
    probe = VR.float_framebuffer(1, 1)
    GLContext.draw(p, probe) do
        GL.glActiveTexture(GL.GL_TEXTURE0); GL.glBindTexture(GL.GL_TEXTURE_2D, scene.color.id); GL.glBindSampler(0, 0)
    end
    c = only(VR.read_texture(probe.color))
    @test [c.r, c.g, c.b, c.alpha] ≈ [1, 0.5, 0, 1] atol = 0.01
    foreach(VR.gl_delete, (p, probe))
    # RGBA16F keeps a faint value that 8 bits would round to 0
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, flt.fbo)
    GL.glClearColor(0.001, 0.25, 0.5, 1); GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    @test VR.read_texture(flt.color)[3, 2] == RGBA{Float16}(0.001, 0.25, 0.5, 1)
    VR.glcheck()
    foreach(VR.gl_delete, (scene, flt))
end

@testitem "instance buffer" setup = [GLContext] begin
    import ModernGL as GL, VolumeRendering as VR
    using ColorTypes, FixedPointNumbers, StaticArrays
    # each instance: a square of half-size `half` px around pixel-space `center`, coloured `color`
    vert = """
        #version 410 core
        layout(location = 0) in vec3 center; layout(location = 1) in float half_px; layout(location = 2) in vec3 color;
        uniform vec2 viewport;
        flat out vec3 vcolor;
        void main() {
            vec2 corner = vec2((gl_VertexID & 1) == 0 ? -1.0 : 1.0, (gl_VertexID >> 1) == 0 ? -1.0 : 1.0);
            gl_Position = vec4((center.xy + corner * half_px) / viewport * 2.0 - 1.0, 0.0, 1.0);
            vcolor = color;
        }
        """
    frag = "#version 410 core\nflat in vec3 vcolor; out vec4 frag; void main() { frag = vec4(vcolor, 1.0); }"
    records = [
        (center = SVector{3,Float32}(1.5, 1.5, 0), half = 0.5f0, color = RGB{Float32}(1, 0, 0)),
        (center = SVector{3,Float32}(5, 6, 0),     half = 1.0f0, color = RGB{Float32}(0, 1, 0)),
    ]
    buf = VR.InstanceBuffer(records)
    @test buf.n == 2
    p = VR.Program(vert, frag, [:viewport], Symbol[])
    fb = VR.scene_framebuffer(8, 8)
    GLContext.draw(p, fb; count = 4, instances = buf.n, vao = buf.vao, mode = GL.GL_TRIANGLE_STRIP) do
        GL.glUniform2f(p.locations[:viewport], 8, 8)
    end
    img = VR.read_texture(fb.color)
    red, green, none = RGBA{N0f8}(1, 0, 0, 1), RGBA{N0f8}(0, 1, 0, 1), RGBA{N0f8}(0, 0, 0, 0)
    expected = [(i, j) == (2, 2) ? red : (i in 5:6 && j in 6:7) ? green : none for i in 1:8, j in 1:8]
    @test img == expected
    VR.gl_delete(buf)
    @test GL.glIsVertexArray(buf.vao) == GL.GL_FALSE && GL.glIsBuffer(buf.vbo) == GL.GL_FALSE

    # zero instances: valid, draws nothing
    nobuf = VR.InstanceBuffer(empty(records))
    @test nobuf.n == 0
    GLContext.draw(p, fb; count = 4, instances = nobuf.n, vao = nobuf.vao, mode = GL.GL_TRIANGLE_STRIP) do
        GL.glUniform2f(p.locations[:viewport], 8, 8)
    end
    @test all(==(none), VR.read_texture(fb.color))
    VR.glcheck()
    foreach(VR.gl_delete, (p, fb, nobuf))
end

@testitem "resource cache" setup = [GLContext] begin
    import ModernGL as GL, VolumeRendering as VR
    cache = VR.ResourceCache((8, 6), (4, 3))
    @test (cache.scene_fb.color.size, cache.vol_fb.color.size, cache.accum_fb.color.size) == ((8, 6), (4, 3), (8, 6))

    # textures by array identity
    A, B = rand(Float32, 4, 4), rand(Float32, 4, 4)
    tex(X) = get!(() -> VR.upload(X), cache.entries, VR.Identity(X))
    tA = tex(A)
    @test tex(A) === tA
    @test tex(copy(A)).id != tA.id
    tB = tex(B)

    # programs by source text
    frag(c) = "#version 410 core\nout vec4 frag; void main() { frag = vec4($c); }"
    prog(f) = get!(() -> VR.Program(GLContext.FULLSCREEN_VERT, f, Symbol[], Symbol[]), cache.entries, (GLContext.FULLSCREEN_VERT, f))
    p1 = prog(frag("1.0"))
    @test prog(frag("1.0")) === p1
    @test prog(String(codeunits(frag("1.0")))) === p1     # equal text, different String object
    p2 = prog(frag("0.5"))
    @test p2.id != p1.id
    s = get!(() -> VR.GLSampler((:clamp, :clamp), :linear), cache.entries, ((:clamp, :clamp), :linear))
    @test length(cache.entries) == 6
    VR.glcheck()

    # sweep frees exactly the entries the plan does not list
    VR.sweep!(cache, Set{Any}([VR.Identity(A), (GLContext.FULLSCREEN_VERT, frag("1.0"))]))
    @test issetequal(keys(cache.entries), [VR.Identity(A), (GLContext.FULLSCREEN_VERT, frag("1.0"))])
    @test GL.glIsTexture(tA.id) == GL.GL_TRUE && GL.glIsProgram(p1.id) == GL.GL_TRUE
    @test GL.glIsTexture(tB.id) == GL.GL_FALSE
    @test GL.glIsProgram(p2.id) == GL.GL_FALSE
    @test GL.glIsSampler(s.id) == GL.GL_FALSE
    tex(B)                                                  # re-created after being swept
    @test haskey(cache.entries, VR.Identity(B)) && GL.glIsTexture(cache.entries[VR.Identity(B)].id) == GL.GL_TRUE
    VR.glcheck()

    # framebuffers are replaced only on a size change
    scene, vol = cache.scene_fb, cache.vol_fb
    VR.fit_framebuffers!(cache, (8, 6), (2, 2))
    @test cache.scene_fb === scene && cache.vol_fb !== vol && cache.vol_fb.color.size == (2, 2)
    @test GL.glIsFramebuffer(vol.fbo) == GL.GL_FALSE || vol.fbo == cache.vol_fb.fbo   # freed first, so its name may be reused

    GL.glBindVertexArray(cache.empty_vao); GL.glBindVertexArray(0)   # a VAO name becomes an object on first bind
    @test GL.glIsVertexArray(cache.empty_vao) == GL.GL_TRUE
    VR.free!(cache)
    @test isempty(cache.entries)
    @test GL.glIsVertexArray(cache.empty_vao) == GL.GL_FALSE
    @test GL.glIsTexture(tA.id) == GL.GL_FALSE && GL.glIsProgram(p1.id) == GL.GL_FALSE
    @test GL.glIsFramebuffer(cache.scene_fb.fbo) == GL.GL_FALSE && GL.glIsTexture(cache.scene_fb.depth.id) == GL.GL_FALSE
    VR.glcheck()
end

@testitem "Camera" begin
    import VolumeRendering as VR
    using StaticArrays, LinearAlgebra

    cam = Camera(; center = SVector(1.0, 2.0, 3.0), radius = 2)
    @test cam.eye ≈ [5.670993664969137, 3.868397465987655, 6.269695565478396]
    @test (cam.lookat, cam.up, cam.projection, cam.fov, cam.ortho_half) == ([1, 2, 3], [0, 0, 1], :perspective, 45.0, 2.2)
    @test VR.distance(cam) ≈ 6
    @test Camera(; radius = 1).lookat == [0, 0, 0]
    right, up, fwd = VR.view_basis(cam)
    @test [right up fwd]' * [right up fwd] ≈ I
    @test fwd ≈ normalize(cam.lookat - cam.eye)

    # looking down -z from (0, 0, 3), screen-up = +y
    top(projection) = Camera(SVector(0.0, 0.0, 3.0), SVector(0.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0), projection, 90.0, 2.0)
    clip(cam, p) = VR.view_proj(cam, 2.0) * SVector(p..., 1.0)
    @test VR.fovscale(top(:perspective)) ≈ 1 && VR.fovscale(top(:ortho)) == 2
    @test clip(top(:perspective), (1, 1, 0)) ≈ [0.5, 1, 0, 3]        # w = depth along forward
    @test clip(top(:perspective), (0, 0, 5)) ≈ [0, 0, 0, -2]         # behind the eye: w < 0
    @test clip(top(:ortho), (1, 1, 0)) ≈ [0.25, 0.5, 0, 1]
    @test clip(top(:ortho), (1, 1, 7)) ≈ [0.25, 0.5, 0, 1]
    @test VR.view_proj(top(:perspective), 1.5)[3, :] == zeros(4)     # no near/far

    # zoom clamps to 1e-4..1e4 × the reference's distance / ortho_half
    ref = top(:perspective)
    c = top(:perspective); zoom!(c, 0.5, ref)
    @test c.eye ≈ [0, 0, 1.5] && c.lookat == [0, 0, 0]
    zoom!(c, 1e-9, ref); @test VR.distance(c) ≈ 3e-4
    zoom!(c, 1e20, ref); @test VR.distance(c) ≈ 3e4
    @test c.eye ≈ [0, 0, 3e4]
    o = top(:ortho); zoom!(o, 1e-9, top(:ortho)); @test o.ortho_half ≈ 2e-4 && o.eye == [0, 0, 3]
    zoom!(o, 1e20, top(:ortho)); @test o.ortho_half ≈ 2e4

    # reset! copies every field, the projection included
    c = top(:ortho); c.fov = 30; c.ortho_half = 9; orbit!(c, 1, 1)
    @test VR.reset!(c, top(:perspective)) === c
    @test all(n -> getfield(c, n) == getfield(top(:perspective), n), fieldnames(Camera))

    side() = Camera(SVector(3.0, 0.0, 0.0), SVector(0.0, 0.0, 0.0), SVector(0.0, 0.0, 1.0), :perspective, 90.0, 1.0)
    @test orbit!(side(), π/2, 0).eye ≈ [0, 3, 0]                       # yaw about world up
    @test orbit!(side(), 0, 0.5).eye ≈ 3 .* [cos(0.5), 0, sin(0.5)]   # pitch raises the eye
    @test orbit!(side(), 0, 10).eye ≈ 3 .* [sin(0.02), 0, cos(0.02)]  # clamped short of the pole
    c = flylook!(side(), π/2, 0)
    @test c.eye == [3, 0, 0] && c.lookat ≈ [3, -3, 0]

    c = top(:perspective); pan!(c, 0.5, 0.25)    # half-size of the view at the lookat = distance·tan(fov/2) = 3
    @test c.eye ≈ [-1.5, 0.75, 3] && c.lookat ≈ [-1.5, 0.75, 0]
    c = top(:ortho); pan!(c, 0.5, 0)
    @test c.eye ≈ [-1, 0, 3]
    c = top(:perspective); VR.fly!(c, 1, 0.5, 0, 2)
    @test c.eye ≈ [1, 0, 1] && c.lookat ≈ [1, 0, -2]
end

@testitem "Texture" begin
    import VolumeRendering as VR
    using AxisKeys
    A = rand(Float32, 2, 3, 4)
    t = Texture(A)
    @test (t.data === A, t.wrap, t.filter) == (true, (:clamp, :clamp, :clamp), :linear)
    @test Texture(A; wrap = :repeat, filter = :nearest).wrap == (:repeat, :repeat, :repeat)
    @test Texture(A; wrap = (:repeat, :clamp, :mirror)).wrap == (:repeat, :clamp, :mirror)
    @test Texture(rand(3)).wrap == (:clamp,)
    @test_throws "wrap" Texture(A; wrap = :wrap)
    @test_throws "wrap" Texture(A; wrap = (:clamp, :clamp, :bad))
    @test_throws MethodError Texture(A; wrap = (:clamp, :clamp))
    @test_throws "filter" Texture(A; filter = :cubic)
    @test_throws "3-D" Texture(rand(2, 2, 2, 2))
    @test_throws "filter" Texture(A, (:clamp, :clamp, :clamp), :bad)   # every constructor validates
    K = KeyedArray(A; x = 1:2, y = 1:3, z = 1:4)
    @test VR.cache_key(K) === A
    @test VR.cache_key(A) === A
end

@testitem "shapes" begin
    import VolumeRendering as VR
    using StaticArrays, IntervalSets

    s = Sphere(; center = SVector(1.0, 0.0, -1.0), radius = 2)
    @test Sphere(; radius = 1).center == [0, 0, 0]
    @test bounds(s) == (; center = [1, 0, -1], radius = 2)
    @test aabb(s) == (; lo = [-1, -2, -3], hi = [3, 2, 1])
    @test uniforms(s) == (; sphere_center = [1, 0, -1], sphere_radius = 2)

    b = Box(0..2, -1..1, 1..5)
    @test b.x isa ClosedInterval{Float64}
    @test aabb(b) == (; lo = [0, -1, 1], hi = [2, 1, 5])
    @test bounds(b).center == [1, 0, 3] && bounds(b).radius ≈ sqrt(1 + 1 + 4)
    @test uniforms(b) == (; box_lo = [0, -1, 1], box_hi = [2, 1, 5])

    @test Plane(:x, 0.5, b) == Plane(SVector(0.5, -1, 1), SVector(0, 2, 0), SVector(0, 0, 4))
    @test Plane(:y, 0, b) == Plane(SVector(0, 0, 1), SVector(2, 0, 0), SVector(0, 0, 4))
    @test Plane(:z, 3, b) == Plane(SVector(0, -1, 3), SVector(2, 0, 0), SVector(0, 2, 0))
    @test_throws ArgumentError Plane(:w, 0, b)
    p = Plane(:z, 3, b)
    @test aabb(p) == (; lo = [0, -1, 3], hi = [2, 1, 3])
    @test bounds(p).center == [1, 0, 3] && bounds(p).radius ≈ sqrt(2)
    @test uniforms(p) == (; plane_origin = [0, -1, 3], plane_u = [2, 0, 0], plane_v = [0, 2, 0])
    # a parallelogram is enclosed through its longer diagonal
    q = Plane(SVector(0.0, 0, 0), SVector(1.0, 0, 0), SVector(3.0, 1, 0))
    @test bounds(q).radius ≈ sqrt(17) / 2
    @test aabb(q) == (; lo = [0, 0, 0], hi = [4, 1, 0])

    @test all(x -> occursin("vec2 vr_hit(vec3 ro, vec3 rd)", shape_glsl(x)), (s, b, p))
    @test all(VR.isshape, (s, b, p))
    @test !VR.isshape(1.0)
end

@testitem "GridField" begin
    import VolumeRendering as VR
    using AxisKeys, IntervalSets, ColorTypes, Unitful

    # Cartesian, in a non-xyz dim order; :color
    C = KeyedArray(rand(RGBA{Float32}, 4, 2, 3); z = 0:3, x = 0.0:1.0:1.0, y = 1.0:-0.25:0.5)
    g = GridField(C)
    @test (value_kind(g), g.interpolation) == (:color, :linear)
    @test field_glsl(g) == """
        vec4 value(vec3 p) {
            vec3 idx  = vec3(((p.z) - 0.0) * 1.0, ((p.x) - 0.0) * 1.0, ((p.y) - 1.0) * -4.0);
            vec3 dims = vec3(4.0, 2.0, 3.0);
            return texture(grid, (idx + 0.5) / dims);
        }
        float stepSize(vec3 p) { return 0.25; }
        """
    @test Box(g) == Box(0..1, 0.5..1, 0..3)
    @test value_range(g) == 0..1
    @test textures(g) == (; grid = Texture(C; wrap = :clamp, filter = :linear))
    @test uniforms(g) == (;)

    # scalar, NaN-skipping value range, nearest
    S = KeyedArray([NaN 2.0; -1.5 NaN;;; 0.5 3.0; 1.0 1.0]; x = 0:1, y = 0:1, z = 0:1)
    s = GridField(S; interpolation = :nearest)
    @test value_kind(s) === :scalar
    @test value_range(s) === -1.5..3.0
    @test textures(s).grid.filter === :nearest
    @test endswith(split(field_glsl(s), "\n")[4], "(idx + 0.5) / dims).r;")

    # (lon, lat) in both dim orders
    lat, lon = -1.0:0.5:1.0, 0.0:0.5:1.5
    expected_latlon = """
        float value(vec3 p) {
            float vr_lat = asin(clamp(p.z / length(p), -1.0, 1.0));
            float vr_lon = 0.0 + mod(atan(p.y, p.x) - 0.0, 6.283185307179586);
            vec2 idx  = vec2(((vr_lat) - -1.0) * 2.0, ((vr_lon) - 0.0) * 2.0);
            vec2 dims = vec2(5.0, 4.0);
            return texture(grid, (idx + 0.5) / dims).r;
        }
        float stepSize(vec3 p) { return 0.5 * length(p); }
        """
    ll = GridField(KeyedArray(zeros(5, 4); lat, lon))
    @test field_glsl(ll) == expected_latlon
    @test textures(ll).grid.wrap == (:clamp, :repeat)
    lo = GridField(KeyedArray(zeros(4, 5); lon, lat))
    @test field_glsl(lo) == replace(expected_latlon,
        "vec2(((vr_lat) - -1.0) * 2.0, ((vr_lon) - 0.0) * 2.0)" => "vec2(((vr_lon) - 0.0) * 2.0, ((vr_lat) - -1.0) * 2.0)",
        "vec2(5.0, 4.0)" => "vec2(4.0, 5.0)")
    @test textures(lo).grid.wrap == (:repeat, :clamp)
    @test_throws ArgumentError Box(ll)

    # descending lat (IONEX), and descending lon
    d = field_glsl(GridField(KeyedArray(zeros(9, 7); lat = 1.0:-0.25:-1.0, lon = -3.0:1.0:3.0)))
    @test occursin("float vr_lon = -3.0 + mod(atan(p.y, p.x) - -3.0, 6.283185307179586);", d)
    @test occursin("vec2 idx  = vec2(((vr_lat) - 1.0) * -4.0, ((vr_lon) - -3.0) * 1.0);", d)
    @test occursin("return 0.25 * length(p);", d)
    d = field_glsl(GridField(KeyedArray(zeros(9, 7); lat = 1.0:-0.25:-1.0, lon = 3.0:-1.0:-3.0)))
    @test occursin("float vr_lon = 3.0 - mod(3.0 - atan(p.y, p.x), 6.283185307179586);", d)
    @test occursin("((vr_lon) - 3.0) * -1.0", d)

    # u"°" keys become radians through convert; dimensionful keys throw Julia's conversion error
    nums(s) = parse.(Float64, getfield.(eachmatch(r"-?\d+\.\d+(?:e-?\d+)?", s), :match))
    @test nums(axis_index_glsl((90:-2.5:-90)u"°", "vr_lat")) ≈ [1.5707963267948966, -22.918311805232932]
    deg = GridField(KeyedArray(zeros(73, 72); lat = (90:-2.5:-90)u"°", lon = (-180:5:175)u"°"))
    @test occursin("float vr_lon = -3.141592653589793 + mod(atan(p.y, p.x) - -3.141592653589793, 6.283185307179586);", field_glsl(deg))
    @test nums(match(r"return (.*) \* length", field_glsl(deg))[1]) ≈ [0.04363323129985824]   # 2.5° in radians
    km = GridField(KeyedArray(zeros(2, 2, 2); x = (0:1)u"km", y = 0:1, z = 0:1))
    @test_throws Unitful.DimensionError field_glsl(km)
    @test_throws Unitful.DimensionError Box(km)

    # validation
    @test_throws "dim names" GridField(KeyedArray(zeros(2, 2); x = 0:1, y = 0:1))
    @test_throws "dim names" GridField(KeyedArray(zeros(2, 2, 2); x = 0:1, y = 0:1, r = 0:1))
    @test_throws "interpolation" GridField(S; interpolation = :cubic)
    @test_throws MethodError GridField(zeros(2, 2, 2))   # raw arrays are never wrapped
    s.data = KeyedArray(zeros(2, 2); lat = 0:1, lon = 0:1)      # assignment is plain; any KeyedArray is accepted
    @test value_kind(s) === :scalar
end

@testitem "GLSLField" begin
    using IntervalSets
    f = GLSLField("float value(vec3 p) { return 1.0; } float stepSize(vec3 p) { return 0.1; }")
    @test (value_kind(f), textures(f), uniforms(f), value_range(f)) == (:scalar, (;), (;), 0..1)
    T = Texture(rand(8))
    f = GLSLField("…"; kind = :color, textures = (; lut = T), uniforms = (; k = 2.0), value_range = -1..1)
    @test (field_glsl(f), value_kind(f), textures(f), uniforms(f), value_range(f)) == ("…", :color, (; lut = T), (; k = 2.0), -1.0..1.0)
    @test_throws "kind" GLSLField(""; kind = :vector)
end

@testitem "TiledField" setup = [GLContext] begin
    import ModernGL as GL, VolumeRendering as VR
    using AxisKeys, ColorSchemes, ColorTypes, FixedPointNumbers, IntervalSets, StaticArrays

    # brick geometry and shared boundary texels
    for dims in ((8, 2, 2), (7, 2, 2), (9, 2, 2), (3, 8, 2), (3, 2, 8), (9, 7, 3))
        data = reshape(Float32.(1:prod(dims)), dims)
        field = KeyedArray(data; x = 1:dims[1], y = 1:dims[2], z = 1:dims[3])
        tiled = TiledField(field; maxdim = 4)
        @test all(<=(4), tiled.D)
        for I in CartesianIndices(data)
            mapped = ntuple(d -> VR._map_axis(tiled, d, I[d] - 1), 3)
            brick = ntuple(d -> mapped[d][1], 3)
            local_index = ntuple(d -> Int(mapped[d][2]) + 1, 3)
            @test VR.brick_data(tiled, brick)[local_index...] == data[I]
        end
        for d in 1:3
            tiled.k[d] > 1 || continue
            a = VR.brick_data(tiled, (0, 0, 0))
            b = VR.brick_data(tiled, ntuple(i -> Int(i == d), 3))
            @test selectdim(a, d, tiled.D[d]) == selectdim(b, d, 1)
        end
    end

    # pre-materialized bricks: textures(t) returns identical array references across calls and interpolation toggles
    dims = (64, 48, 8)
    data = reshape(Float32.(1:prod(dims)), dims)
    tiled = TiledField(KeyedArray(data; x = 1:dims[1], y = 1:dims[2], z = 1:dims[3]); maxdim = 32)
    brick = VR.brick_data(tiled, (0, 0, 0))
    @test brick == data[1:tiled.D[1], 1:tiled.D[2], 1:tiled.D[3]]
    @test (@allocated VR.brick_data(tiled, (0, 0, 0))) <= 2sizeof(brick) + 16_384
    tex1 = textures(tiled)
    tiled.interpolation = :nearest
    tex2 = textures(tiled)
    @test map(t -> t.data, tex1) === map(t -> t.data, tex2)
    @test all(t -> t.filter === :nearest, values(tex2))
    tiled.interpolation = :linear
    @test (value_kind(tiled), value_range(tiled), Box(tiled)) == (:scalar, 1.0..Float64(prod(dims)), Box(1..64, 1..48, 1..8))
    @test applicable(VR.controls!, tiled)

    # validation: maxdim, empty axes, > 13 bricks, non-Cartesian, interpolation
    small = KeyedArray(ones(Float32, 2, 2, 2); x = 1:2, y = 1:2, z = 1:2)
    @test_throws "maxdim must be at least 2" TiledField(small; maxdim = 1)
    @test_throws "axes must not be empty" TiledField(KeyedArray(zeros(Float32, 0, 2, 2); x = 1:0, y = 1:2, z = 1:2))
    @test_throws "maximum of 13" TiledField(KeyedArray(ones(Float32, 5, 5, 5); x = 1:5, y = 1:5, z = 1:5); maxdim = 2)
    @test_throws "Cartesian" TiledField(KeyedArray(ones(Float32, 4, 4); lat = 1:4, lon = 1:4))
    @test_throws "interpolation" TiledField(small; interpolation = :cubic)

    # rays crossing tiles match a single texture within 1 LSB across modes, orientations, and filters
    dims = (9, 7, 9)
    A = KeyedArray([Float32(0.1 + 0.3sin(i)^2 + 0.3cos(j)^2 + 0.3sin(k)^2)
                    for i in 1:dims[1], j in 1:dims[2], k in 1:dims[3]];
                   x = range(-1, 1; length = dims[1]), y = range(-1, 1; length = dims[2]), z = range(-1, 1; length = dims[3]))
    gf = GridField(A)
    tf = TiledField(A; maxdim = 6)
    @test tf.k == (2, 2, 2)
    cmap = Colormap(ColorSchemes.viridis, value_range(gf))
    box = Box(gf)
    for eye in ((3.0, 0.0, 0.0), (-3.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, -3.0, 0.0),
                (0.0, 0.0, 3.0), (0.0, 0.0, -3.0), (3.0, 3.0, 3.0), (-3.0, -3.0, -3.0)),
        mode in (:dvr, :mip, :average), interp in (:nearest, :linear)
        gf.interpolation = tf.interpolation = interp
        up = abs(eye[3]) > abs(eye[2]) ? SVector(0.0, 1.0, 0.0) : SVector(0.0, 0.0, 1.0)
        cam = Camera(SVector(eye), SVector(0.0, 0.0, 0.0), up, :ortho, 45.0, 0.81)
        plain = render_image(View([Volume(Mapped(gf, cmap); region = box, mode)]; camera = cam, refine_frames = 1), 32, 32)
        bricks = render_image(View([Volume(Mapped(tf, cmap); region = box, mode)]; camera = cam, refine_frames = 1), 32, 32)
        @test maximum(map((a, b) -> max(abs(Int(reinterpret(a.r)) - Int(reinterpret(b.r))),
                                        abs(Int(reinterpret(a.g)) - Int(reinterpret(b.g))),
                                        abs(Int(reinterpret(a.b)) - Int(reinterpret(b.b)))), plain, bricks)) <= 1
    end

    # sampling matches a single texture on Volume, Plane Surface, and Sphere Surface
    B = KeyedArray([Float32(isodd(i) + j / 70 + k / 30) for i in 1:9, j in 1:7, k in 1:3];
                   x = range(-1, 1; length = 9), y = range(-1, 1; length = 7), z = -1:1)
    gB, tB = GridField(B), TiledField(B; maxdim = 6)
    cmapB = Colormap(ColorSchemes.viridis, value_range(gB))
    top = GLContext.topcam(1.2)
    for interp in (:nearest, :linear),
        make_mark in (f -> Volume(Mapped(f, cmapB); region = Box(f), mode = :mip),
                      f -> Surface(Plane(:z, 0.35, Box(f)); color = Mapped(f, cmapB)),
                      f -> Surface(Clipped(Sphere(radius = 0.7), Box(f)); color = Mapped(f, cmapB)))
        gB.interpolation = tB.interpolation = interp
        plain = render_image(View([make_mark(gB)]; camera = top, refine_frames = 1), 96, 96)
        bricks = render_image(View([make_mark(tB)]; camera = top, refine_frames = 1), 96, 96)
        @test maximum(map((a, b) -> max(abs(Int(reinterpret(a.r)) - Int(reinterpret(b.r))),
                                        abs(Int(reinterpret(a.g)) - Int(reinterpret(b.g))),
                                        abs(Int(reinterpret(a.b)) - Int(reinterpret(b.b)))), plain, bricks)) <= 1
    end
    VR.glcheck()
end

@testitem "Clipped" setup = [GLContext] begin
    import ModernGL as GL, VolumeRendering as VR
    using IntervalSets, LinearAlgebra, StaticArrays

    sbox = Box(-12..12, -12..12, -5..5)
    @test_throws "Clipped shape" Clipped(1.0, sbox)
    cs = Clipped(Sphere(radius = 10.0), sbox)
    @test aabb(cs) == (; lo = [-10, -10, -5], hi = [10, 10, 5])
    @test bounds(cs).center == [0, 0, 0] && bounds(cs).radius ≈ norm(SVector(10.0, 10.0, 5.0))

    # equatorial band from above, from the side, and from inside
    sw = 128
    pixels = range(-12 + 12 / sw, 12 - 12 / sw; length = sw)
    radius2 = [x^2 + y^2 for y in reverse(pixels), x in pixels]
    ring = (75 .<= radius2) .& (radius2 .< 100)
    band = [x^2 + z^2 < 100 && abs(z) <= 5 for z in reverse(pixels), x in pixels]
    sdepth(shape; eye = SVector(0.0, 0.0, 30.0), lookat = SVector(0.0, 0.0, 0.0), up = SVector(0.0, 1.0, 0.0)) =
        VR.depth_image(View([Surface(shape)]; camera = Camera(eye, lookat, up, :ortho, 45.0, 12.0), refine_frames = 1), sw, sw)

    @test (sdepth(cs) .> 0) == ring
    @test (sdepth(cs; eye = SVector(0.0, 30.0, 0.0), up = SVector(0.0, 0.0, 1.0)) .> 0) == band
    @test (sdepth(cs; eye = SVector(0.0, 0.0, 0.0), lookat = SVector(0.0, 1.0, 0.0), up = SVector(0.0, 0.0, 1.0)) .> 0) == band

    # back wall visible through the clipped front cap (sphere centered at z = 2, clipped to z ∈ [-5, 5])
    shifted = Clipped(Sphere(; center = SVector(0.0, 0.0, 2.0), radius = 10.0), sbox)
    dep = sdepth(shifted)
    visible = (51 .<= radius2) .& (radius2 .< 100)
    @test (dep .> 0) == visible
    near = 1e-6 * 12.0
    expected_dep = zeros(Float32, sw, sw)
    for I in CartesianIndices(expected_dep)
        visible[I] || continue
        height = sqrt(100 - radius2[I])
        zhit = height <= 3 ? 2 + height : 2 - height
        expected_dep[I] = near / (30.0 - zhit)
    end
    @test maximum(abs.(dep .- expected_dep)) <= 1e-9

    # small box inside the sphere misses completely
    @test all(iszero, sdepth(Clipped(Sphere(radius = 10.0), Box(-2..2, -2..2, -2..2))))
    VR.glcheck()
end

@testitem "oblique Plane" setup = [GLContext] begin
    import ModernGL as GL, VolumeRendering as VR
    using AxisKeys, ColorSchemes, ColorTypes, FixedPointNumbers, IntervalSets, LinearAlgebra, StaticArrays

    box = Box(-1..1, -1..1, -1..1)
    c = Plane(SVector(1.0, 2.0, 3.0), 0.25, box)
    n = normalize(SVector(1.0, 2.0, 3.0))
    for pt in (c.shape.origin, c.shape.origin + c.shape.u, c.shape.origin + c.shape.v, c.shape.origin + c.shape.u + c.shape.v)
        @test dot(n, pt) ≈ 0.25
    end
    @test all(aabb(c).lo .>= -1) && all(aabb(c).hi .<= 1)
    @test aabb(Plane(SVector(1.0, 1.0, 1.0), 0.0, box)) == (; lo = [-1, -1, -1], hi = [1, 1, 1])
    @test_throws "normal must be finite and nonzero" Plane(SVector(0.0, 0.0, 0.0), 0.0, box)
    @test_throws "normal must be finite and nonzero" Plane(SVector(NaN, 0.0, 1.0), 0.0, box)
    @test_throws "pos must be finite" Plane(SVector(0.0, 0.0, 1.0), Inf, box)

    w = 80
    xyz = -1.0:0.1:1.0
    A = KeyedArray(Float32[x + 2y + 3z + 10 for x in xyz, y in xyz, z in xyz]; x = xyz, y = xyz, z = xyz)
    gf = GridField(A)
    tf = TiledField(A; maxdim = 12)
    cmap = Colormap(ColorSchemes.viridis, value_range(gf))

    function axis_cam(axis; half = 1.5, d = 3.0)
        eye = SVector(ntuple(i -> i == axis ? d : 0.0, 3))
        up = axis == 3 ? SVector(0.0, 1.0, 0.0) : SVector(0.0, 0.0, 1.0)
        Camera(eye, SVector(0.0, 0.0, 0.0), up, :ortho, 45.0, half)
    end
    function plane_points(cam)
        right, up, _ = VR.view_basis(cam)
        xs = [((col - 0.5) / w * 2 - 1) * cam.ortho_half for col in 1:w]
        ys = [(1 - (row - 0.5) / w * 2) * cam.ortho_half for row in 1:w]
        [cam.lookat + x * right + y * up for y in ys, x in xs]
    end
    buffers(marks, cam) = (v = View(marks; camera = cam, refine_frames = 1); (render_image(v, w, w), VR.depth_image(v, w, w)))

    # axis presets and custom normals agree
    for (i, axis) in enumerate((:x, :y, :z))
        cam = axis_cam(i)
        nvec = SVector(ntuple(j -> Float64(j == i), 3))
        @test buffers([Surface(Plane(axis, 0.2, box); color = Mapped(gf, cmap))], cam) ==
              buffers([Surface(Plane(nvec, 0.2, box); color = Mapped(gf, cmap))], cam)
    end

    # oblique (1, 1, 1) plane clips strictly to box
    ocam = Camera(SVector(3.0, 3.0, 3.0), SVector(0.0, 0.0, 0.0), SVector(0.0, 0.0, 1.0), :ortho, 45.0, 1.5)
    oplane = Plane(SVector(1.0, 1.0, 1.0), 0.0, box)
    img, dep = buffers([Surface(oplane; color = Mapped(gf, cmap))], ocam)
    expected = map(p -> all(abs.(p) .<= 1), plane_points(ocam))
    @test any(expected) && !all(expected)
    @test (dep .> 0) == expected
    @test any(!=(RGB{N0f8}(0, 0, 0)), img[expected])
    @test all(==(RGB{N0f8}(0, 0, 0)), img[.!expected])
    @test buffers([Surface(oplane; color = Mapped(tf, cmap))], ocam) == (img, dep)
    revA = KeyedArray(reverse(parent(A); dims = 3); x = xyz, y = xyz, z = reverse(xyz))
    @test buffers([Surface(oplane; color = Mapped(GridField(revA), cmap))], ocam) == (img, dep)
    @test all(iszero, buffers([Surface(Plane(SVector(1.0, 1.0, 1.0), 5.0, box); color = Mapped(gf, cmap))], ocam)[2])
    VR.glcheck()
end

@testitem "radial_planes" setup = [GLContext] begin
    import ModernGL as GL, VolumeRendering as VR
    using AxisKeys, ColorSchemes, ColorTypes, FixedPointNumbers, IntervalSets, StaticArrays, Unitful

    box = Box(-1..1, -1..1, -1..1)
    rad = radial_planes(box; center = (0.25, -0.5), nslices = 4, offset = 30u"°")
    rad_ref = radial_planes(box; center = (0.25, -0.5), nslices = 4, offset = π / 6)
    @test length(rad) == 4 && all(((a, b),) -> a.shape.origin ≈ b.shape.origin && a.shape.u ≈ b.shape.u && a.shape.v ≈ b.shape.v, zip(rad, rad_ref))
    r0 = first(radial_planes(box; center = (0.25, -0.5), nslices = 4, offset = 0.0)).shape
    @test r0.origin == [0.25, -0.5, -1] && r0.u[1] > 0 && r0.u[2] == r0.u[3] == 0 && r0.v == [0, 0, 2]
    @test_throws "at least 1" radial_planes(box; nslices = 0)
    @test_throws "offset must be finite" radial_planes(box; offset = Inf)
    @test_throws "center must be finite" radial_planes(box; center = (NaN, 0.0))

    w = 80
    xyz = -1.0:0.1:1.0
    A = KeyedArray(Float32[x + 2y + 3z + 10 for x in xyz, y in xyz, z in xyz]; x = xyz, y = xyz, z = xyz)
    gf = GridField(A)
    tf = TiledField(A; maxdim = 12)
    cmap = Colormap(ColorSchemes.viridis, value_range(gf))
    rcam = Camera(SVector(0.0, 3.0, 0.0), SVector(0.0, 0.0, 0.0), SVector(0.0, 0.0, 1.0), :ortho, 45.0, 1.5)
    right, up, _ = VR.view_basis(rcam)
    xs = [((col - 0.5) / w * 2 - 1) * rcam.ortho_half for col in 1:w]
    ys = [(1 - (row - 0.5) / w * 2) * rcam.ortho_half for row in 1:w]
    pts = [rcam.lookat + x * right + y * up for y in ys, x in xs]
    buffers(marks) = (v = View(marks; camera = rcam, refine_frames = 1); (render_image(v, w, w), VR.depth_image(v, w, w)))

    r1 = [Surface(p; color = Mapped(gf, cmap)) for p in radial_planes(box; center = (0.25, -0.2), nslices = 1)]
    @test (buffers(r1)[2] .> 0) == map(p -> 0.25 <= p[1] <= 1 && abs(p[3]) <= 1, pts)
    r1_pi = [Surface(p; color = Mapped(gf, cmap)) for p in radial_planes(box; center = (0.25, -0.2), nslices = 1, offset = π)]
    @test (buffers(r1_pi)[2] .> 0) == map(p -> -1 <= p[1] <= 0.25 && abs(p[3]) <= 1, pts)
    r2 = [Surface(p; color = Mapped(gf, cmap)) for p in radial_planes(box; center = (0.25, -0.2), nslices = 2)]
    r2_img, r2_dep = buffers(r2)
    full_img, full_dep = buffers([Surface(Plane(:y, -0.2, box); color = Mapped(gf, cmap))])
    @test r2_dep ≈ full_dep
    @test maximum(map((a, b) -> max(abs(Int(reinterpret(a.r)) - Int(reinterpret(b.r))),
                                    abs(Int(reinterpret(a.g)) - Int(reinterpret(b.g))),
                                    abs(Int(reinterpret(a.b)) - Int(reinterpret(b.b)))), r2_img, full_img)) <= 1
    r3_planes = radial_planes(box; center = (0.25, -0.2), nslices = 3, offset = π / 5)
    r3_buf = buffers([Surface(p; color = Mapped(gf, cmap)) for p in r3_planes])
    @test r3_buf != (r2_img, r2_dep)
    @test buffers([Surface(p; color = Mapped(tf, cmap)) for p in r3_planes]) == r3_buf
    VR.glcheck()
end

@testitem "SurfaceMesh" begin
    import VolumeRendering as VR
    using AxisKeys, IntervalSets

    heights = KeyedArray(Float32[10 20 30; 40 50 NaN]; x = [100, 300], y = [20, 25, 40])
    mesh = SurfaceMesh(heights)
    @test SurfaceMesh(mesh) === mesh
    @test size(mesh.vertices) == (3, 6)
    @test mesh.indices == UInt32[0, 1, 3, 0, 3, 2]
    @test all(isfinite, mesh.vertices[:, mesh.indices .+ 1])
    @test mesh.vertices[:, 1] == [100, 20, 10]
    @test isempty(SurfaceMesh(KeyedArray(fill(NaN32, 2, 2); x = 1:2, y = 1:2)).indices)
    for keys in ((1:3, 1:2), (3:-1:1, 2:-1:1))
        full = SurfaceMesh(KeyedArray(ones(3, 2); x = keys[1], y = keys[2]))
        @test size(full.vertices) == (3, 6) && length(full.indices) == 12 && maximum(full.indices) == 5
    end
    @test_throws "2-D KeyedArray" SurfaceMesh(ones(2, 2))
    @test_throws "at least 2 samples" SurfaceMesh(KeyedArray(ones(1, 2); x = 1:1, y = 1:2))
    @test_throws "coordinates must be finite" SurfaceMesh(KeyedArray(ones(2, 2); x = [0, Inf], y = [0, 1]))
    @test_throws "strictly increasing or decreasing" SurfaceMesh(KeyedArray(ones(2, 2); x = [0, 0], y = [0, 1]))
    @test_throws "heights must be finite or NaN" SurfaceMesh(KeyedArray(fill(Inf, 2, 2); x = 1:2, y = 1:2))

    x, y, vals = check_surface(heights)
    @test x == [100.0, 300.0] && y == [20.0, 25.0, 40.0] && isequal(vals, Matrix{Float64}(parent(heights)))
    @test value_range(heights) == value_range(mesh) == 10.0..50.0
    for val in (0f0, -20f0, 100f0)
        r = value_range(KeyedArray(fill(val, 2, 2); x = 1:2, y = 1:2))
        lo, hi = endpoints(r)
        @test isfinite(lo) && isfinite(hi) && lo < val < hi
    end
    @test value_range(KeyedArray(fill(NaN32, 2, 2); x = 1:2, y = 1:2)) == 0.0..1.0
end

@testitem "HeightSurface" setup = [GLContext] begin
    import ModernGL as GL, VolumeRendering as VR
    using AxisKeys, ColorSchemes, ColorTypes, FixedPointNumbers, IntervalSets, StaticArrays

    heights = KeyedArray(Float32[10 20 30; 40 50 NaN]; x = [100, 300], y = [20, 25, 40])
    mesh = SurfaceMesh(heights)
    hs = HeightSurface(heights)
    @test hs.mesh isa SurfaceMesh && hs.shading && !hs.flip_depth && isnothing(hs.clip)
    @test applicable(VR.controls!, hs) && !VR.jittered(hs)
    @test_throws "HeightSurface scale" HeightSurface(mesh; scale = SVector(1.0, Inf, 1.0))
    @test_throws "HeightSurface offset" HeightSurface(mesh; offset = SVector(NaN, 0.0, 0.0))
    @test_throws "HeightSurface colour" HeightSurface(mesh; color = [[RGB(1.0, 0.0, 0.0)]])

    # pixel-level rendering: opaque color, field sampling, box clipping, and VBO reuse
    w = 80
    xyz = -1.0:0.25:1.0
    A = KeyedArray(Float32[10 + x + 2y + 3z for x in xyz, y in xyz, z in xyz]; x = xyz, y = xyz, z = xyz)
    gf = GridField(A)
    tf = TiledField(A; maxdim = 6)
    cmap = Colormap(ColorSchemes.viridis, value_range(gf))
    box = Box(gf)
    flat = KeyedArray(zeros(Float32, 2, 2); x = -2:4:2, y = -2:4:2)
    top = GLContext.topcam(1.5)
    buffers(mark) = (v = View([mark]; camera = top, refine_frames = 1); (render_image(v, w, w), VR.depth_image(v, w, w), v))

    mark = HeightSurface(flat; color = RGB(1.0, 0.0, 0.0), clip = box, shading = false)
    img, dep, view = buffers(mark)
    ref_img, ref_dep, _ = buffers(Surface(Plane(:z, 0.0, box); color = Mapped(gf, cmap)))
    @test dep ≈ ref_dep rtol = 1e-6
    inside = dep .> 0
    @test any(inside) && !all(inside)
    @test all(==(RGB{N0f8}(1, 0, 0)), img[inside]) && all(==(RGB{N0f8}(0, 0, 0)), img[.!inside])
    mbuf = only(filter(x -> x isa VR.MeshBuffer, collect(values(view.cache.entries))))
    mark.color = Mapped(gf, cmap)
    sampled_img, sampled_dep = render_image(view, w, w), VR.depth_image(view, w, w)
    @test sampled_dep == dep
    @test maximum(map((a, b) -> max(abs(Int(reinterpret(a.r)) - Int(reinterpret(b.r))),
                                    abs(Int(reinterpret(a.g)) - Int(reinterpret(b.g))),
                                    abs(Int(reinterpret(a.b)) - Int(reinterpret(b.b)))), sampled_img, ref_img)) <= 1
    @test only(filter(x -> x isa VR.MeshBuffer, collect(values(view.cache.entries)))) === mbuf   # mesh buffer reused
    tiled_img, tiled_dep, _ = buffers(HeightSurface(flat; color = Mapped(tf, cmap), clip = box, shading = false))
    @test (tiled_img, tiled_dep) == (sampled_img, sampled_dep)

    # slope shading, scale exaggeration, flip_depth, and viewing from below
    roof = KeyedArray(Float32[0 0; 0.5 0.5; 0 0]; x = -1:1, y = -1:2:1)
    rmark = HeightSurface(roof; clip = box)
    rview = View([rmark]; camera = top, refine_frames = 1)
    shaded, rdep = render_image(rview, w, w), VR.depth_image(rview, w, w)
    r8(c) = Int(reinterpret(c.r))
    @test r8(shaded[40, 50]) > r8(shaded[40, 30]) + 10
    rmark.shading = false
    unlit, unlit_dep = render_image(rview, w, w), VR.depth_image(rview, w, w)
    @test unlit_dep == rdep && r8(unlit[40, 30]) == r8(unlit[40, 50]) && shaded != unlit
    rmark.shading = true; rmark.scale = SVector(1.0, 1.0, 2.0)
    exag = render_image(rview, w, w)
    @test r8(exag[40, 50]) - r8(exag[40, 30]) > r8(shaded[40, 50]) - r8(shaded[40, 30])
    rmark.scale = SVector(1.0, 1.0, 1.0); rmark.flip_depth = true
    flipped = render_image(rview, w, w)
    @test r8(flipped[40, 30]) > r8(flipped[40, 50]) + 10
    rmark.flip_depth = false
    rview.camera.eye = -rview.camera.eye
    below, below_dep = render_image(rview, w, w), VR.depth_image(rview, w, w)
    @test any(>(0), below_dep) && all(c -> r8(c) >= 40, below[below_dep .> 0])

    # height-value coloring (Mapped(mesh, cmap) and Colormap): flip_depth flips depth and geometry while keeping value colors
    vcmap = Colormap(ColorSchemes.gray1, 0.0..0.5)
    vmark = HeightSurface(roof; color = Mapped(rmark.mesh, vcmap), shading = false)
    vview = View([vmark]; camera = top, refine_frames = 1)
    vimg1, vdep1 = render_image(vview, w, w), VR.depth_image(vview, w, w)
    vmark.flip_depth = true
    vimg2, vdep2 = render_image(vview, w, w), VR.depth_image(vview, w, w)
    @test vimg1 == vimg2 && vdep1 != vdep2
    vmark.color = vcmap
    @test render_image(vview, w, w) == vimg1

    # depth sign clipping, all-NaN surface, and ResourceCache cleanup on removal
    neg_box = Box(-1..1, -1..1, -1..0)
    dmark = HeightSurface(KeyedArray(fill(0.5f0, 2, 2); x = -1:2:1, y = -1:2:1); clip = neg_box)
    dview = View([dmark]; camera = top, refine_frames = 1)
    @test all(iszero, VR.depth_image(dview, w, w))
    dmark.flip_depth = true
    @test any(>(0), VR.depth_image(dview, w, w))
    dmark.mesh = KeyedArray(fill(NaN32, 2, 2); x = -1:2:1, y = -1:2:1)
    @test all(iszero, VR.depth_image(dview, w, w)) && isempty(dmark.mesh.indices)
    buf = only(filter(x -> x isa VR.MeshBuffer, collect(values(dview.cache.entries))))
    empty!(dview.marks)
    render!(dview, w, w)
    @test GL.glIsBuffer(buf.vbo) == GL.GL_FALSE && GL.glIsBuffer(buf.ebo) == GL.GL_FALSE
    VR.glcheck()
end

@testitem "Colormap, Mapped" begin
    using ColorSchemes
    import VolumeRendering as VR
    using IntervalSets, AxisKeys
    c = Colormap(ColorSchemes.viridis, 0..1)
    @test (c.scheme, c.window, c.log) == (ColorSchemes.viridis, 0.0..1.0, false)
    @test Colormap(ColorSchemes.afmhot, 1e-3..1.0; log = true).log
    @test_throws "minimum < maximum" Colormap(ColorSchemes.viridis, 1..1)
    @test_throws "minimum < maximum" Colormap(ColorSchemes.viridis, 2..1)
    @test_throws "positive" Colormap(ColorSchemes.viridis, 0..1; log = true)
    @test_throws "positive" Colormap(ColorSchemes.viridis, -1..1; log = true)

    g = GridField(KeyedArray(rand(2, 2, 2); x = 0:1, y = 0:1, z = 0:1))
    vals = [[1.0, NaN], [-2.0, 3.0, 0.5]]
    @test Mapped(g, c).source === g
    @test Mapped(vals, c).source === vals
    @test value_range(vals) === -2.0..3.0
    colorfield = GLSLField(""; kind = :color)
    @test_throws "Mapped source" Mapped(colorfield, c)
    @test_throws "Mapped source" Mapped(1.0, c)
    @test_throws MethodError Mapped(g, :viridis)
    @test_throws MethodError Colormap(:viridis, 0..1)                 # the scheme is a ColorScheme object, not a name
end

@testitem "marks: construction and validate" begin
    using ColorSchemes
    import VolumeRendering as VR
    using AxisKeys, IntervalSets, ColorTypes, StaticArrays

    g = GridField(KeyedArray(rand(2, 2, 2); x = 0:1, y = 0:1, z = 0:1))
    rgba = GridField(KeyedArray(rand(RGBA{Float32}, 2, 2, 2); x = 0:1, y = 0:1, z = 0:1))
    m = Mapped(g, Colormap(ColorSchemes.viridis, 0..1))
    box = Box(g)
    pl = [[SVector(0.0, 0, 0), SVector(1.0, 0, 0)], [SVector(0.0, 1, 0), SVector(1.0, 1, 0), SVector(1.0, 2, 0)]]

    v = Volume(m; region = box, label = "v")
    @test (v.color, v.region, v.extinction, v.mode, v.density, v.step_scale, v.max_samples, v.label) ==
          (m, box, [[0, 0], [1, 1]], :dvr, 1.0, 0.5, 4096, "v")
    @test Volume(rgba; region = Sphere(radius = 1)).color === rgba
    @test Volume(m; region = box, mode = :mip).mode === :mip
    @test_throws "requires a Mapped" Volume(rgba; region = box, mode = :mip)            # MIP on a colour field
    @test_throws "Volume colour" Volume(RGB(1.0, 0, 0); region = box)                  # constant colour
    @test_throws "Volume colour" Volume(Mapped([[1.0, 2.0]], m.colormap); region = box)
    @test_throws "mark colour" Volume(g; region = box)                                 # bare :scalar field
    @test_throws UndefKeywordError Volume(m)                                           # missing region
    @test_throws "Volume region" Volume(m; region = nothing)
    @test_throws "mode" Volume(m; region = box, mode = :max)
    @test_throws "at least 2" Volume(m; region = box, extinction = [SVector(0.0, 0.0)])
    @test_throws "sorted" Volume(m; region = box, extinction = [SVector(1.0, 0.0), SVector(0.0, 1.0)])
    @test_throws "0..1" Volume(m; region = box, extinction = [SVector(0.0, 0.0), SVector(1.0, 2.0)])
    @test_throws "density" Volume(m; region = box, density = 0)
    @test_throws "step_scale" Volume(m; region = box, step_scale = -1)
    @test_throws "max_samples" Volume(m; region = box, max_samples = 0)
    @test_throws "max_samples" Volume(m; region = box, max_samples = 2^31)
    @test Volume(rgba; region = box, extinction = SVector{2,Float64}[]).extinction == []   # only for Mapped colours

    s = Surface(Sphere(radius = 1))
    @test (s.color, s.opacity, s.label) == (RGB(0.5, 0.5, 0.55), 1.0, nothing)
    @test Surface(box; color = m, opacity = 0.5).color === m
    @test Surface(box; color = rgba).color === rgba
    @test_throws "alpha-less" Surface(box; color = RGBA(1.0, 0, 0, 0.5))
    @test_throws "Surface colour" Surface(box; color = Mapped([[1.0, 2.0]], m.colormap))
    @test_throws "Surface colour" Surface(box; color = [[RGB(1.0, 0, 0)]])
    @test_throws "opacity" Surface(box; opacity = 0)
    @test_throws "opacity" Surface(box; opacity = 1.5)
    @test_throws "Surface shape" Surface(1.0)

    l = Lines(pl)
    @test (l.polylines, l.color, l.width_px) == (pl, RGB(0.8, 0.8, 0.85), 1.0)
    colors = [[RGB(1.0, 0, 0), RGB(0.0, 1, 0)], [RGB(0.0, 0, 1), RGB(1.0, 1, 1), RGB(0.0, 0, 0)]]
    @test Lines(pl; color = colors).color === colors
    @test Lines(pl; color = Mapped([[1.0, 2.0], [3.0, 4.0, 5.0]], m.colormap)).color isa Mapped
    @test Lines(pl; color = m).color === m
    @test Lines(pl; color = rgba).color === rgba
    @test_throws "nesting" Lines(pl; color = [[RGB(1.0, 0, 0), RGB(0.0, 1, 0)], [RGB(0.0, 0, 1)]])
    @test_throws "nesting" Lines(pl; color = [colors[1]])
    @test_throws "nesting" Lines(pl; color = Mapped([[1.0, 2.0], [3.0, 4.0]], m.colormap))
    @test_throws "alpha-less" Lines(pl; color = [[RGBA(1.0, 0, 0, 1), RGBA(0.0, 1, 0, 1)], [RGBA(0.0, 0, 1, 1) for _ in 1:3]])
    @test_throws "width_px" Lines(pl; width_px = 0)
    @test_throws "mark colour" Lines(pl; color = g)

    # validation runs at construction and render, never on assignment
    v.mode = :average; v.color = rgba
    @test_throws "requires a Mapped" VR.validate(v)
    s.color = m
    m.colormap.log = true                               # window 0..1: invariant violated, caught at render
    @test_throws "positive" VR.validate(s)
    m.colormap.log = false; m.source = rgba
    @test_throws "Mapped source" VR.validate(s)

    @test VR.jittered(Volume(Mapped(g, Colormap(ColorSchemes.viridis, 0..1)); region = box))
    @test !VR.jittered(l) && !VR.jittered(Surface(box))
end

@testitem "GLSL identifiers" begin
    import VolumeRendering as VR
    using StaticArrays, IntervalSets
    @test VR.check_identifier(:rho) === :rho
    @test all(n -> VR.check_identifier(n) === n, [:_a, :A9_b, :Rt, :x_])
    for (bad, msg) in [(Symbol("9a"), "invalid"), (Symbol("a-b"), "invalid"), (Symbol(""), "invalid"), (Symbol("a b"), "invalid"),
                       (:ρ, "invalid"), (:naïve, "invalid"), (Symbol("a\n"), "invalid"),
                       (:vr_cmap, "vr_"), (:gl_Position, "gl_"), (:a__b, "__"), (:__a, "__"),
                       (:filter, "keyword"), (:sample, "keyword"), (:in, "keyword"), (:output, "keyword"),
                       (:row_major, "keyword"), (:float, "keyword"), (:precise, "keyword"),
                       (:value, "renderer"), (:stepSize, "renderer"), (:main, "renderer")]
        @test_throws msg VR.check_identifier(bad)
    end

    # user-defined objects follow the same rules as the built-ins
    struct Obj; t::NamedTuple; u::NamedTuple; end
    VR.textures(o::Obj) = o.t
    VR.uniforms(o::Obj) = o.u
    T = Texture(rand(2))
    sphere = Sphere(radius = 1)
    @test isnothing(VR.check_declared_names([Obj((; lut = T), (; k = 1.0)), sphere]))
    @test_throws "vr_" VR.check_declared_names([Obj((;), (; vr_k = 1.0))])
    @test_throws "keyword" VR.check_declared_names([Obj((; filter = T), (;))])
    @test_throws "twice" VR.check_declared_names([Obj((; k = T), (; k = 1.0))])
    err = try VR.check_declared_names([Obj((;), (; sphere_radius = 1.0)), sphere]) catch e; e end
    @test err isa ArgumentError && occursin("Obj", err.msg) && occursin("Sphere", err.msg)
    # a field and a shape assembled into one program
    field = GLSLField(""; textures = (; lut = T), uniforms = (; box_lo = 1.0))
    err = try VR.check_declared_names([field, Box(0..1, 0..1, 0..1)]) catch e; e end
    @test err isa ArgumentError && occursin("box_lo", err.msg) && occursin("GLSLField", err.msg) && occursin("Box", err.msg)
end

@testitem "View" begin
    import VolumeRendering as VR
    using StaticArrays, ColorTypes
    cam = Camera(; radius = 2)
    s = Surface(Sphere(radius = 1))
    v = View([s]; camera = cam)
    @test v.marks == [s] && v.marks isa Vector{Any}
    @test v.camera === cam && v.initial_camera !== cam && v.initial_camera.eye == cam.eye
    @test (v.background, v.max_volume_px, v.refine_frames, v.flymode) == (RGB(0, 0, 0), 2048, 16, false)
    @test v.fly_speed ≈ 6 / 200
    @test isnothing(v.cache) && isnothing(v.last) && v.nsamp == 0          # construction allocates no GL resources
    cam.projection = :ortho
    @test VR.default_fly_speed(cam) ≈ 2.2 / 200
    orbit!(cam, 1, 1); zoom!(cam, 3, v.initial_camera)
    @test reset!(v) === v && v.camera === cam                            # restored in place
    @test all(n -> getfield(cam, n) == getfield(Camera(; radius = 2), n), fieldnames(Camera))
    @test View(Any[]; camera = cam, fly_speed = 1, refine_frames = 1).fly_speed == 1
    @test_throws "refine_frames" View(Any[]; camera = cam, refine_frames = 0)
    @test_throws "max_volume_px" View(Any[]; camera = cam, max_volume_px = 0)
    v.refine_frames = 0                                                  # plain assignment; caught at render
    @test_throws "refine_frames" VR.validate(v)
end

@testitem "snapshot" begin
    using ColorSchemes
    import VolumeRendering as VR
    using AxisKeys, IntervalSets, ColorTypes, StaticArrays

    grid(n) = KeyedArray(rand(Float32, n, n, n); x = 1:n, y = 1:n, z = 1:n)
    pl, vals = [[SVector(0.0, 0, 0), SVector(1.0, 0, 0)]], [[NaN, 1.0]]
    function scene(A)
        g = GridField(A)
        m = Mapped(g, Colormap(ColorSchemes.viridis, 0..1))
        vol = Volume(m; region = Box(g))
        sl = Surface(Plane(:z, 2, Box(g)); color = m)
        ln = Lines(pl; color = Mapped(vals, m.colormap))
        (; view = View([vol, sl, ln]; camera = Camera(; bounds(vol.region)...)), g, m, vol, sl, ln)
    end
    snap(view) = VR.snapshot(view, 64, 48)
    same(f) = (s0 = snap(sc.view); f(); isequal(s0, snap(sc.view)))

    A = grid(4)
    sc = scene(A)
    @test isequal(snap(sc.view), snap(sc.view))                     # NaN leaves compare equal to themselves
    @test isequal(snap(sc.view), snap(scene(A).view))               # equal values, distinct objects (same bulk arrays)
    @test !isequal(VR.snapshot(sc.view, 64, 48), VR.snapshot(sc.view, 65, 48))
    @test same(() -> sc.m.colormap = Colormap(ColorSchemes.viridis, 0..1))       # equal values → equal
    @test same(() -> sc.g.data = KeyedArray(A; x = 1:4, y = 1:4, z = 1:4))   # new wrapper, same parent array
    @test same(() -> A[1] = 99)                                      # in-place edit: undetected
    @test same(() -> sc.ln.polylines[1][1] = SVector(5.0, 0, 0))
    @test !same(() -> sc.g.data = grid(4))                           # new array
    @test !same(() -> sc.ln.polylines = [[SVector(0.0, 0, 0), SVector(1.0, 0, 0)]])
    @test !same(() -> sc.ln.color.source = [[NaN, 1.0]])
    @test !same(() -> sc.m.colormap.window = 0..2)
    @test !same(() -> sc.m.colormap.log = true)
    @test !same(() -> sc.m.colormap.scheme = ColorSchemes.inferno)
    @test !same(() -> sc.g.interpolation = :nearest)
    @test !same(() -> sc.vol.extinction[2] = SVector(1.0, 0.5))     # by value, also in place
    @test !same(() -> sc.vol.region = Box(0..1, 0..1, 0..1))
    @test !same(() -> sc.vol.max_samples = 100)
    @test !same(() -> sc.sl.shape = Plane(:z, 3, sc.vol.region))
    @test !same(() -> sc.sl.opacity = 0.5)
    @test !same(() -> orbit!(sc.view.camera, 0.1, 0))
    @test !same(() -> sc.view.flymode = true)
    @test !same(() -> push!(sc.view.marks, Surface(Sphere(radius = 1))))
    @test same(() -> sc.vol.label = "volume")                        # labels are UI only

    # cost O(marks): independent of the grid size
    small, large = scene(grid(4)).view, scene(grid(160)).view
    t(view) = minimum(_ -> @elapsed(snap(view)), 1:300)
    t(small); t(large)
    @test t(large) < 3 * t(small)
end

@testitem "generated GLSL compiles" setup = [GLContext] begin
    import ModernGL as GL, VolumeRendering as VR
    using AxisKeys, ColorTypes, StaticArrays, IntervalSets
    compile(src) = (s = VR.compile_shader(GL.GL_FRAGMENT_SHADER, "#version 410 core\n" * src); GL.glDeleteShader(s); true)
    shape_decls = [(Sphere(radius = 1), "uniform vec3 sphere_center; uniform float sphere_radius;"),
                   (Box(0..1, 0..1, 0..1), "uniform vec3 box_lo, box_hi;"),
                   (Plane(:x, 0.5, Box(0..1, 0..1, 0..1)), "uniform vec3 plane_origin, plane_u, plane_v;")]
    for (s, decls) in shape_decls
        @test compile("$decls\n$(shape_glsl(s))\nout vec4 frag; void main() { frag = vec4(vr_hit(vec3(0.0), vec3(1.0, 0.0, 0.0)), 0.0, 1.0); }")
    end
    fields = [GridField(KeyedArray(rand(2, 3, 4); z = 0:1, x = 0.0:0.5:1.0, y = 4.0:-1.0:1.0)),
              GridField(KeyedArray(rand(RGBA{Float32}, 2, 2, 2); x = 0:1, y = 0:1, z = 0:1)),
              GridField(KeyedArray(rand(3, 4); lat = 1.0:-1.0:-1.0, lon = 0.0:1.5:4.5)),
              GridField(KeyedArray(rand(4, 3); lon = 3.0:-2.0:-3.0, lat = -1.0:1.0:1.0))]
    for f in fields
        sampler = ndims(f.data) == 3 ? "sampler3D" : "sampler2D"
        out = value_kind(f) === :color ? "value(p)" : "vec4(value(p))"
        @test compile("uniform $sampler grid;\n$(field_glsl(f))\nout vec4 frag; void main() { vec3 p = vec3(1.0); frag = $out + stepSize(p); }")
    end
    VR.glcheck()
end

@testitem "program assembly" begin
    import VolumeRendering as VR
    using StaticArrays, ColorTypes, ColorSchemes, IntervalSets
    cam = Camera(; radius = 1)
    asm(m, cam, w, h) = last(only(VR.assemble(m, (; camera = cam, w, h, vsize = (w, h), sample = 0))))

    # uniform type table
    for (x, T) in [(true, "bool"), (3, "int"), (Int8(3), "int"), (2.5, "float"), (1//2, "float"), (π, "float"),
                   (SVector(1, 2), "ivec2"), (SVector(1, 2, 3, 4), "ivec4"), (SVector(1.0, 2.0, 3.0), "vec3"),
                   (SVector(1f0, 2f0, 3f0, 4f0), "vec4"), (SMatrix{3,3}(1.0:9), "mat3"), (SMatrix{4,4}(1:16), "mat4")]
        @test VR.uniform_decl(:k, x) == "uniform $T k;\n"
    end
    for bad in [SVector(1.0), SVector{5}(1.0:5), SMatrix{2,2}(1.0:4), [1.0, 2.0], "s", :s, 1.0 + 2im, nothing]
        @test_throws "uniform k: unsupported" VR.uniform_decl(:k, bad)
    end

    cmap = Colormap(ColorSchemes.gray1, 0.5..2.0)
    glsl = "float value(vec3 p) { return 0.0; } float stepSize(vec3 p) { return 1.0; }"
    surf(f; shape = Sphere(radius = 1)) = Surface(shape; color = Mapped(f, cmap))

    # declaration order: textures (units 0, 1, …), then uniforms; the colormap follows
    f = GLSLField(glsl; textures = (; a = Texture(rand(2, 2, 2)), b = Texture(rand(4))), uniforms = (; k = 1.0, R = SMatrix{3,3}(1.0:9)))
    d = asm(surf(f; shape = Box(0..1, 0..1, 0..1)), cam, 64, 48)
    @test occursin("""
        uniform sampler3D a;
        uniform sampler1D b;
        uniform float k;
        uniform mat3 R;
        uniform vec3 box_lo;
        uniform vec3 box_hi;
        uniform sampler1D vr_cmap;
        uniform float vr_cmap_lo, vr_cmap_hi;
        uniform float vr_opacity;
        """, d.frag)
    @test startswith(d.frag, "#version 410 core\n")
    @test occursin(glsl, d.frag) && occursin("vec2 vr_hit", d.frag)
    @test findfirst(glsl, d.frag)[1] < findfirst("vec2 vr_hit", d.frag)[1] < findfirst("float vr_norm", d.frag)[1] <
          findfirst("vec4 vr_color", d.frag)[1] < findfirst("void main", d.frag)[1]
    @test map(u -> u.name, d.units) == [:a, :b, :vr_cmap]
    @test (d.uniforms.k, d.uniforms.R, d.uniforms.box_lo, d.uniforms.vr_cmap_lo, d.uniforms.vr_cmap_hi, d.uniforms.vr_opacity) ==
          (1.0, SMatrix{3,3}(1.0:9), [0, 0, 0], 0.5, 2.0, 1.0)
    @test d.uniforms.vr_viewport === SVector(64.0, 48.0) && d.uniforms.vr_aspect == 64 / 48

    # the colormap's log flag is baked into the text; the window is uploaded as log(lo), log(hi)
    cmap.log = true
    dl = asm(surf(f; shape = Box(0..1, 0..1, 0..1)), cam, 64, 48)
    @test occursin("log(max(v, 1e-30))", dl.frag) && !occursin("log(max(v, 1e-30))", d.frag)
    @test (dl.uniforms.vr_cmap_lo, dl.uniforms.vr_cmap_hi) == (log(0.5), log(2.0))
    cmap.log = false

    # one program text per structure: values go to uniforms
    s1 = asm(Surface(Sphere(radius = 1); color = RGB(1.0, 0, 0)), cam, 64, 48)
    s2 = asm(Surface(Sphere(; center = SVector(1.0, 2, 3), radius = 2); color = RGB(0.0, 1, 0), opacity = 1), cam, 32, 32)
    @test (s1.vert, s1.frag) == (s2.vert, s2.frag)
    @test s2.uniforms.vr_rgb === SVector(0f0, 1f0, 0f0) && s2.uniforms.sphere_radius == 2
    pl = [[SVector(0.0, 0, 0), SVector(1.0, 0, 0)]]
    srcs = map(c -> (d = asm(Lines(pl; color = c), cam, 64, 48); (d.vert, d.frag)),
               [RGB(1.0, 0, 0), [[RGB(1.0, 0, 0), RGB(0.0, 1, 0)]], Mapped([[1.0, 2.0]], cmap), Mapped(f, cmap)])
    @test allunique(srcs)
    @test occursin("layout(location = 2) in vec3 vr_pa;", srcs[2][1]) && occursin("flat in vec3 vr_va, vr_vb;", srcs[2][2])
    @test occursin("layout(location = 2) in float vr_pa;", srcs[3][1]) && occursin("flat in float vr_va, vr_vb;", srcs[3][2])
    @test !occursin("vr_pa", srcs[1][1])

    # error paths, all before any GL call
    many(n) = GLSLField(glsl; textures = NamedTuple{Tuple(Symbol.("t", 1:n))}(ntuple(_ -> Texture(rand(4)), n)))
    @test length(asm(surf(many(15)), cam, 64, 48).units) == 16      # 15 field textures + the colormap
    @test_throws "17 samplers, more than GL_MAX_TEXTURE_IMAGE_UNITS = 16" asm(surf(many(16)), cam, 64, 48)
    @test_throws "uniform k: unsupported" asm(surf(GLSLField(glsl; uniforms = (; k = [1.0]))), cam, 64, 48)
    @test_throws "declared by both" asm(surf(GLSLField(glsl; uniforms = (; sphere_radius = 1.0))), cam, 64, 48)
    @test_throws "vr_" asm(surf(GLSLField(glsl; uniforms = (; vr_k = 1.0))), cam, 64, 48)
    @test_throws "unsupported" asm(Lines(pl; color = Mapped(GLSLField(glsl; uniforms = (; k = "x")), cmap)), cam, 64, 48)

    # impostor rectangle: projected aabb ±1 px, clamped; full screen when a corner is behind the eye
    top = Camera(SVector(0.0, 0, 3), SVector(0.0, 0, 0), SVector(0.0, 1, 0), :ortho, 90.0, 2.0)
    @test VR.impostor_rect(Box(-1..0.5, 0..1, -1..1), top, 40, 20) ≈ [-0.25 - 0.05, 0 - 0.1, 0.125 + 0.05, 0.5 + 0.1]
    @test VR.impostor_rect(Box(-9..9, 0..1, -1..1), top, 40, 20) ≈ [-1.05, -0.1, 1.05, 0.6]
    persp = Camera(SVector(0.0, 0, 3), SVector(0.0, 0, 0), SVector(0.0, 1, 0), :perspective, 90.0, 1.0)
    @test VR.impostor_rect(Sphere(; center = SVector(0.0, 0, 2.5), radius = 1), persp, 40, 20) == [-1, -1, 1, 1]
end

@testitem "render! frame protocol" setup = [GLContext] begin
    import ModernGL as GL, VolumeRendering as VR
    using StaticArrays, ColorTypes, FixedPointNumbers
    red, blue, green = RGB{N0f8}(1, 0, 0), RGB{N0f8}(0, 0, 1), RGB{N0f8}(0, 1, 0)
    scene() = View([Surface(Sphere(radius = 0.5); color = RGB(1.0, 0, 0))]; camera = GLContext.topcam(), background = RGB(0.0, 0.0, 1.0))
    view = scene()
    @test_throws ArgumentError render!(view, 0, 8)
    @test_throws ArgumentError render!(view, 8, 0)
    @test isnothing(view.cache)

    # hostile host state; every pass sets its own and the host's framebuffer is restored
    host = VR.scene_framebuffer(4, 4)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, host.fbo)
    GL.glEnable(GL.GL_SCISSOR_TEST); GL.glScissor(0, 0, 1, 1)
    GL.glEnable(GL.GL_CULL_FACE); GL.glCullFace(GL.GL_FRONT_AND_BACK)
    GL.glColorMask(false, false, false, false); GL.glDepthMask(false); GL.glDepthFunc(GL.GL_NEVER); GL.glDepthRange(0.5, 0.5)
    GL.glEnable(GL.GL_BLEND); GL.glBlendFunc(GL.GL_ZERO, GL.GL_ZERO); GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
    tex = render!(view, 32, 24)
    fbo = Ref{GL.GLint}(0); GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING, fbo)
    @test fbo[] == host.fbo
    @test tex == view.cache.scene_fb.color.id && view.nsamp == 1
    img = render_image(view, 32, 24)
    GL.glDisable(GL.GL_SCISSOR_TEST); GL.glDisable(GL.GL_CULL_FACE); GL.glCullFace(GL.GL_BACK)
    GL.glColorMask(true, true, true, true); GL.glDepthMask(true); GL.glDepthFunc(GL.GL_LESS); GL.glDepthRange(0, 1)
    GL.glDisable(GL.GL_BLEND); GL.glBlendFunc(GL.GL_ONE, GL.GL_ZERO); GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    @test img == render_image(scene(), 32, 24)                       # same as with clean state
    @test size(img) == (24, 32) && count(==(blue), img) + count(==(red), img) == 24 * 32
    @test count(==(red), img) ≈ π * 6^2 rtol = 0.1

    # render on demand: an unchanged snapshot skips the frame entirely (the clobbered scene stays)
    clobber!() = (GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, view.cache.scene_fb.fbo); GL.glClearColor(0, 1, 0, 1);
                  GL.glClear(GL.GL_COLOR_BUFFER_BIT); GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0))
    clobber!()
    @test render!(view, 32, 24) == tex && view.nsamp == 1
    view.marks[1].color = RGB(1.0, 0, 0)                              # equal value
    @test all(==(green), render_image(view, 32, 24))
    view.marks[1].shape = Sphere(radius = 0.75)
    big = render_image(view, 32, 24)
    @test count(==(red), big) ≈ π * 9^2 rtol = 0.1
    clobber!()
    @test !any(==(green), render_image(view, 33, 24))                # a size change re-renders
    @test count(==(red), render_image(view, 32, 24)) == count(==(red), big)

    # validation runs at render
    view.marks[1].opacity = 2
    @test_throws "opacity" render!(view, 32, 24)
    view.marks[1].opacity = 1
    view.refine_frames = 0
    @test_throws "refine_frames" render!(view, 32, 24)
    view.refine_frames = 16

    # free! deletes every GL resource; the next render! starts over
    prog = only(values(view.cache.entries)); scene_fb = view.cache.scene_fb
    @test free!(view) === nothing
    @test isnothing(view.cache) && GL.glIsFramebuffer(scene_fb.fbo) == GL.GL_FALSE
    GL.glUseProgram(0)                                                # a deleted program stays alive while current (GL rule)
    @test GL.glIsProgram(prog.id) == GL.GL_FALSE
    @test render_image(view, 32, 24) == big
    free!(view); free!(view)                                          # idempotent
    VR.gl_delete(host)
    VR.glcheck()
end

@testitem "depth and placement" setup = [GLContext] begin
    import VolumeRendering as VR
    using StaticArrays, ColorTypes, FixedPointNumbers, IntervalSets
    red, green = RGB{N0f8}(1, 0, 0), RGB{N0f8}(0, 1, 0)
    sphere(r = 1.0; center = SVector(0.0, 0, 0)) = Surface(Sphere(; center, radius = r); color = RGB(1.0, 0, 0))
    line(pts...) = Lines([collect(pts)]; color = RGB(0.0, 1, 0))
    top = GLContext.topcam()

    # opaque sphere silhouette: exactly the pixel centres inside the disc (ortho), area π r² (perspective)
    img = render_image(View([sphere(0.5)]; camera = top), 64, 64)
    @test (img .== red) == [hypot(GLContext.pixel_xy(1, 64, r, c)...) < 0.5 for r in 1:64, c in 1:64]
    persp = render_image(View([sphere(1.0)]; camera = GLContext.topcam(; projection = :perspective)), 64, 64)
    @test count(==(red), persp) == count(hypot(r - 32.5, c - 32.5) < tan(asin(1 / 3)) * 32 for r in 1:64, c in 1:64)   # angular radius asin(1/3), fs = 1

    # a face-on Plane equals the near face of the Box it cuts
    box = Box(-0.5..0.5, -0.4..0.3, -0.5..0.5)
    oblique = Camera(SVector(0.3, 0.2, 3), SVector(0.0, 0, 0), SVector(0.0, 1, 0), :perspective, 40.0, 1.0)
    faceon(shape) = View([Surface(shape; color = RGB(1.0, 0, 0))]; camera = Camera(SVector(0.1, -0.1, 3), SVector(0.1, -0.1, 0), SVector(0.0, 1, 0), :perspective, 40.0, 1.0))
    @test render_image(faceon(box), 48, 40) == render_image(faceon(Plane(:z, 0.5, box)), 48, 40)
    @test VR.depth_image(faceon(box), 48, 40) ≈ VR.depth_image(faceon(Plane(:z, 0.5, box)), 48, 40) rtol = 1e-6
    @test count(==(red), render_image(faceon(Plane(:z, 0.5, box)), 48, 40)) > 300

    # eye inside a sphere: its far surface fills the view at t = radius
    inside = Camera(SVector(0.0, 0, 0), SVector(0.0, 0, -1), SVector(0.0, 1, 0), :perspective, 60.0, 1.0)
    @test all(==(red), render_image(View([sphere(1.0)]; camera = inside), 16, 12))
    @test VR.depth_image(View([sphere(1.0)]; camera = inside), 16, 12) ≈ fill(1e-6, 12, 16) rtol = 1e-5

    # a 1 px line lands on the computed pixel row; a new polylines vector moves it
    ln = line(SVector(-0.9, 0.335, 0), SVector(0.9, 0.335, 0))       # pixel y = 26.7 from the bottom of 40 → row 14
    v = View([ln]; camera = top)
    @test findall(==(green), render_image(v, 40, 40)[:, 20]) == [14]
    ln.polylines = [[SVector(-0.9, -0.2125, 0), SVector(0.9, -0.2125, 0)]]   # pixel y = 15.75 → row 25
    @test findall(==(green), render_image(v, 40, 40)[:, 20]) == [25]

    # list-order tie rule: a line tangent to an opaque sphere at its pole (exact depth tie at the centre pixel)
    top2 = GLContext.topcam(2.0)
    tangent = line(SVector(-1.0, 0, 1), SVector(1.0, 0, 1))
    alone = render_image(View([tangent]; camera = top2), 65, 65) .== green
    @test count(alone) == 33 && alone[33, 33]
    @test VR.depth_image(View([tangent]; camera = top2), 65, 65)[33, 33] == VR.depth_image(View([sphere()]; camera = top2), 65, 65)[33, 33]
    @test (render_image(View([sphere(), tangent]; camera = top2), 65, 65) .== green) == alone      # listed after: no pixel lost
    before = render_image(View([tangent, sphere()]; camera = top2), 65, 65)
    @test (before .== green) == (alone .& ((1:65) .!= 33)') && before[33, 33] == red               # listed before: hidden where it touches

    # a line behind an opaque sphere is hidden, in front is not
    row = findall(==(green), render_image(View([line(SVector(-0.9, 0.078125, 0.8), SVector(0.9, 0.078125, 0.8))]; camera = top), 32, 32)[:, 10])
    behind = render_image(View([sphere(0.5), line(SVector(-0.9, 0.078125, -0.8), SVector(0.9, 0.078125, -0.8))]; camera = top), 32, 32)
    front = render_image(View([sphere(0.5), line(SVector(-0.9, 0.078125, 0.8), SVector(0.9, 0.078125, 0.8))]; camera = top), 32, 32)
    @test only(row) == 15
    @test all(==(red), behind[15, 12:21]) && behind[15, 3] == green
    @test all(==(green), front[15, 3:30])

    # orientation: a mark in the upper-left world quadrant lands at low row / low column, in both readbacks
    v = View([sphere(0.25; center = SVector(-0.5, 0.5, 0))]; camera = top)
    img, dep = render_image(v, 40, 32), VR.depth_image(v, 40, 32)
    @test size(img) == size(dep) == (32, 40)
    hit = findall(==(red), img)
    @test !isempty(hit) && all(i -> i[1] <= 16 && i[2] <= 20, hit)
    @test findall(>(0), dep) == hit
end

@testitem "ray hits: Sphere, Box, Plane" setup = [GLContext] begin
    import VolumeRendering as VR
    using StaticArrays, ColorTypes, IntervalSets, LinearAlgebra
    # independent Float64 reference: the pixel-centre ray of `vr_ray` and analytic hits
    function ray(cam, w, h, row, col)
        right, up, fwd = VR.view_basis(cam); fs = VR.fovscale(cam)
        x, y = (col - 0.5) / w * 2 - 1, (h - row + 0.5) / h * 2 - 1
        cam.projection === :ortho ? (cam.eye + x * fs * w / h * right + y * fs * up, fwd) :
                                    (cam.eye, normalize(fwd + x * fs * w / h * right + y * fs * up))
    end
    function hit(s::Sphere, ro, rd)
        oc = ro - s.center; b = dot(oc, rd); disc = b^2 - (dot(oc, oc) - s.radius^2)
        disc < 0 ? NaN : (t = -b - sqrt(disc); t > 0 ? t : -b + sqrt(disc))
    end
    function hit(bx::Box, ro, rd)
        (; lo, hi) = aabb(bx); t0, t1 = (lo - ro) ./ rd, (hi - ro) ./ rd
        tn, tf = maximum(min.(t0, t1)), minimum(max.(t0, t1))
        tn <= tf ? (tn > 0 ? tn : tf) : NaN
    end
    function hit(p::Plane, ro, rd)
        n = cross(p.u, p.v); t = dot(p.origin - ro, n) / dot(rd, n); q = ro + t * rd - p.origin
        s, w = dot(q, p.u) / dot(p.u, p.u), dot(q, p.v) / dot(p.v, p.v)
        0 <= s <= 1 && 0 <= w <= 1 ? t : NaN
    end
    near(cam) = 1e-6 * (cam.projection === :ortho ? cam.ortho_half : VR.distance(cam))
    expected(shape, cam, w, h) = [(t = hit(shape, ray(cam, w, h, r, c)...); isnan(t) || t <= 0 ? 0.0 : near(cam) / t) for r in 1:h, c in 1:w]
    depth(shape, cam, w, h) = VR.depth_image(View([Surface(shape)]; camera = cam), w, h)

    oblique(proj) = Camera(SVector(2.0, -1.5, 2.5), SVector(0.1, 0.0, -0.1), SVector(0.0, 0, 1), proj, 50.0, 1.3)
    shapes = [Sphere(; center = SVector(0.1, 0.2, -0.1), radius = 0.8), Box(-0.6..0.5, -0.4..0.7, -0.5..0.3),
              Plane(SVector(-0.5, -0.6, 0.1), SVector(1.1, 0.2, 0.0), SVector(0.1, 0.3, 0.9))]
    for shape in shapes, proj in (:perspective, :ortho)
        cam = oblique(proj)
        d, e = depth(shape, cam, 48, 40), expected(shape, cam, 48, 40)
        ok = (d .> 0) .== (e .> 0)
        @test count(!, ok) <= 3                                       # silhouette pixels may round either way
        @test d[ok .& (e .> 0)] ≈ e[ok .& (e .> 0)] rtol = 1e-4
        @test count(e .> 0) > 100
    end

    # axis-aligned ortho view of a Box: rd has exact zero components (the rd == 0 slab path)
    cam = GLContext.topcam(1.0)
    d = depth(Box(-0.5..0.25, -0.25..0.5, -1..0.5), cam, 32, 32)
    inbox = [(xy = GLContext.pixel_xy(1, 32, r, c); -0.5 < xy[1] < 0.25 && -0.25 < xy[2] < 0.5) for r in 1:32, c in 1:32]
    @test d == ifelse.(inbox, Float32(1e-6 / 2.5), 0f0)
end

@testitem "Lines colours and coverage" setup = [GLContext] begin
    import VolumeRendering as VR
    using StaticArrays, ColorTypes, FixedPointNumbers, ColorSchemes, IntervalSets
    top = GLContext.topcam()
    xy(r, c) = GLContext.pixel_xy(1, 64, r, c)
    P(r, c) = SVector(xy(r, c)..., 0.0)                     # world point at the centre of pixel (row, col)
    img(lines...) = render_image(View(collect(Any, lines); camera = top), 64, 64)
    lit(im) = im .!= RGB{N0f8}(0, 0, 0)
    gray(c) = (float(c.r) + float(c.g) + float(c.b)) / 3
    near(a, b) = maximum(abs, a .- b) < 0.01
    cmap = Colormap(ColorSchemes.gray1, 0..1)
    pl = [[P(20, 9), P(20, 57)]]                            # x from -0.734 to 0.766, along row 20
    # every attribute is taken at the true segment point under the pixel: the endpoints sit on pixel centres 9 and 57
    f = (0:48) ./ 48

    # constant
    im = img(Lines(pl; color = RGB(0.2, 0.4, 0.6)))
    @test findall(lit(im)) == CartesianIndex.(20, 9:57)
    @test all(==(RGB{N0f8}(0.2, 0.4, 0.6)), im[20, 9:57])

    # per-vertex colours, interpolated along the segment
    im = img(Lines(pl; color = [[RGB(1.0, 0, 0), RGB(0.0, 0, 1)]]))
    @test near(map(c -> float(c.r), im[20, 9:57]), 1 .- f)
    @test near(map(c -> float(c.b), im[20, 9:57]), f)

    # Mapped(values) through the colormap window
    im = img(Lines(pl; color = Mapped([[0.0, 2.0]], Colormap(ColorSchemes.gray1, 0..2))))
    @test near(gray.(im[20, 9:57]), f)

    # Mapped(field) samples the field at the fragment's world position
    fx = GLSLField("float value(vec3 p) { return 0.5 * p.x + 0.5; } float stepSize(vec3 p) { return 1.0; }")
    im = img(Lines(pl; color = Mapped(fx, cmap)))
    @test near(gray.(im[20, 9:57]), [0.5 * xy(20, c)[1] + 0.5 for c in 9:57])   # the true segment point under each pixel

    # :color field
    yellow_top = GLSLField("vec4 value(vec3 p) { return vec4(1.0, p.y > 0.0 ? 1.0 : 0.0, 0.0, 1.0); } float stepSize(vec3 p) { return 1.0; }";
                           kind = :color)
    im = img(Lines([[P(5, 30), P(60, 30)]]; color = yellow_top))
    @test all(==(RGB{N0f8}(1, 1, 0)), im[5:32, 30]) && all(==(RGB{N0f8}(1, 0, 0)), im[33:60, 30])

    # a zero-length segment draws a one-pixel dot; NaN-free
    @test findall(lit(img(Lines([[P(40, 11), P(40, 11)]])))) == [CartesianIndex(40, 11)]

    # a corner between two segments has no gap: exactly the pixels of both arms
    im = img(Lines([[P(10, 10), P(10, 30), P(50, 30)]]))
    @test findall(lit(im)) == sort([CartesianIndex.(10, 10:30); CartesianIndex.(11:50, 30)])

    # several polylines in one mark; one-point polylines contribute no segment
    im = img(Lines([[P(5, 5), P(5, 9)], [P(60, 60)], [P(30, 2), P(30, 3)]]))
    @test findall(lit(im)) == sort([CartesianIndex.(5, 5:9); CartesianIndex.(30, 2:3)])
    @test !any(lit(img(Lines(Vector{SVector{3,Float64}}[]))))
end

@testitem "Lines eye-plane clip" setup = [GLContext] begin
    import VolumeRendering as VR
    using StaticArrays, ColorTypes
    cam = GLContext.topcam(; projection = :perspective)              # w = 3 - z
    w(p) = 3 - p[3]
    # Float64 reference of the eye-plane clip in `lines_vert`: move the behind endpoint along the segment to w = 1e-4 × the front endpoint's w
    clipped(a, b) = a + (b - a) * ((1 - 1e-4) * w(a) / (w(a) - w(b)))
    img(pl) = render_image(View([Lines([pl])]; camera = cam), 48, 48)
    a = SVector(0.2, 0.013, 0.0)
    for ratio in (1e1, 1e3, 1e4, 1e5, 1e6, 1e7), dir in (SVector(-0.4, 0.3), SVector(0.0, 0.0))
        b = SVector(a[1] + dir[1] * ratio, a[2] + dir[2] * ratio, 3 + 3ratio)   # w(b) = -ratio × w(a)
        ref = img([a, clipped(a, b)])
        @test count(!=(RGB{VR.N0f8}(0, 0, 0)), ref) > 10
        @test img([a, b]) == ref
        @test img([b, a]) == ref
    end

    # payloads follow the drawn segment: per-vertex z equals the field p.z pixel for pixel, receding and eye-plane clipped
    using ColorSchemes, IntervalSets
    cmap = Colormap(ColorSchemes.gray1, -2.0..3.0)
    pz = GLSLField("float value(vec3 p) { return p.z; } float stepSize(vec3 p) { return 1.0; }")
    shade(pl, color) = render_image(View([Lines([pl]; color)]; camera = cam), 64, 64)
    for (p, q) in ((SVector(-0.6, 0.01, -2.0), SVector(0.7, 0.01, 2.5)),     # receding, both in front
                   (SVector(-0.5, 0.01, 0.0), SVector(4.5, 0.01, 8.0))),     # q behind the eye (z > 3)
        pl in ([p, q], [q, p])
        ref = shade(pl, Mapped(pz, cmap))
        @test count(!=(RGB{VR.N0f8}(0, 0, 0)), ref) > 20
        @test shade(pl, Mapped([getindex.(pl, 3)], cmap)) == ref
    end
end

@testitem "uniform upload: bool, int, ivec, mat4" setup = [GLContext] begin
    import VolumeRendering as VR
    using StaticArrays, ColorTypes, FixedPointNumbers
    M = SMatrix{4,4}(reshape(0.0:15, 4, 4) ./ 16)
    f = GLSLField("""
        vec4 value(vec3 p) { return vec4(flag && k == 3 ? 1.0 : 0.0, float(n.z - n.x) * 0.2, M[1][2], 1.0); }
        float stepSize(vec3 p) { return 1.0; }
        """; kind = :color, uniforms = (; flag = true, k = 3, n = SVector(1, 2, 3), M))
    img = render_image(View([Surface(Sphere(radius = 0.5); color = f)]; camera = GLContext.topcam()), 8, 8)
    @test img[4, 4] == RGB{N0f8}(1, 0.4, M[3, 2])                    # GLSL M[col][row]: column-major upload, no transpose
end

@testitem "(lon, lat) GridField on a Surface(Sphere)" setup = [GLContext] begin
    import VolumeRendering as VR
    using StaticArrays, ColorTypes, ColorSchemes, IntervalSets, AxisKeys
    lat = range(-5π / 12; step = π / 6, length = 6)               # cell centres -75°…75°
    lon = range(0.0; step = π / 6, length = 12)                    # 0…330°: wraps between the last and first cells
    side = Camera(SVector(5.0, 0, 0), SVector(0.0, 0, 0), SVector(0.0, 0, 1), :ortho, 45.0, 1.2)   # screen right = +y, up = +z
    shell(A; interpolation) = View([Surface(Sphere(radius = 1); color = Mapped(GridField(A; interpolation), Colormap(ColorSchemes.gray1, 0..1)))];
                                    camera = side)

    # one bright cell at lat = 45°, lon = 30° lights the upper-right quadrant only
    A = KeyedArray(zeros(6, 12); lat, lon)
    A[5, 2] = 1
    im = render_image(shell(A; interpolation = :nearest), 48, 48)
    bright = findall(c -> c.r > 0.5, im)
    @test length(bright) > 10 && all(i -> i[1] < 24 && i[2] > 24, bright)

    # smooth data bright across the lon seam (lon = 0 faces the camera): no dark column
    B = KeyedArray([(1 + cos(lo)) / 2 for la in lat, lo in lon]; lat, lon)
    im = render_image(shell(B; interpolation = :linear), 48, 48)
    mid = map(c -> float(c.r), im[24, 12:37])
    @test minimum(mid) > 0.8
    @test maximum(abs, diff(mid)) < 0.03
end

@testitem "user-defined field type and Shape" setup = [GLContext] begin
    import VolumeRendering as VR
    using StaticArrays, ColorTypes, FixedPointNumbers, ColorSchemes, IntervalSets

    # the PlasmaScope shape of field: 3-D :repeat texture, 2-D LUT, mat3 rotation uniform
    mutable struct Striped
        arr::Array{Float32,3}
        lut::Matrix{Float32}
        R::SMatrix{3,3,Float64,9}
    end
    VR.value_kind(::Striped) = :scalar
    VR.textures(f::Striped) = (; rho = Texture(f.arr; wrap = (:repeat, :clamp, :clamp), filter = :nearest), lut = Texture(f.lut; filter = :nearest))
    VR.uniforms(f::Striped) = (; Rt = f.R')
    VR.field_glsl(::Striped) = """
        float value(vec3 p) {
            vec3 q = Rt * p;
            return texture(rho, vec3(q.x, 0.5, 0.5)).r + texture(lut, vec2(0.5, 0.5 * q.y + 0.5)).r;
        }
        float stepSize(vec3 p) { return 0.1; }
        """
    rot(a) = SMatrix{3,3}(cos(a), sin(a), 0, -sin(a), cos(a), 0, 0, 0, 1)
    f = Striped(reshape(Float32[0, 0.5], 2, 1, 1), reshape(Float32[0, 0.25], 1, 2), rot(0))
    view = View([Surface(Plane(:z, 0, Box(-1..1, -1..1, -1..1)); color = Mapped(f, Colormap(ColorSchemes.gray1, 0..1)))];
                camera = GLContext.topcam())
    # x-stripes of period 1 (wrap :repeat also for x < 0) plus 0.25 for y > 0, in the rotated frame q = Rᵀ p
    function expected(R)
        map(CartesianIndices((64, 64))) do I
            q = R' * SVector(GLContext.pixel_xy(1, 64, Tuple(I)...)..., 0.0)
            (mod(q[1], 1) >= 0.5 ? 0.5 : 0.0) + (q[2] > 0 ? 0.25 : 0.0)
        end
    end
    gray(im) = map(c -> float(c.r), im)
    @test maximum(abs, gray(render_image(view, 64, 64)) - expected(rot(0))) < 0.01
    f.R = rot(π / 2)
    @test maximum(abs, gray(render_image(view, 64, 64)) - expected(rot(π / 2))) < 0.01

    # a user-defined Shape: an ellipsoid
    struct Ellipsoid
        center::SVector{3,Float64}
        axes::SVector{3,Float64}
    end
    VR.shape_glsl(::Ellipsoid) = """
        vec2 vr_hit(vec3 ro, vec3 rd) {
            vec3 o = (ro - ell_center) / ell_axes, d = rd / ell_axes;
            float a = dot(d, d), b = dot(o, d), c = dot(o, o) - 1.0, disc = b * b - a * c;
            if (disc < 0.0) return vec2(1e30, -1e30);
            float s = sqrt(disc); return vec2((-b - s) / a, (-b + s) / a);
        }
        """
    VR.uniforms(e::Ellipsoid) = (; ell_center = e.center, ell_axes = e.axes)
    VR.bounds(e::Ellipsoid) = (; e.center, radius = maximum(e.axes))
    VR.aabb(e::Ellipsoid) = (; lo = e.center - e.axes, hi = e.center + e.axes)
    e = Ellipsoid(SVector(0.1, -0.2, 0.0), SVector(0.7, 0.35, 0.5))
    im = render_image(View([Surface(e; color = RGB(1.0, 0, 0))]; camera = GLContext.topcam()), 64, 64)
    inside = [(xy = GLContext.pixel_xy(1, 64, r, c); ((xy[1] - 0.1) / 0.7)^2 + ((xy[2] + 0.2) / 0.35)^2 < 1) for r in 1:64, c in 1:64]
    @test count((im .== RGB{N0f8}(1, 0, 0)) .!= inside) <= 2
    @test count(inside) > 500

    # as Volume sources: the field's MIP from the top is its value v (z-independent), coloured gray v with alpha v → v²;
    # the ellipsoid as a region bounds a constant field
    f.R = rot(0)
    fvol = View([Volume(Mapped(f, Colormap(ColorSchemes.gray1, 0..1)); region = Box(-1..1, -1..1, -0.5..0.5), mode = :mip)];
                camera = GLContext.topcam())
    @test maximum(abs, gray(render_image(fvol, 64, 64)) - expected(rot(0)) .^ 2) < 0.01
    f.R = rot(π / 2)
    @test maximum(abs, gray(render_image(fvol, 64, 64)) - expected(rot(π / 2)) .^ 2) < 0.01
    constant = GLSLField("float value(vec3 p) { return 1.0; } float stepSize(vec3 p) { return 0.05; }")
    im = render_image(View([Volume(Mapped(constant, Colormap(ColorSchemes.gray1, 0..1)); region = e, mode = :mip)]; camera = GLContext.topcam()), 64, 64)
    @test count((im .== RGB{N0f8}(1, 1, 1)) .!= inside) <= 2
end

@testitem "Volume" setup = [GLContext] begin
    import VolumeRendering as VR
    using StaticArrays, ColorTypes, FixedPointNumbers, ColorSchemes, IntervalSets, AxisKeys
    top = GLContext.topcam()
    gray(c) = float(c.r)
    lit(im) = count(c -> max(c.r, c.g, c.b) > 4 / 255, im) / length(im)
    xs = range(-1, 1; length = 24)
    grid(f) = KeyedArray([f(x, y, z) for x in xs, y in xs, z in xs]; x = xs, y = xs, z = xs)
    gauss(x0 = 0.0) = grid((x, y, z) -> exp(-((x - x0)^2 + y^2 + z^2) / 0.1))
    mip(f, cmap = Colormap(ColorSchemes.gray1, 0..1); kw...) = Volume(Mapped(f, cmap); region = Box(-1..1, -1..1, -1..1), mode = :mip, kw...)
    region = Box(-0.5..0.5, -0.5..0.5, -0.5..0.5)

    # Gaussian volume (DVR) lit: a centred blob, brightest in the middle, dark corners
    g = GridField(gauss())
    cmap = Colormap(ColorSchemes.viridis, value_range(g))
    vol = Volume(Mapped(g, cmap); region = Box(g))
    view = View([vol]; camera = Camera(; bounds(vol.region)...))
    im = render_image(view, 64, 64)
    @test 0.03 < lit(im) < 0.5
    @test im[1, 1] == im[64, 64] == RGB{N0f8}(0, 0, 0) && im[32, 32] != RGB{N0f8}(0, 0, 0)

    # cmap window / log change re-renders
    cmap.window = 0.5..1.0
    im2 = render_image(view, 64, 64)
    @test im2 != im && lit(im2) < lit(im)
    cmap.log = true
    @test render_image(view, 64, 64) ∉ (im, im2)
    cmap.log = false; cmap.window = value_range(g)
    @test render_image(view, 64, 64) == im

    # an opaque sphere in front of the volume lowers its lit fraction (black: it adds no lit pixel itself)
    front = Surface(Sphere(; center = (view.camera.eye + view.camera.lookat) / 2, radius = 0.3); color = RGB(0.0, 0, 0))
    view.marks = [vol, front]
    @test lit(render_image(view, 64, 64)) < 0.7 * lit(im)
    view.marks = [vol]

    # nearest vs linear differ (and the toggle re-renders)
    g.interpolation = :nearest
    @test render_image(view, 64, 64) != im
    g.interpolation = :linear

    # field swap with identical GLSL re-uploads: the image changes
    vol.color.source = GridField(gauss(0.4))
    @test field_glsl(vol.color.source) == field_glsl(g)
    @test render_image(view, 64, 64) != im
    @test !haskey(view.cache.entries, VR.Identity(VR.cache_key(g.data)))   # the old texture is freed (GL may reuse its id)
    vol.color.source = g

    # custom axis type: pixel-identical to the range
    struct MyAxis <: AbstractVector{Float64}
        r::StepRangeLen{Float64}
    end
    Base.size(a::MyAxis) = size(a.r)
    Base.getindex(a::MyAxis, i::Int) = a.r[i]
    VR.axis_index_glsl(k::MyAxis, coordvar::AbstractString) = axis_index_glsl(k.r, coordvar)
    A = gauss()
    custom = GridField(KeyedArray(parent(A); x = MyAxis(xs), y = MyAxis(xs), z = MyAxis(xs)))
    @test render_image(View([Volume(Mapped(custom, cmap); region = Box(custom))]; camera = view.camera), 64, 64) == im

    # +x lands right (ortho MIP), also with a coarser march of the same aspect
    blob = GridField(gauss(0.5))
    for max_volume_px in (2048, 24)
        img = render_image(View([mip(blob)]; camera = top, max_volume_px), 64, 48)
        bright = findall(c -> gray(c) > 0.5, img)
        @test !isempty(bright) && all(i -> i[2] > 32, bright)
        # centroid: x = 0.5 is 44 px from the left (ortho half-width 4/3), y = 0 is 24 px from the top
        @test sum(i -> i[2], bright) / length(bright) ≈ 44.5 atol = 1.5
        @test sum(i -> i[1], bright) / length(bright) ≈ 24.5 atol = 1
    end

    # MIP of a negative-valued field; average is the dt-weighted mean
    neg = GLSLField("float value(vec3 p) { return p.x > 0.0 ? -0.5 : -1.0; } float stepSize(vec3 p) { return 0.1; }")
    im = render_image(View([mip(neg, Colormap(ColorSchemes.gray1, -1..0); region)]; camera = top), 32, 32)
    @test all(c -> isapprox(gray(c), 0.25; atol = 0.01), im[9:24, 17:24]) && all(==(RGB{N0f8}(0, 0, 0)), im[9:24, 9:16])
    zf = GLSLField("float value(vec3 p) { return p.z; } float stepSize(vec3 p) { return 0.02; }")
    zvol(mode) = View([Volume(Mapped(zf, Colormap(ColorSchemes.gray1, -1..1)); region = Box(-0.5..0.5, -0.5..0.5, -1..1), mode)]; camera = top)
    @test all(c -> isapprox(gray(c), 1; atol = 0.01), render_image(zvol(:mip), 32, 32)[9:24, 9:24])
    @test all(c -> isapprox(gray(c), 0.25; atol = 0.01), render_image(zvol(:average), 32, 32)[9:24, 9:24])   # mean 0 → u = 0.5, alpha 0.5

    # RGBA colour field: red / green halves (DVR, σ in alpha)
    C = KeyedArray([x < 0 ? RGBA{Float32}(1, 0, 0, 50) : RGBA{Float32}(0, 1, 0, 50) for x in -0.75:0.5:0.75, y in 0:1, z in 0:1];
                   x = -0.75:0.5:0.75, y = -0.5:1.0:0.5, z = -0.5:1.0:0.5)
    rg = GridField(C; interpolation = :nearest)
    im = render_image(View([Volume(rg; region = Box(rg))]; camera = top), 32, 32)
    @test all(==(RGB{N0f8}(1, 0, 0)), im[10:23, 8:15]) && all(==(RGB{N0f8}(0, 1, 0)), im[10:23, 18:25])
    @test im[2, 2] == RGB{N0f8}(0, 0, 0)

    # GLSLField with a 1-D LUT texture and uniforms
    lutdata = Float32[0, 1]
    lutf(gain) = GLSLField("float value(vec3 p) { return gain * texture(lut, 0.5 * p.x + 0.5).r; } float stepSize(vec3 p) { return 0.1; }";
                           textures = (; lut = Texture(lutdata; filter = :nearest)), uniforms = (; gain))
    lv = mip(lutf(1.0); region)
    v = View([lv]; camera = top)
    im = render_image(v, 32, 32)
    @test all(c -> isapprox(gray(c), 1; atol = 0.01), im[9:24, 17:24]) && all(==(RGB{N0f8}(0, 0, 0)), im[9:24, 9:16])
    lv.color.source = lutf(0.5)
    @test all(c -> isapprox(gray(c), 0.25; atol = 0.01), render_image(v, 32, 32)[9:24, 17:24])

    # volume.region change clips the volume
    full = render_image(View([mip(GridField(gauss()))]; camera = top), 32, 32)
    half = mip(GridField(gauss()))
    half.region = Box(-1..0, -1..1, -1..1)
    clipped = render_image(View([half]; camera = top), 32, 32)
    @test clipped[:, 1:16] == full[:, 1:16] && all(==(RGB{N0f8}(0, 0, 0)), clipped[:, 17:32])

    # a Mapped shared by a Volume and a Plane Surface: replacing its source updates both
    c1 = GLSLField("float value(vec3 p) { return 0.25; } float stepSize(vec3 p) { return 0.1; }")
    c2 = GLSLField("float value(vec3 p) { return 0.75; } float stepSize(vec3 p) { return 0.1; }")
    colored = Mapped(c1, Colormap(ColorSchemes.gray1, 0..1))
    shared = View([Volume(colored; region = Box(-1..0, -1..1, -0.5..0.5), mode = :mip),
                   Surface(Plane(:z, 0, Box(0..1, -1..1, -1..1)); color = colored)]; camera = top)
    im = render_image(shared, 32, 32)
    @test gray.(im[16, [8, 24]]) ≈ [0.25^2, 0.25] atol = 0.01
    colored.source = c2
    im = render_image(shared, 32, 32)
    @test gray.(im[16, [8, 24]]) ≈ [0.75^2, 0.75] atol = 0.01
end

@testitem "Volume: march accuracy, LUTs, clipping" setup = [GLContext] begin
    import ModernGL as GL, VolumeRendering as VR
    using StaticArrays, ColorTypes, FixedPointNumbers, ColorSchemes, IntervalSets, AxisKeys
    top = GLContext.topcam()
    gray(c) = float(c.r)
    constant(v; step = 0.1) = GLSLField("float value(vec3 p) { return k; } float stepSize(vec3 p) { return $step; }"; uniforms = (; k = v))
    flat = [SVector(0.0, 0.5), SVector(1.0, 0.5)]           # σ = 0.5 for every u

    # DVR law over a slab of thickness 1: alpha = 1 - exp(-σ·density·L), independent of step_scale; white colour, black background
    slab(step_scale) = View([Volume(Mapped(constant(1.0), Colormap(ColorSchemes.gray1, 0..1)); region = Box(-0.5..0.5, -0.5..0.5, -0.5..0.5),
                                    extinction = flat, density = 2.0, step_scale)]; camera = top)
    for step_scale in (0.5, 2.5)                              # 20 and 4 samples per ray
        @test all(c -> isapprox(gray(c), 1 - exp(-1); atol = 0.01), render_image(slab(step_scale), 32, 32)[9:24, 9:24])
    end

    # eye inside the region: the march starts at the eye; path length 1 to the near face at the centre, longer off-axis
    inside = Camera(SVector(0.0, 0, 0), SVector(0.0, 0, -1), SVector(0.0, 1, 0), :perspective, 60.0, 1.0)
    im = render_image(View([Volume(Mapped(constant(1.0), Colormap(ColorSchemes.gray1, 0..1)); region = Box(-1..1, -1..1, -1..1),
                                   extinction = flat, density = 2.0)]; camera = inside), 32, 32)
    @test gray(im[16, 16]) ≈ 1 - exp(-1) atol = 0.01
    @test gray(im[1, 1]) > gray(im[16, 16]) + 0.05

    # max_samples coarsens a long march, never cuts it short: MIP and average of p.z along a 10-deep box reach its far end
    # and its mean whatever the step — tiny, zero, negative or NaN (a NaN uniform: GLSL has no NaN literal)
    depth = Box(-1..1, -1..1, 0..10)
    zview(step, max_samples, mode) = View([Volume(Mapped(GLSLField("float value(vec3 p) { return p.z; } float stepSize(vec3 p) { return k; }";
                                                                   uniforms = (; k = step)), Colormap(ColorSchemes.gray1, 0..10));
                                                  region = depth, mode, step_scale = 1.0, max_samples, extinction = [SVector(0.0, 1.0), SVector(1.0, 1.0)])];
                                          camera = GLContext.topcam(; eye = -1.0), refine_frames = 1)
    for step in (0.01, 1e-4, 0.0, -1.0, NaN), max_samples in (4096, 100)
        @test all(c -> isapprox(10gray(c), 10; atol = 10 / max_samples + 0.05), render_image(zview(step, max_samples, :mip), 8, 8))
        @test all(c -> isapprox(10gray(c), 5; atol = 10 / max_samples + 0.05), render_image(zview(step, max_samples, :average), 8, 8))
    end
    # and a coarsened DVR march keeps the exact opacity law
    @test all(c -> isapprox(gray(c), 1 - exp(-1); atol = 0.01),
              render_image(View([Volume(Mapped(constant(1.0; step = 1e-6), Colormap(ColorSchemes.gray1, 0..1)); region = Box(-0.5..0.5, -0.5..0.5, -0.5..0.5),
                                        extinction = flat, density = 2.0, max_samples = 64)]; camera = top), 32, 32)[9:24, 9:24])

    # (lon, lat) region through the origin, where the step shrinks to 0: alpha = 1 - exp(-σ·chord) on every ray
    ll = GridField(KeyedArray(ones(5, 8); lat = range(-π/2, π/2, 5), lon = range(0, 2π * 7/8, 8)))
    llimg = render_image(View([Volume(Mapped(ll, Colormap(ColorSchemes.gray1, 0..1)); region = Sphere(radius = 1.0), density = 0.5,
                                    extinction = [SVector(0.0, 1.0), SVector(1.0, 1.0)], step_scale = 0.1)]; camera = top), 32, 32)
    chord_alpha(row, col) = let (x, y) = GLContext.pixel_xy(1.0, 32, row, col)
        x^2 + y^2 < 0.95 ? 1 - exp(-sqrt(1 - x^2 - y^2)) : nothing
    end
    @test all(I -> (a = chord_alpha(Tuple(I)...); isnothing(a) || isapprox(gray(llimg[I]), a; atol = 0.01)), CartesianIndices(llimg))

    # a ray that hits the region but takes no sample (step ≫ slab) is a miss: the white background shows
    thin = Box(-0.5..0.5, -0.5..0.5, -0.01..0.01)
    for mode in (:mip, :average)
        vol = Volume(Mapped(constant(0.5; step = 1.0), Colormap(ColorSchemes.gray1, 0..1)); region = thin, mode, step_scale = 10.0,
                     extinction = [SVector(0.0, 1.0), SVector(1.0, 1.0)])
        img = render_image(View([vol]; camera = top, background = RGB(1.0, 1, 1)), 32, 32)
        @test sum(gray, img) / length(img) > 0.99
    end

    # 1-D LUTs: texel i holds f(i/255) and u = i/255 reads it exactly (alternating colours / σ expose any half-texel shift)
    stripes = ColorScheme([isodd(i) ? RGB(1.0, 1, 1) : RGB(0.0, 0, 0) for i in 0:255])
    plane = Plane(:z, 0, Box(-1..1, -1..1, -1..1))
    for i in (0, 1, 63, 64, 191, 254, 255)
        img = render_image(View([Surface(plane; color = Mapped(constant(i / 255), Colormap(stripes, 0..1)))]; camera = top), 8, 8)
        @test all(==(isodd(i) ? RGB{N0f8}(1, 1, 1) : RGB{N0f8}(0, 0, 0)), img)
        ext = [SVector(j / 255, isodd(j) ? 1.0 : 0.0) for j in 0:255]
        img = render_image(View([Volume(Mapped(constant(i / 255), Colormap(ColorSchemes.gray1, 0..1)); region = Box(-1..1, -1..1, -0.5..0.5),
                                       mode = :mip, extinction = ext)]; camera = top), 8, 8)
        @test all(c -> isapprox(gray(c), isodd(i) ? i / 255 : 0.0; atol = 0.5 / 255), img)
    end

    # the extinction LUT is cached by value: an equal new vector reuses the texture, a changed one replaces it
    vol = Volume(Mapped(constant(0.5), Colormap(ColorSchemes.gray1, 0..1)); region = Box(-0.5..0.5, -0.5..0.5, -0.5..0.5), extinction = flat)
    view = View([vol]; camera = top)
    luts(view) = filter(x -> x isa VR.GLTexture{Float32,1}, collect(values(view.cache.entries)))
    im = render_image(view, 16, 16)
    lut = only(luts(view))
    vol.extinction = copy(flat); vol.density = 1.5            # density forces a frame
    render_image(view, 16, 16)
    @test only(luts(view)) === lut && view.cache.entries[(:extinction, flat)] === lut
    vol.extinction = [SVector(0.0, 0.25), SVector(1.0, 0.25)]
    im2 = render_image(view, 16, 16)
    @test !haskey(view.cache.entries, (:extinction, flat)) && only(luts(view)) === view.cache.entries[(:extinction, vol.extinction)]
    vol.density = 1.0
    @test render_image(view, 16, 16) != im2                   # density changes the image
    @test render_image(view, 16, 16) != im

    # clip accuracy: MIP of -p.z behind an opaque plane at z = 0 reaches exactly 0 → u = 0.5 (window -1..1, σ ≥ 1 → alpha 1).
    # A coarse perspective march clips at the nearest depth of each cell's footprint: conservative, so u ≤ 0.5, exact on axis
    negz = GLSLField("float value(vec3 p) { return -p.z; } float stepSize(vec3 p) { return 0.01; }")
    for projection in (:ortho, :perspective), max_volume_px in (2048, 12)
        v = View([Surface(Plane(:z, 0, Box(-5..5, -5..5, -1..1)); color = RGB(0.0, 0, 0)),
                  Volume(Mapped(negz, Colormap(ColorSchemes.gray1, -1..1)); region = Box(-5..5, -5..5, -1..1), mode = :mip,
                         extinction = [SVector(0.0, 1.0), SVector(1.0, 1.0)])];
                 camera = GLContext.topcam(; projection), max_volume_px)
        img = gray.(render_image(v, 40, 32))
        if projection === :perspective && max_volume_px == 12
            @test maximum(img) <= 0.5 && minimum(img) > 0.35
            @test img[16, 20] ≈ 0.5 atol = 0.02
        else
            @test all(≈(0.5; atol = 0.01), img)
        end
    end
    VR.glcheck()
end

@testitem "translucent surfaces" setup = [GLContext] begin
    using StaticArrays, ColorTypes, FixedPointNumbers, IntervalSets
    top = GLContext.topcam()
    near(c, ref) = all(isapprox.((float(c.r), float(c.g), float(c.b)), ref; atol = 0.01))
    sq = Box(-0.5..0.5, -0.5..0.5, -1..1)

    # half-opacity sphere over black: the far and the near layer each blend at 0.5 → 1 - 0.5² of the colour
    s = Surface(Sphere(radius = 0.5); color = RGB(1.0, 0.5, 0), opacity = 0.5)
    @test near(render_image(View([s]; camera = top), 32, 32)[16, 16], (0.75, 0.375, 0))
    # a translucent plane is blended once
    p = Surface(Plane(:z, 0, sq); color = RGB(1.0, 1, 1), opacity = 0.5)
    im = render_image(View([p]; camera = top), 32, 32)
    @test all(c -> near(c, (0.5, 0.5, 0.5)), im[10:23, 10:23]) && im[2, 2] == RGB{N0f8}(0, 0, 0)

    # two translucent surfaces composite in list order, regardless of depth
    red = Surface(Plane(:z, 0.5, sq); color = RGB(1.0, 0, 0), opacity = 0.5)
    green = Surface(Plane(:z, -0.5, sq); color = RGB(0.0, 1, 0), opacity = 0.5)
    @test near(render_image(View([red, green]; camera = top), 32, 32)[16, 16], (0.25, 0.5, 0))
    @test near(render_image(View([green, red]; camera = top), 32, 32)[16, 16], (0.5, 0.25, 0))

    # clipped by the opaque depth only: behind an opaque plane it is hidden, over it it blends
    opaque = Surface(Plane(:z, 0, sq); color = RGB(0.0, 0, 1))
    @test render_image(View([red, opaque]; camera = top), 32, 32)[16, 16] == render_image(View([opaque, red]; camera = top), 32, 32)[16, 16]
    @test near(render_image(View([opaque, red]; camera = top), 32, 32)[16, 16], (0.5, 0, 0.5))
    @test render_image(View([opaque, green]; camera = top), 32, 32)[16, 16] == RGB{N0f8}(0, 0, 1)
end

@testitem "refinement" setup = [GLContext] begin
    import ModernGL as GL, VolumeRendering as VR
    using StaticArrays, ColorTypes, FixedPointNumbers, ColorSchemes, IntervalSets
    f = GLSLField("float value(vec3 p) { return exp(-dot(p, p) * 4.0); } float stepSize(vec3 p) { return 0.05; }")
    vol = Volume(Mapped(f, Colormap(ColorSchemes.viridis, 0..1)); region = Box(-1..1, -1..1, -1..1))
    lines = Lines([[SVector(-0.9, 0.3, 0), SVector(0.9, 0.3, 0)]])
    view = View([vol, lines]; camera = Camera(; radius = 1), refine_frames = 4)

    # unchanged inputs: refine_frames frames into accum_fb, then the same texture without touching GL
    texs = map(_ -> render!(view, 32, 24), 1:3)
    @test view.nsamp == 3 && all(==(view.cache.accum_fb.color.id), texs)
    @test render!(view, 32, 24) == texs[1] && view.nsamp == 4
    final = render_image(view, 32, 24)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, view.cache.accum_fb.fbo); GL.glClearColor(0, 1, 0, 1); GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    @test render!(view, 32, 24) == texs[1] && view.nsamp == 4
    @test all(==(RGB{N0f8}(0, 1, 0)), render_image(view, 32, 24))   # converged: nothing re-rendered
    # the average differs from a single jittered frame
    view.refine_frames = 1
    @test render!(view, 32, 24) == view.cache.scene_fb.color.id && view.nsamp == 1
    single = render_image(view, 32, 24)
    @test single != final
    view.refine_frames = 4

    # camera change resets
    render_image(view, 32, 24)
    orbit!(view.camera, 0.1, 0)
    render!(view, 32, 24)
    @test view.nsamp == 1
    @test render_image(view, 32, 24) != final && view.nsamp == 4

    # no volume → one frame, the scene texture displayed
    view.marks = [lines]
    @test render!(view, 32, 24) == view.cache.scene_fb.color.id && view.nsamp == 1
    @test render!(view, 32, 24) == view.cache.scene_fb.color.id && view.nsamp == 1

    # a Volume removed then re-added: a byte-identical overlay-only image in between
    view = View([vol, lines]; camera = Camera(; radius = 1), refine_frames = 4)
    withvol = render_image(view, 32, 24)
    deleteat!(view.marks, 1)
    @test render_image(view, 32, 24) == render_image(View([lines]; camera = Camera(; radius = 1)), 32, 24)
    @test !any(x -> x isa VR.GLTexture{<:Any,1}, values(view.cache.entries))   # colormap and extinction LUTs freed
    pushfirst!(view.marks, vol)
    @test render_image(view, 32, 24) == withvol
    VR.glcheck()
end

@testitem "resource cache through render!" setup = [GLContext] begin
    import ModernGL as GL, VolumeRendering as VR
    using StaticArrays, ColorTypes, ColorSchemes, IntervalSets, AxisKeys
    entries(view, T) = filter(x -> x isa T, collect(values(view.cache.entries)))
    top = GLContext.topcam()
    pl = [[SVector(-0.8, 0.01, 0), SVector(0.8, 0.01, 0)]]

    # a Mapped window change re-renders without re-uploading the vertex buffer
    cmap = Colormap(ColorSchemes.gray1, 0..1)
    lines = Lines(pl; color = Mapped([[0.0, 1.0]], cmap))
    view = View([lines]; camera = top)
    im1 = render_image(view, 32, 32)
    buf = only(entries(view, VR.InstanceBuffer))
    cmap.window = 0..2
    im2 = render_image(view, 32, 32)
    @test im1 != im2
    @test only(entries(view, VR.InstanceBuffer)) === buf && GL.glIsBuffer(buf.vbo) == GL.GL_TRUE
    cmap.log = true; cmap.window = 0.1..1
    @test render_image(view, 32, 32) ∉ (im1, im2)
    @test only(entries(view, VR.InstanceBuffer)) === buf

    # a removed mark's resources are freed on the next rendered frame and re-created when re-added
    g = GridField(KeyedArray(rand(4, 8); lat = range(-1.2, 1.2; length = 4), lon = range(0, 2π; length = 9)[1:8]))
    surf = Surface(Sphere(radius = 0.5); color = Mapped(g, cmap))
    view = View([surf, lines]; camera = top)
    full = render_image(view, 32, 32)
    tex, buf = only(entries(view, VR.GLTexture{Float32,2})), only(entries(view, VR.InstanceBuffer))
    lut = only(entries(view, VR.GLTexture{RGBA{Float32},1}))
    bound(unit) = (GL.glActiveTexture(GL.GL_TEXTURE0 + unit); r = Ref{GL.GLint}(-1); GL.glGetIntegerv(GL.GL_SAMPLER_BINDING, r); r[])
    @test bound(0) == bound(1) == 0                                   # sampler objects unbound after the frame
    @test length(entries(view, VR.Program)) == 2 && length(entries(view, VR.GLSampler)) == 2   # (clamp, repeat) grid, (clamp,) colormap
    deleteat!(view.marks, 2)
    render!(view, 32, 32)
    @test isempty(entries(view, VR.InstanceBuffer)) && GL.glIsBuffer(buf.vbo) == GL.GL_FALSE
    @test only(entries(view, VR.GLTexture{Float32,2})) === tex && length(entries(view, VR.Program)) == 1
    push!(view.marks, lines)
    @test render_image(view, 32, 32) == full
    @test length(entries(view, VR.InstanceBuffer)) == 1 && only(entries(view, VR.GLTexture{RGBA{Float32},1})) === lut
    view.marks = [lines]                                              # texture, colormap LUT and samplers freed
    render!(view, 32, 32)
    @test GL.glIsTexture(tex.id) == GL.GL_FALSE && GL.glIsTexture(lut.id) == GL.GL_TRUE    # the lines still use the colormap
    @test length(entries(view, VR.GLSampler)) == 1

    # program count = number of distinct source texts
    view = View([Surface(Sphere(radius = 0.2)), Surface(Sphere(; center = SVector(0.5, 0, 0), radius = 0.1); color = RGB(0.0, 1, 0)),
                 Surface(Box(-0.9..(-0.7), -0.9..(-0.7), -0.1..0.1)), Lines(pl), Lines([[SVector(0.0, -0.5, 0), SVector(0.5, -0.5, 0)]]),
                 Lines(pl; color = [[RGB(1.0, 0, 0), RGB(0.0, 0, 1)]])]; camera = top)
    render!(view, 32, 32)
    @test length(entries(view, VR.Program)) == 4                      # sphere, box, constant lines, per-vertex lines
    @test length(entries(view, VR.InstanceBuffer)) == 3
    free!(view)
    @test isnothing(view.cache)
    VR.glcheck()
end

@testitem "multi-view isolation" setup = [GLContext] begin
    import VolumeRendering as VR
    using StaticArrays, ColorTypes, ColorSchemes, IntervalSets, AxisKeys
    g = GridField(KeyedArray(rand(3, 6); lat = range(-1, 1; length = 3), lon = range(0, 5; length = 6)))
    scene_a() = View([Surface(Sphere(radius = 0.6); color = Mapped(g, Colormap(ColorSchemes.viridis, 0..1))),
                      Lines([[SVector(-1.0, 0.3, 0.7), SVector(1.0, -0.2, 0.7)]]; width_px = 3)]; camera = Camera(; radius = 1))
    scene_b() = View([Surface(Box(-0.5..0.5, -0.5..0.5, -0.5..0.5); color = RGB(0.1, 0.8, 0.3))]; camera = GLContext.topcam(),
                     background = RGB(0.3, 0.3, 0.3))
    ia, ib = render_image(scene_a(), 40, 30), render_image(scene_b(), 24, 36)
    a, b = scene_a(), scene_b()
    for _ in 1:3
        @test render_image(a, 40, 30) == ia
        @test render_image(b, 24, 36) == ib
        a.camera.fov += 1; render!(a, 40, 30); a.camera.fov -= 1     # force re-renders, interleaved
        b.background = RGB(0.0, 0.0, 0.0); render!(b, 24, 36); b.background = RGB(0.3, 0.3, 0.3)
    end
    @test a.cache !== b.cache
    free!(a)
    @test render_image(b, 24, 36) == ib

    # with Volumes, interleaving partial refinement of two views gives each its standalone image
    f = GLSLField("float value(vec3 p) { return exp(-dot(p, p) * 4.0); } float stepSize(vec3 p) { return 0.05; }")
    vscene(cmap, cam) = View([Volume(Mapped(f, Colormap(cmap, 0..1)); region = Box(-1..1, -1..1, -1..1)), Lines([[SVector(-0.9, 0.3, 0), SVector(0.9, 0.3, 0)]])];
                             camera = cam, refine_frames = 4)
    va() = vscene(ColorSchemes.viridis, Camera(; radius = 1)); vb() = vscene(ColorSchemes.inferno, GLContext.topcam())
    ia, ib = render_image(va(), 40, 30), render_image(vb(), 24, 36)
    a, b = va(), vb()
    render!(a, 40, 30); render!(b, 24, 36); render!(b, 24, 36); render!(a, 40, 30); render!(b, 24, 36)
    @test (a.nsamp, b.nsamp) == (2, 3)
    @test render_image(a, 40, 30) == ia
    @test render_image(b, 24, 36) == ib
end

@testitem "helpers: graticule, on_sphere, box_edges, axis_lines" begin
    import VolumeRendering as VR
    using StaticArrays, LinearAlgebra, IntervalSets, ColorTypes, Unitful
    midpoints(pl) = [(a + b) / 2 for (a, b) in zip(pl, @view pl[2:end])]
    x, y = SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0)
    unit = Sphere(radius = 1.0)

    @test VR.parallel_lats(3) ≈ [-π/4, 0, π/4]
    @test isempty(VR.parallel_lats(0))
    @test VR.meridian_lons(6) ≈ [0, π/3, 2π/3, π, 4π/3, 5π/3]
    @test isempty(VR.meridian_lons(0))
    @test VR.parallel_lats([0.1, 0.2]) == [0.1, 0.2]                  # explicit angles pass through

    # defaults: 3 rings of 65 points, 6 half-meridians of 33
    @test length.(graticule(unit)) == [65, 65, 65, 33, 33, 33, 33, 33, 33]
    # equator only, 8 segments: one closed ring of 9 points on z = 0, out to 1 / cos(π/8)
    eq = only(graticule(unit; parallels = [0.0], meridians = 0, segments = 8))
    @test length(eq) == 9
    @test eq[1] ≈ eq[end] ≈ SVector(1.0823922002923940, 0, 0)
    @test getindex.(eq, 3) == zeros(9)
    @test norm.(midpoints(eq)) ≈ fill(1.0, 8)
    # a meridian is half a circle, so it gets half the segments: pole to pole in 9 points, 22.5° steps
    mer = only(graticule(unit; parallels = 0, meridians = [0.0], segments = 16))
    @test length(mer) == 9
    @test mer[1] ≈ SVector(0, 0, -1.0195911582083184) && mer[end] ≈ SVector(0, 0, 1.0195911582083184)
    # angles only reach sin/cos: Unitful angles work with no conversion
    @test graticule(unit; parallels = [30u"°"], meridians = [90u"°"], segments = 8) ≈
          graticule(unit; parallels = [π/6], meridians = [π/2], segments = 8)

    # a 90° step: both ends out to √2, the chord's midpoint touching the unit sphere
    @test on_sphere([[x, y]], unit) ≈ [[SVector(√2, 0, 0), SVector(0, √2, 0)]]
    @test on_sphere([[x, y]], unit) isa Vector{Vector{SVector{3,Float64}}}
    # only directions count; any AbstractVector works
    @test on_sphere([[2x, 5y]], unit) ≈ on_sphere([[[1.0, 0, 0], [0, 1.0, 0]]], unit) ≈ on_sphere([[x, y]], unit)
    # the widest step sets the scale for all curves: its midpoint touches, the others' are outside
    c = on_sphere([[x, y], [x, normalize(x + y)]], unit)
    @test norm.(midpoints(c[1])) ≈ [1.0]
    @test norm.(midpoints(c[2])) ≈ [cos(π/8) / cos(π/4)]
    # around a centre: the equator of a radius-2 sphere at (1,0,0), in 8 steps of 45°
    ring = only(graticule(Sphere(center = x, radius = 2.0); parallels = [0.0], meridians = 0, segments = 8))
    @test norm.(ring .- Ref(x)) ≈ fill(2.164784400584788, 9)         # 2 / cos(π/8)
    @test norm.(midpoints(ring) .- Ref(x)) ≈ fill(2.0, 8)
    # a parallel's step is shorter than its longitude step: σ with cos σ = sin²φ + cos²φ·cos(45°)
    par = only(graticule(unit; parallels = [π/6], meridians = 0, segments = 8))
    @test norm.(par) ≈ fill(1.0598996098693099, 9)
    @test norm.(midpoints(par)) ≈ fill(1.0, 8)

    # box edges: 12 distinct axis-aligned two-point edges between corners
    edges = box_edges(Box(0..2, 0..4, 6..12))
    @test edges isa Vector{Vector{SVector{3,Float64}}} && all(==(2) ∘ length, edges)
    @test sort([norm(b - a) for (a, b) in edges]) == [2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6]
    @test all(((a, b),) -> count(a .!= b) == 1, edges) && allunique(Set.(edges))
    @test all(p -> p[1] ∈ (0, 2) && p[2] ∈ (0, 4) && p[3] ∈ (6, 12), reduce(vcat, edges))

    ax = axis_lines(SVector(1.0, 2.0, 3.0), 0.5)
    @test ax.polylines == [[SVector(0.5, 2, 3), SVector(1.5, 2, 3)], [SVector(1, 1.5, 3), SVector(1, 2.5, 3)],
                           [SVector(1, 2, 2.5), SVector(1, 2, 3.5)]]
    @test ax.colors == [fill(RGB(1.0, 0, 0), 2), fill(RGB(0.0, 1, 0), 2), fill(RGB(0.0, 0, 1), 2)]
    @test Lines(ax.polylines; color = ax.colors).color === ax.colors
end

@testitem "helpers: streamlines, glyphs" begin
    using StaticArrays, LinearAlgebra, IntervalSets, ColorSchemes
    # UniformStreamlines' NaN-separated paths: one-point pieces kept, empty pieces (leading/trailing/repeated NaN) dropped
    P(x) = SVector(x, 0.0, 0.0)
    @test VolumeRendering.split_at_nan(P.([NaN, 1, 2, NaN, NaN, 3, NaN, 4, 5, 6, NaN])) == [P.([1, 2]), [P(3)], P.([4, 5, 6])]
    @test VolumeRendering.split_at_nan(P.([1, 2])) == [P.([1, 2])]
    calls = Ref(0)
    swirl(p) = (calls[] += 1; SVector(-p[2], p[1], 0.2))     # circles about z, rising: |V| = hypot(x, y, 0.2)
    vf = VectorField(swirl; region = Box(-1..1, -1..1, -0.5..0.5))
    @test vf.region == Box(-1..1, -1..1, -0.5..0.5)
    s = streamlines(vf)
    @test s isa Streamlines && length(s.polylines) > 10
    @test length.(s.values) == length.(s.polylines) && all(>(1), length.(s.polylines))
    @test all(l -> all(p -> all(abs.(p) .<= (1, 1, 0.5) .+ 1e-9), l), s.polylines)
    @test reduce(vcat, s.values) ≈ [hypot(p[1], p[2], 0.2) for l in s.polylines for p in l]
    # integrated along the field: each streamline keeps its distance from the z axis
    @test all(l -> -(reverse(extrema(p -> hypot(p[1], p[2]), l))...) < 1e-3, s.polylines)
    @test Lines(s.polylines; color = Mapped(s.values, Colormap(ColorSchemes.viridis, 0.1..2))) isa Lines

    vertices = Set(reduce(vcat, s.polylines))
    for every in (12, 5)
        n0 = calls[]
        g = glyphs(s; every)
        narrows = length(g.polylines) ÷ 2
        @test calls[] - n0 == narrows                                 # one speed per arrow, no re-integration
        @test narrows > 10 && length.(g.polylines) == repeat([3, 2], narrows) && length.(g.values) == length.(g.polylines)
        shafts, heads = g.polylines[1:2:end], g.polylines[2:2:end]
        bases, tips = first.(shafts), getindex.(shafts, 2)
        @test bases ⊆ vertices && first.(heads) == tips
        @test norm.(tips .- bases) ≈ fill(1, narrows)                 # UniformStreamlines' unit arrow (scale 1)
        @test norm.(last.(shafts) .- tips) ≈ norm.(last.(heads) .- tips) ≈ fill(0.35 * √2, narrows)   # head strokes at 45°, headfrac 0.35
        @test g.values ≈ [fill(hypot(b[1], b[2], 0.2), n) for b in bases for n in (3, 2)]
    end
end

@testitem "helper lines render" setup = [GLContext] begin
    using StaticArrays, ColorTypes, FixedPointNumbers, ColorSchemes, IntervalSets
    green = RGB{N0f8}(0, 1, 0)
    sphere = Sphere(radius = 1.0)
    # face-on curves on a sphere, alone vs listed after an opaque sphere of their radius: no pixel may be lost.
    # Seen from +x, screen-right is +y; the half-pixel shift puts the vertical curve on a pixel-column centre.
    side = Camera(SVector(3.0, -0.65 / 128, 0), SVector(0.0, -0.65 / 128, 0), SVector(0.0, 0, 1), :ortho, 90.0, 1.3)
    lost(lines) = let alone = render_image(View([lines]; camera = side), 128, 128) .== green
        (count(alone), count(alone .& .!(render_image(View([Surface(sphere), lines]; camera = side), 128, 128) .== green)))
    end
    meridian = Lines(graticule(sphere; parallels = 0, meridians = [0.0], segments = 16); color = RGB(0.0, 1, 0))
    arc = Lines(on_sphere([[SVector(cos(a), 0, sin(a)) for a in -1.2:0.3:1.2]], sphere); color = RGB(0.0, 1, 0))
    @test first(lost(meridian)) ∈ 100:104                    # one column, 2 / cos(π/16) of the 1 / 1.3 half-height
    @test last(lost(meridian)) == 0                          # chord midpoints tie with the sphere: list order keeps them
    @test first(lost(arc)) > 80
    @test last(lost(arc)) == 0

    # streamlines coloured by |V|, and their glyphs
    vf = VectorField(p -> SVector(-p[2], p[1], 0.2); region = Box(-1..1, -1..1, -1..1))
    s = streamlines(vf)
    cmap = Colormap(ColorSchemes.viridis, 0.2..1.5)
    img = render_image(View([Lines(s.polylines; color = Mapped(s.values, cmap))]; camera = GLContext.topcam(1.2)), 128, 128)
    lit = findall(!=(RGB{N0f8}(0, 0, 0)), img)
    ρ = map(I -> hypot(GLContext.pixel_xy(1.2, 128, Tuple(I)...)...), lit)    # seen from +z, |V| = hypot(ρ, 0.2)
    @test length(lit) > 2000 && all(abs.(extrema(ρ) .- (0.15, 1.25)) .< 0.05)
    expected = get.(Ref(ColorSchemes.viridis), clamp.((hypot.(ρ, 0.2) .- 0.2) ./ 1.3, 0, 1))
    @test maximum(((c, e),) -> maximum(abs, (c.r - e.r, c.g - e.g, c.b - e.b)), zip(img[lit], expected)) < 0.03
    g = glyphs(s)
    img = render_image(View([Lines(g.polylines; color = Mapped(g.values, cmap))]; camera = GLContext.topcam(1.2)), 128, 128)
    @test count(!=(RGB{N0f8}(0, 0, 0)), img) > 500
end

@testitem "UI: panel sections, log toggle, extinction edits" begin
    import VolumeRendering as VR
    using AxisKeys, ColorSchemes, ColorTypes, IntervalSets, StaticArrays
    xs = range(-1, 1, 4)
    g = GridField(KeyedArray(rand(4, 4, 4); x = xs, y = xs, z = xs))
    gc = GridField(KeyedArray(fill(RGBA(1.0, 0.0, 0.0, 1.0), 4, 4, 4); x = xs, y = xs, z = xs))
    glf = GLSLField("float value(vec3 p) { return 1.0; }\nfloat stepSize(vec3 p) { return 0.1; }")
    c1 = Colormap(ColorSchemes.viridis, 0.0..1.0)
    c2 = Colormap(ColorSchemes.turbo, 1.0..2.0)
    shared = Mapped(g, c1)
    marks = [Volume(shared; region = Box(g)), Surface(Plane(:z, 0.0, Box(g)); color = shared), Surface(Box(g); color = gc),
             Surface(Sphere(radius = 1); color = Mapped(glf, c1)), Lines([[SVector(0.0, 0, 0), SVector(1.0, 0, 0)]]; color = Mapped([[1.0, 2.0]], c2)),
             Lines([[SVector(0.0, 0, 0), SVector(1.0, 0, 0)]]), Volume(Mapped(g, c2); region = Box(g))]
    view = View(marks; camera = Camera(; radius = 1))
    # deduplicated by ===, first-appearance order; per-vertex values and constants contribute no field
    @test VR.panel_fields(view) == [g, gc, glf] && VR.panel_fields(view)[1] === g
    @test VR.panel_colormaps(view) == [c1, c2] && VR.panel_colormaps(view)[2] === c2
    @test applicable(VR.controls!, g) && !applicable(VR.controls!, glf)
    @test all(m -> applicable(VR.controls!, m), [marks; view; c1])

    # log toggle: lo ≤ 0 < hi moves the window, hi ≤ 0 is refused, lo > 0 keeps it, turning off keeps the window
    c = Colormap(ColorSchemes.viridis, -2.0..4.0)
    @test VR.set_log!(c, true) === c && c.log && c.window == 0.004..4.0
    @test (VR.set_log!(c, false); !c.log && c.window == 0.004..4.0)
    c = Colormap(ColorSchemes.viridis, 0.5..4.0); VR.set_log!(c, true)
    @test c.log && c.window == 0.5..4.0
    c = Colormap(ColorSchemes.viridis, -3.0..0.0); VR.set_log!(c, true)
    @test !c.log && c.window == -3.0..0.0
    c = Colormap(ColorSchemes.viridis, (-3.0)..(-1.0)); VR.set_log!(c, true)
    @test !c.log && c.window == (-3.0)..(-1.0)

    # extinction edits return new sorted vectors in 0..1; end points keep u, inner points stay between neighbours
    e = [SVector(0.0, 0.0), SVector(0.5, 0.2), SVector(1.0, 1.0)]
    @test VR.drag_point(e, 2, SVector(0.7, 0.4)) == [SVector(0.0, 0.0), SVector(0.7, 0.4), SVector(1.0, 1.0)]
    @test VR.drag_point(e, 2, SVector(1.5, -1.0)) == [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(1.0, 1.0)]
    @test VR.drag_point(e, 1, SVector(0.3, 0.6)) == [SVector(0.0, 0.6), SVector(0.5, 0.2), SVector(1.0, 1.0)]
    @test VR.drag_point(e, 3, SVector(0.3, 2.0)) == [SVector(0.0, 0.0), SVector(0.5, 0.2), SVector(1.0, 1.0)]
    e2 = [SVector(0.1, 0.0), SVector(0.9, 1.0)]   # end points keep their own u, also when it is not 0 / 1
    @test VR.drag_point(e2, 1, SVector(0.5, 0.3)) == [SVector(0.1, 0.3), SVector(0.9, 1.0)]
    @test VR.drag_point(e2, 2, SVector(0.0, 0.4)) == [SVector(0.1, 0.0), SVector(0.9, 0.4)]
    @test VR.insert_point(e, SVector(0.25, 0.9)) == [SVector(0.0, 0.0), SVector(0.25, 0.9), SVector(0.5, 0.2), SVector(1.0, 1.0)]
    @test VR.insert_point(e, SVector(0.75, 1.3)) == [SVector(0.0, 0.0), SVector(0.5, 0.2), SVector(0.75, 1.0), SVector(1.0, 1.0)]
    @test e == [SVector(0.0, 0.0), SVector(0.5, 0.2), SVector(1.0, 1.0)]   # never mutated in place
end

@testitem "UI: ShowView and ShowControls frames" setup = [GLContext] begin
    import VolumeRendering as VR, CImGui as ig
    using AxisKeys, ColorSchemes, StaticArrays
    ctx = ig.CreateContext()
    @test ig.lib.ImGui_ImplOpenGL3_Init("#version 410")   # renderer backend on the hidden test context; no platform backend
    io = ig.GetIO()
    io.DisplaySize = ig.ImVec2(800, 600); io.DeltaTime = 1 / 60; io.IniFilename = C_NULL

    xs = range(-1, 1, 8)
    g = GridField(KeyedArray([exp(-4(x^2 + y^2 + z^2)) for x in xs, y in xs, z in xs]; x = xs, y = xs, z = xs))
    colored = Mapped(g, Colormap(ColorSchemes.viridis, value_range(g)))
    view = View([Volume(colored; region = Box(g)), Surface(Plane(:z, 0.0, Box(g)); color = colored, opacity = 0.5),
                 Lines([[SVector(0.0, 0, 0), SVector(1.0, 0, 0)]]; label = "axis")]; camera = Camera(; radius = 2))
    winpos = Ref(ig.ImVec2(0, 0))
    function frame()   # the host window starts at (0, 0)–(400, 400), free to move; the image fills its content region
        ig.lib.ImGui_ImplOpenGL3_NewFrame(); ig.NewFrame()
        ig.SetNextWindowPos(ig.ImVec2(0, 0), ig.ImGuiCond_Once); ig.SetNextWindowSize(ig.ImVec2(400, 400), ig.ImGuiCond_Once)
        ig.Begin("host"); VR.ShowView(view); winpos[] = ig.GetWindowPos(); ig.End()
        ig.Render(); ig.lib.ImGui_ImplOpenGL3_RenderDrawData(Ptr{Cint}(ig.GetDrawData()))
    end
    event(f, args...) = (f(io, args...); frame())

    event(ig.AddMousePosEvent, 200, 200)
    @test !isnothing(view.cache)                                   # rendered
    eye, d = view.camera.eye, VR.distance(view.camera)
    event(ig.AddMouseButtonEvent, 0, true); event(ig.AddMousePosEvent, 240, 210); event(ig.AddMouseButtonEvent, 0, false)
    @test view.camera.eye != eye && VR.distance(view.camera) ≈ d   # left drag orbits
    @test (winpos[].x, winpos[].y) == (0, 0)                       # … and the view owns the drag: the host window stays put
    event(ig.AddMouseWheelEvent, 0, 1)
    @test VR.distance(view.camera) ≈ d * exp(-VR.ZOOM_SPEED)        # wheel zooms
    view.flymode = true; lookat = view.camera.lookat
    event(ig.AddKeyEvent, ig.ImGuiKey_W, true); frame(); event(ig.AddKeyEvent, ig.ImGuiKey_W, false)
    @test VR.distance(view.camera) ≈ d * exp(-VR.ZOOM_SPEED) && view.camera.lookat != lookat   # W flies eye and lookat together
    event(ig.AddMousePosEvent, 600, 500); lookat = view.camera.lookat
    event(ig.AddKeyEvent, ig.ImGuiKey_W, true); frame(); event(ig.AddKeyEvent, ig.ImGuiKey_W, false)
    @test view.camera.lookat == lookat                              # … only while the view is hovered
    event(ig.AddMousePosEvent, 200, 200)
    view.flymode = false

    lookat = view.camera.lookat
    event(ig.AddMouseButtonEvent, 1, true); foreach(k -> event(ig.AddMousePosEvent, 200 + 10k, 210), 1:3); event(ig.AddMouseButtonEvent, 1, false)
    @test !VR.controls_open(view) && view.camera.lookat != lookat   # a right drag pans and does not toggle
    event(ig.AddMouseButtonEvent, 1, true); event(ig.AddMouseButtonEvent, 1, false)
    @test VR.controls_open(view)                                    # right click without drag opens the controls
    lookat = view.camera.lookat
    event(ig.AddMouseButtonEvent, 2, true); event(ig.AddMousePosEvent, 230, 220); event(ig.AddMouseButtonEvent, 2, false)
    @test VR.controls_open(view) && view.camera.lookat != lookat    # a middle drag pans too
    @test (winpos[].x, winpos[].y) == (0, 0)
    foreach(_ -> frame(), 1:3)                                      # ShowControls: every section, incl. the extinction editor
    VR.glcheck()

    ig.lib.ImGui_ImplOpenGL3_Shutdown()
    VR.ImPlot.DestroyContext(VR.ImPlot.GetCurrentContext())
    ig.DestroyContext(ctx)
    VR.free!(view)
end

@testitem "_" begin
    import Aqua
    import CompatHelperLocal as CHL
    Aqua.test_all(VolumeRendering; ambiguities = false)
    Aqua.test_ambiguities(VolumeRendering)
    CHL.@check()
end
