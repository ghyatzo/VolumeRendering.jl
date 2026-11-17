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

@testitem "Colormap, Mapped" begin
    import VolumeRendering as VR
    using IntervalSets, AxisKeys
    c = Colormap(:viridis, 0..1)
    @test (c.scheme, c.window, c.log) == (:viridis, 0.0..1.0, false)
    @test Colormap(:afmhot, 1e-3..1.0; log = true).log
    @test_throws "minimum < maximum" Colormap(:viridis, 1..1)
    @test_throws "minimum < maximum" Colormap(:viridis, 2..1)
    @test_throws "positive" Colormap(:viridis, 0..1; log = true)
    @test_throws "positive" Colormap(:viridis, -1..1; log = true)

    g = GridField(KeyedArray(rand(2, 2, 2); x = 0:1, y = 0:1, z = 0:1))
    vals = [[1.0, NaN], [-2.0, 3.0, 0.5]]
    @test Mapped(g, c).source === g
    @test Mapped(vals, c).source === vals
    @test value_range(vals) === -2.0..3.0
    colorfield = GLSLField(""; kind = :color)
    @test_throws "Mapped source" Mapped(colorfield, c)
    @test_throws "Mapped source" Mapped(1.0, c)
    @test_throws MethodError Mapped(g, :viridis)
end

@testitem "marks: construction and validate" begin
    import VolumeRendering as VR
    using AxisKeys, IntervalSets, ColorTypes, StaticArrays

    g = GridField(KeyedArray(rand(2, 2, 2); x = 0:1, y = 0:1, z = 0:1))
    rgba = GridField(KeyedArray(rand(RGBA{Float32}, 2, 2, 2); x = 0:1, y = 0:1, z = 0:1))
    m = Mapped(g, Colormap(:viridis, 0..1))
    box = Box(g)
    pl = [[SVector(0.0, 0, 0), SVector(1.0, 0, 0)], [SVector(0.0, 1, 0), SVector(1.0, 1, 0), SVector(1.0, 2, 0)]]

    v = Volume(m; region = box, label = "v")
    @test (v.color, v.region, v.extinction, v.mode, v.density, v.step_scale, v.label) ==
          (m, box, [[0, 0], [1, 1]], :dvr, 1.0, 0.5, "v")
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

    @test VR.jittered(Volume(Mapped(g, Colormap(:viridis, 0..1)); region = box))
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
end

@testitem "snapshot" begin
    import VolumeRendering as VR
    using AxisKeys, IntervalSets, ColorTypes, StaticArrays

    grid(n) = KeyedArray(rand(Float32, n, n, n); x = 1:n, y = 1:n, z = 1:n)
    pl, vals = [[SVector(0.0, 0, 0), SVector(1.0, 0, 0)]], [[NaN, 1.0]]
    function scene(A)
        g = GridField(A)
        m = Mapped(g, Colormap(:viridis, 0..1))
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
    @test same(() -> sc.m.colormap = Colormap(:viridis, 0..1))       # equal values → equal
    @test same(() -> sc.g.data = KeyedArray(A; x = 1:4, y = 1:4, z = 1:4))   # new wrapper, same parent array
    @test same(() -> A[1] = 99)                                      # in-place edit: undetected (documented)
    @test same(() -> sc.ln.polylines[1][1] = SVector(5.0, 0, 0))
    @test !same(() -> sc.g.data = grid(4))                           # new array
    @test !same(() -> sc.ln.polylines = [[SVector(0.0, 0, 0), SVector(1.0, 0, 0)]])
    @test !same(() -> sc.ln.color.source = [[NaN, 1.0]])
    @test !same(() -> sc.m.colormap.window = 0..2)
    @test !same(() -> sc.m.colormap.log = true)
    @test !same(() -> sc.g.interpolation = :nearest)
    @test !same(() -> sc.vol.extinction[2] = SVector(1.0, 0.5))     # by value, also in place
    @test !same(() -> sc.vol.region = Box(0..1, 0..1, 0..1))
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

@testitem "_" begin
    import Aqua
    import CompatHelperLocal as CHL
    Aqua.test_all(VolumeRendering; ambiguities = false)
    Aqua.test_ambiguities(VolumeRendering)
    CHL.@check()
end
