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

@testitem "_" begin
    import Aqua
    import CompatHelperLocal as CHL
    Aqua.test_all(VolumeRendering; ambiguities = false)
    Aqua.test_ambiguities(VolumeRendering)
    CHL.@check()
end
