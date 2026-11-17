# ── widget ──

const ROT_SPEED = 0.007   # rad per pixel of drag
const ZOOM_SPEED = 0.12   # per mouse-wheel notch

# whether the floating controls window of a view is shown; kept off the View and collected with it
const CONTROLS_OPEN = WeakKeyDict{View,Bool}()
controls_open(view::View) = get(CONTROLS_OPEN, view, false)

# ImGui IDs from the object itself, so a shared object keeps one widget state wherever it appears
push_id(x) = CImGui.PushID(Ptr{Cvoid}(objectid(x)))

"""    VolumeRendering.ShowView(view; size = ImVec2(-CImGui.FLT_MIN, -CImGui.FLT_MIN), controls = true)

Draw `view` as an interactive image in the current Dear ImGui window, filling `size` (by default the remaining
content region; CImGui item-size rules), rendered at device-pixel resolution. Not exported.

Over the image: drag orbits, right/middle drag pans, the wheel zooms. With `view.flymode`, drag looks around and
W/A/S/D/Space/C fly, Shift faster. With `controls = true`, a right-click without drag toggles a floating window with
[`ShowControls`](@ref).
"""
function ShowView(view::View; size::ImVec2 = ImVec2(-CImGui.FLT_MIN, -CImGui.FLT_MIN), controls::Bool = true)
    sz = CImGui.CalcItemSize(size, 256, 256)   # 256 for a zero component
    dw, dh = max(1, round(Int, sz.x)), max(1, round(Int, sz.y))   # logical points
    fbs = unsafe_load(CImGui.GetIO().DisplayFramebufferScale)
    tex = render!(view, max(1, round(Int, dw * fbs.x)), max(1, round(Int, dh * fbs.y)))   # device px
    push_id(view)
    # an invisible button is the interactive item, the image is drawn into its rect: holding it takes the mouse
    # (no window move, no other widget reacting), and IsItemHovered/IsItemActive give the hover/drag state
    clicked = CImGui.InvisibleButton("view", ImVec2(dw, dh), CImGui.ImGuiButtonFlags_MouseButtonLeft |
                                     CImGui.ImGuiButtonFlags_MouseButtonRight | CImGui.ImGuiButtonFlags_MouseButtonMiddle)
    CImGui.AddImage(CImGui.GetWindowDrawList(), CImGui.ImTextureRef(CImGui.ImTextureID(UInt64(tex))),
                    CImGui.GetItemRectMin(), CImGui.GetItemRectMax(), ImVec2(0, 1), ImVec2(1, 0))   # GL rows are bottom-up
    camera_input!(view, dw, dh)
    # a right click without drag toggles the controls window; a right drag pans
    rmb = CImGui.ImGuiMouseButton_Right
    clicked && CImGui.IsMouseReleased(rmb) && !CImGui.IsMouseDragPastThreshold(rmb) && (CONTROLS_OPEN[view] = !controls_open(view))
    CImGui.PopID()
    controls && controls_open(view) && controls_window!(view)
    nothing
end

# camera motion from the input on the view's item (the last submitted one)
function camera_input!(view::View, w, h)
    io = CImGui.GetIO(); cam = view.camera
    d = unsafe_load(io.MouseDelta)
    if CImGui.IsItemActive()
        if CImGui.IsMouseDown(CImGui.ImGuiMouseButton_Left)
            dyaw, dpitch = -d.x * ROT_SPEED, -d.y * ROT_SPEED
            view.flymode ? flylook!(cam, dyaw, dpitch) : orbit!(cam, dyaw, -dpitch)
        elseif CImGui.IsMouseDown(CImGui.ImGuiMouseButton_Right) || CImGui.IsMouseDown(CImGui.ImGuiMouseButton_Middle)
            pan!(cam, d.x / w, d.y / h)
        end
    end

    # wheel and fly keys act while the view is hovered or held, and the view takes them from the host window
    # (no scrolling) and from other widgets
    if CImGui.SetItemKeyOwner(CImGui.ImGuiKey_MouseWheelY)
        wheel = unsafe_load(io.MouseWheel)
        wheel != 0 && zoom!(cam, exp(-ZOOM_SPEED * wheel), view.initial_camera)
    end
    if view.flymode
        key(k) = CImGui.SetItemKeyOwner(k) && CImGui.IsKeyDown(k) ? 1.0 : 0.0
        fwd = key(CImGui.ImGuiKey_W) - key(CImGui.ImGuiKey_S)
        rgt = key(CImGui.ImGuiKey_D) - key(CImGui.ImGuiKey_A)
        upd = key(CImGui.ImGuiKey_Space) - key(CImGui.ImGuiKey_C)
        speed = view.fly_speed * (unsafe_load(io.KeyShift) ? 5.0 : 1.0)   # Shift sprints
        (fwd != 0 || rgt != 0 || upd != 0) && fly!(cam, fwd, rgt, upd, speed)
    end
end

# floating per-view window; Begin/End inside the host's window is allowed by Dear ImGui
function controls_window!(view::View)
    open = Ref(true)
    (@c CImGui.Begin("Controls##$(objectid(view))", &open[])) && ShowControls(view)
    CImGui.End()
    open[] || (CONTROLS_OPEN[view] = false)
end

# ── controls panel ──

"""    VolumeRendering.controls!(x)

Dear ImGui widgets for the settings of `x`, emitted in its section of [`ShowControls`](@ref). Defined for `View`,
the marks, `Colormap` and `GridField`; define it for a custom field type to give that field a section.
"""
function controls! end

"""    VolumeRendering.ShowControls(view)

Emit the settings panel of `view` into the current Dear ImGui window: a section for the view, one per mark, one per
distinct field with a [`controls!`](@ref) method, and one per distinct `Colormap`. Edits assign the objects' properties.
Not exported.
"""
function ShowControls(view::View)
    section("View", view)
    foreach(((i, m),) -> section(something(m.label, "$(nameof(typeof(m))) $i"), m), pairs(view.marks))
    foreach(((i, f),) -> section("$(nameof(typeof(f))) $i", f), pairs(filter(f -> applicable(controls!, f), panel_fields(view))))
    foreach(((i, c),) -> section("Colormap $i", c), pairs(panel_colormaps(view)))
    nothing
end

section(title, x) = (CImGui.SeparatorText(title); push_id(x); controls!(x); CImGui.PopID())

# objects reachable from the marks' colours, deduplicated by === in first-appearance order
panel_fields(view::View) = unique(Identity, mapreduce(m -> color_fields(m.color), vcat, view.marks; init = []))
panel_colormaps(view::View) = unique(Identity, mapreduce(m -> color_colormaps(m.color), vcat, view.marks; init = []))
color_fields(c::Mapped) = color_fields(c.source)
color_fields(c) = isfield(c) ? [c] : []
color_colormaps(c::Mapped) = [c.colormap]
color_colormaps(_) = []

# ImPlot needs its own context, bound to the host's ImGui context; created lazily on the render thread
ensure_implot!() = ImPlot.GetCurrentContext() == C_NULL && (ImPlot.CreateContext(); ImPlot.SetImGuiContext(CImGui.GetCurrentContext()))

half_width() = (CImGui.GetContentRegionAvail().x - 6) / 2

# a row of radio buttons over (value, label, tooltip) options; returns the clicked value, or nothing
function radio_row(options, current)
    clicked = nothing
    for (i, (value, label, tip)) in enumerate(options)
        i > 1 && CImGui.SameLine()
        CImGui.RadioButton(label, value === current) && (clicked = value)
        isnothing(tip) || CImGui.SetItemTooltip(tip)
    end
    clicked
end

# float slider bound to `obj.name`; clamped also for ctrl-click text input, since out-of-range values fail validate
function slider!(obj, name, label, lo, hi, fmt; log = false)
    v = Ref(Cfloat(getproperty(obj, name)))
    flags = CImGui.ImGuiSliderFlags_AlwaysClamp | (log ? CImGui.ImGuiSliderFlags_Logarithmic : 0)
    (@c CImGui.SliderFloat(label, &v[], lo, hi, fmt, flags)) && setproperty!(obj, name, v[])
end

checkbox(label, on::Bool) = (v = Ref(on); (@c CImGui.Checkbox(label, &v[])) ? v[] : nothing)

const PROJECTIONS = ((:perspective, "perspective", nothing), (:ortho, "ortho", nothing))
const VOLUME_PX = (64, 128, 256, 512, 1024, 2048, 4096)   # discrete `max volume px` steps

function controls!(view::View)
    hw = half_width()
    let p = radio_row(PROJECTIONS, view.camera.projection)
        isnothing(p) || (view.camera.projection = p)
    end
    CImGui.Button("reset view") && reset!(view)
    CImGui.SameLine()
    let v = checkbox("WASD mode", view.flymode)
        isnothing(v) || (view.flymode = v)
    end
    CImGui.SetItemTooltip("W/S forward·back   A/D strafe   Space/C up·down   Shift sprint   drag to look")
    if view.flymode
        CImGui.SameLine(); CImGui.SetNextItemWidth(hw)
        s = default_fly_speed(view.initial_camera)
        slider!(view, :fly_speed, "##flyspeed", s / 10, 10s, "fly speed: %.4g"; log = true)
    end
    CImGui.SetNextItemWidth(hw)
    let v = Ref(Cint(argmin(i -> abs(VOLUME_PX[i] - view.max_volume_px), eachindex(VOLUME_PX)) - 1))   # nearest step
        (@c CImGui.SliderInt("##maxpx", &v[], 0, length(VOLUME_PX) - 1, "max volume px: $(VOLUME_PX[v[]+1])",
                             CImGui.ImGuiSliderFlags_AlwaysClamp)) && (view.max_volume_px = VOLUME_PX[v[]+1])
    end
    CImGui.SameLine(); CImGui.SetNextItemWidth(hw)
    let v = Ref(Cint(view.refine_frames))
        (@c CImGui.SliderInt("##refine", &v[], 1, 64, "refine frames: %d", CImGui.ImGuiSliderFlags_AlwaysClamp)) &&
            (view.refine_frames = v[])
    end
end

const MODES = ((:dvr, "DVR", "direct volume rendering (emission-absorption)"),
               (:mip, "MIP", "maximum intensity projection"),
               (:average, "avg", "average intensity"))

function controls!(m::Volume)
    hw = half_width()
    mapped = m.color isa Mapped   # else a :color field: DVR only, σ from the field
    if mapped
        v = radio_row(MODES, m.mode)
        isnothing(v) || (m.mode = v)
    end
    if m.mode === :dvr
        mapped && CImGui.SameLine()
        CImGui.SetNextItemWidth(hw)
        slider!(m, :density, "##density", 0.02, 5, "density: %.2f"; log = true)
    end
    CImGui.SetNextItemWidth(hw)
    slider!(m, :step_scale, "##stepscale", 1e-3, 1e3, "step scale: %.4g"; log = true)
    CImGui.SameLine(); CImGui.SetNextItemWidth(hw)
    let v = Ref(Cint(m.max_samples))
        (@c CImGui.SliderInt("##maxsamples", &v[], 16, 65536, "max samples: %d",
                             CImGui.ImGuiSliderFlags_AlwaysClamp | CImGui.ImGuiSliderFlags_Logarithmic)) && (m.max_samples = v[])
    end
    CImGui.SetItemTooltip("most samples per ray: a ray that needs more is sampled more coarsely, not cut short")
    mapped && extinction_editor!(m)
end

controls!(m::Surface) = (CImGui.SetNextItemWidth(half_width()); slider!(m, :opacity, "##opacity", 0.01, 1, "opacity: %.2f"))
controls!(m::Lines) = (CImGui.SetNextItemWidth(half_width()); slider!(m, :width_px, "##width", 0.5, 20, "width: %.1f px"; log = true))

controls!(f::GridField) = let v = checkbox("linear interpolation", f.interpolation === :linear)
    isnothing(v) || (f.interpolation = v ? :linear : :nearest)
end

# σ(u) editor: drag points (end points only in σ), double-click off the points adds, right-click on an inner point removes
function extinction_editor!(m::Volume)
    ensure_implot!()
    flags = ImPlot.ImPlotFlags_NoMenus | ImPlot.ImPlotFlags_NoBoxSelect | ImPlot.ImPlotFlags_NoLegend
    ImPlot.BeginPlot("##extinction", ImVec2(-1, 150), flags) || return
    ImPlot.SetupAxes("u", "extinction")
    ImPlot.SetupAxesLimits(0, 1, 0, 1, ImPlot.ImPlotCond_Always)
    e = m.extinction
    ImPlot.PlotLine("##curve", first.(e), last.(e))
    edited = e; on_point = false
    for (i, p) in pairs(e)
        x, y, hovered = Ref(p[1]), Ref(p[2]), Ref(false)
        ImPlot.DragPoint(i, x, y, ImVec4(1, 1, 1, 1), 5, 0, C_NULL, hovered) && (edited = drag_point(e, i, SVector(x[], y[])))
        on_point |= hovered[]
        hovered[] && CImGui.IsMouseClicked(1) && 1 < i < length(e) && (edited = deleteat!(copy(e), i))
    end
    if !on_point && ImPlot.IsPlotHovered() && CImGui.IsMouseDoubleClicked(0)
        mp = ImPlot.GetPlotMousePos()
        edited = insert_point(e, SVector(mp.x, mp.y))
    end
    ImPlot.EndPlot()
    edited === e || (m.extinction = edited)
end

# a dragged point stays between its neighbours (so the vector stays sorted and point indices keep their identity
# during the drag); end points move only in σ
function drag_point(e, i, p)
    u = i in (1, length(e)) ? e[i][1] : clamp(p[1], e[i-1][1], e[i+1][1])
    setindex!(copy(e), SVector(u, clamp(p[2], 0, 1)), i)
end
insert_point(e, p) = sort!([e; [clamp.(p, 0, 1)]]; by = first)

# ColormapButtons over these schemes, registered by name in the current ImPlot context on first use
const SCHEMES = (viridis = ColorSchemes.viridis, inferno = ColorSchemes.inferno, turbo = ColorSchemes.turbo,
                 afmhot = ColorSchemes.afmhot, grays = ColorSchemes.grays)
const IMPLOT_CMAP_N = 32   # samples; ImPlot interpolates between them
function implot_cmap(name::Symbol)
    idx = ImPlot.GetColormapIndex(String(name))
    idx >= 0 && return idx
    cols = map(c -> ImVec4(c.r, c.g, c.b, 1), get(SCHEMES[name], range(0, 1, IMPLOT_CMAP_N)))
    ImPlot.AddColormap(String(name), cols, IMPLOT_CMAP_N, false)   # continuous
end

# turning log on with lo ≤ 0 < hi also moves the window to (hi/1e3)..hi (a log window must be positive); with hi ≤ 0 it is refused
function set_log!(c::Colormap, on::Bool)
    lo, hi = endpoints(c.window)
    on && hi <= 0 && return c
    on && lo <= 0 && (c.window = (hi / 1e3)..hi)
    c.log = on
    c
end

function controls!(c::Colormap)
    ensure_implot!()
    spacing = unsafe_load(CImGui.GetStyle().ItemSpacing).x
    bw = (CImGui.GetContentRegionAvail().x - spacing * (length(SCHEMES) - 1)) / length(SCHEMES)   # equal share of the row
    for (i, (name, scheme)) in enumerate(pairs(SCHEMES))
        i > 1 && CImGui.SameLine()
        ImPlot.ColormapButton(String(name), ImVec2(bw, 0), implot_cmap(name)) && (c.scheme = scheme)
        scheme === c.scheme && CImGui.AddRect(CImGui.GetWindowDrawList(), CImGui.GetItemRectMin(), CImGui.GetItemRectMax(),
                                              CImGui.GetColorU32(CImGui.ImGuiCol_DragDropTarget), 0, 3)
    end
    let v = checkbox("log scale", c.log)
        isnothing(v) || set_log!(c, v)
    end
    # interactive colour scale in value units (scroll zooms, drag pans); written back only on interaction
    window = Tuple(endpoints(c.window))
    ref = Ref(window)
    ImPlotExtra.colorbar!("##colorbar", c.scheme, ref, c.log ? log10 : identity, CImGui.GetContentRegionAvail().x;
                          orientation = :horizontal)
    ref[] == window || (c.window = ref[][1]..ref[][2])
    nothing
end
