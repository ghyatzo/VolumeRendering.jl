# `ShowControls(view)` emits the RenderParams / TransferFunction / Camera widgets into the current
# ImGui window (imgui.jl's right-click window wraps it). Colormap buttons come from ImPlot, over the
# package's ColorSchemes maps registered into it.

using CImGui, ImPlot
using CImGui: ImVec2, ImVec4
using CImGui.CSyntax          # @c — pass Ref pointers to the widget wrappers
import ColorSchemes

const _MODES = (("DVR", "direct volume rendering (emission-absorption)"),
                ("MIP", "maximum intensity projection"),
                ("avg", "average intensity"))
const _RES_VALUES = (64, 128, 256, 512, 1024, 2048, 4096)   # discrete "max render px" steps

# A horizontal row of radio buttons; returns the newly-selected 0-based index, or nothing.
function _radio_row(id, opts, cur)
    sel = nothing
    for (i, (label, tip)) in enumerate(opts)
        i > 1 && CImGui.SameLine()
        CImGui.RadioButton("$label##$id", cur == i - 1) && (sel = i - 1)
        tip !== nothing && CImGui.SetItemTooltip(tip)
    end
    sel
end

# ImPlot needs its own context, bound to the host's ImGui context. Created lazily on the render
# thread (where AddColormap/SampleColormap require the context to be live).
function _ensure_implot!()
    if ImPlot.GetCurrentContext() == C_NULL
        ImPlot.CreateContext()
        ImPlot.SetImGuiContext(CImGui.GetCurrentContext())
    end
end

# Register each ColorSchemes map into ImPlot once (so ColormapButton shows the real gradient) and
# cache symbol → ImPlot colormap index. Names are the lowercase ColorSchemes symbols, distinct from
# ImPlot's capitalized built-ins; the GetColormapIndex guard reuses an existing registration (e.g.
# after a Revise reload where this Dict was reset but ImPlot kept the colormap).
const _CMAPS = Dict{Symbol,Int32}()
const _CMAP_N = 32   # sample points; ImPlot interpolates between them (continuous colormap)
function _cmap_index(sym::Symbol)
    get!(_CMAPS, sym) do
        name = String(sym)
        idx = ImPlot.GetColormapIndex(name)
        idx >= 0 && return Int32(idx)
        scheme = ColorSchemes.colorschemes[sym]
        cols = [ImVec4(c.r, c.g, c.b, 1) for c in ColorSchemes.get(scheme, range(0, 1, _CMAP_N))]
        Int32(ImPlot.AddColormap(name, cols, _CMAP_N, false))   # false = continuous, not qualitative
    end
end

# A row of ImPlot ColormapButtons; a click sets the TF colormap, and the selected one is outlined.
function _colormap_buttons!(tf::TransferFunction)
    for (i, sym) in enumerate(TF_COLORMAPS)
        i > 1 && CImGui.SameLine()
        if ImPlot.ColormapButton(String(sym), ImVec2(80, 0), _cmap_index(sym))
            tf.colormap = sym
            tf.dirty = true
        end
        sym === tf.colormap && CImGui.AddRect(
            CImGui.GetWindowDrawList(), CImGui.GetItemRectMin(), CImGui.GetItemRectMax(),
            CImGui.GetColorU32(CImGui.ImGuiCol_DragDropTarget), 0.0f0, 3.0f0)
    end
end

# Emit the controls into the current ImGui window (no Begin/End of its own).
function ShowControls(view::FieldView)
    _ensure_implot!()
    params = view.params; tf = view.tf; cam = view.camera
    vmin, vmax = value_range(view.field)
    hw = (CImGui.GetContentRegionAvail().x - 6) / 2   # half content width, for paired items

    CImGui.SeparatorText("Rendering")
    let sel = _radio_row("mode", _MODES, params.mode)
        sel !== nothing && (params.mode = sel)
    end
    if params.mode == 0
        CImGui.SameLine(); CImGui.SetNextItemWidth(hw)
        let v = Ref(Cfloat(params.opacity_scale))
            (@c CImGui.SliderFloat("##opacity", &v[], 0.02f0, 5.0f0, "opacity: %.2f",
                                   CImGui.ImGuiSliderFlags_Logarithmic)) && (params.opacity_scale = v[])
        end
    end
    CImGui.SetNextItemWidth(hw)
    let v = Ref(Cfloat(params.step_scale))
        (@c CImGui.SliderFloat("##stepk", &v[], 1.0f-3, 1.0f3, "step scale: %.4g",
                               CImGui.ImGuiSliderFlags_Logarithmic)) && (params.step_scale = v[])
    end
    CImGui.SameLine(); CImGui.SetNextItemWidth(hw)
    let cur = something(findfirst(==(params.max_render_px), _RES_VALUES), length(_RES_VALUES)),
        v = Ref(Cint(cur - 1))
        (@c CImGui.SliderInt("##maxpx", &v[], 0, length(_RES_VALUES) - 1,
                             "max render px: $(_RES_VALUES[v[]+1])")) &&
            (params.max_render_px = _RES_VALUES[v[]+1])
    end
    let v = Ref(params.interp == 1)
        (@c CImGui.Checkbox("trilinear (vs nearest)", &v[])) && (params.interp = v[] ? 1 : 0)
    end

    CImGui.SeparatorText("Transfer function")
    _colormap_buttons!(tf)
    let v = Ref(tf.logscale)
        (@c CImGui.Checkbox("log scale", &v[])) &&
            (tf.logscale = v[]; default_window!(tf, (vmin, vmax)))
    end
    lm = log10(max(vmax, 1e-300))
    lo_min, lo_max = tf.logscale ? (Cfloat(lm - 8), Cfloat(lm + 1)) : (Cfloat(vmin), Cfloat(vmax))
    CImGui.SetNextItemWidth(hw)
    let v = Ref(Cfloat(tf.lo))
        (@c CImGui.SliderFloat("##lo", &v[], lo_min, lo_max, "lo: %.3g")) && (tf.lo = v[])
    end
    CImGui.SameLine(); CImGui.SetNextItemWidth(hw)
    let v = Ref(Cfloat(tf.hi))
        (@c CImGui.SliderFloat("##hi", &v[], lo_min, lo_max, "hi: %.3g")) && (tf.hi = v[])
    end

    CImGui.SeparatorText("View")
    let sel = _radio_row("proj", (("perspective", nothing), ("ortho", nothing)),
                         cam.projection === :perspective ? 0 : 1)
        sel !== nothing && (cam.projection = sel == 0 ? :perspective : :ortho)
    end
    if CImGui.Button("reset view")
        c, r = bounds(view.region)
        reset!(cam, Camera(; center = c, radius = r))
    end
    CImGui.SameLine()
    let v = Ref(params.flymode)
        (@c CImGui.Checkbox("WASD mode", &v[])) && (params.flymode = v[])
    end
    CImGui.SetItemTooltip("W/S forward·back   A/D strafe   Space/C up·down   Shift sprint   drag to look")
    if params.flymode
        CImGui.SameLine(); CImGui.SetNextItemWidth(hw)
        _, r = bounds(view.region)
        let v = Ref(Cfloat(params.fly_speed))
            (@c CImGui.SliderFloat("##flyspeed", &v[], Cfloat(r / 2000), Cfloat(r / 20),
                                   "fly speed: %.4f", CImGui.ImGuiSliderFlags_Logarithmic)) &&
                (params.fly_speed = v[])
        end
    end
    nothing
end
