module VolumeRendering

import ModernGL as GL
import Rotations
using ColorSchemes: ColorScheme
using AxisKeys: AxisKeys, KeyedArray, axiskeys, dimnames, named_axiskeys
using ColorTypes: Color, Colorant, RGB, RGBA, TransparentColor
using FixedPointNumbers: FixedPoint, N0f8
using IntervalSets: ClosedInterval, endpoints, (..)
using LinearAlgebra: cross, dot, norm, normalize
using StaticArrays: SVector, SMatrix, @SMatrix, setindex

export View, Camera, Volume, Surface, Lines, Sphere, Box, Plane, GridField, GLSLField, Texture, Colormap, Mapped, reset!,
       field_glsl, value_kind, textures, uniforms, value_range, axis_index_glsl, shape_glsl, bounds, aabb,
       orbit!, flylook!, pan!, zoom!, fly!, view_proj, render!, render_image, free!

include("gl.jl")
include("camera.jl")
include("objects.jl")
include("marks.jl")
include("render.jl")

end
