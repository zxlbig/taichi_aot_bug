import time
from typing import List, Optional, Tuple
import glm
import numpy as np
import numpy.typing as npt
import taichi as ti
import taichi.math as tm
import mesh2sdf
import trimesh

ti.init(ti.vulkan)

REQUIRED_CAPS = [
    ti.DeviceCapability.spirv_version_1_5,
    ti.DeviceCapability.spirv_has_int8,
    ti.DeviceCapability.spirv_has_int16,
    ti.DeviceCapability.spirv_has_float16
]

ivec3 = tm.ivec3

vec4 = tm.vec4
vec3 = tm.vec3
vec2 = tm.vec2
rgba8 = ti.types.vector(4, ti.u8)
rgba16f = ti.types.vector(4, ti.f16)
rgba32f = ti.types.vector(4, ti.f32)

@ti.kernel
def _merge_sdf_initialize(
    itm_signed_distance: ti.types.ndarray(ti.f32, ndim=3),  # ti.f16
    itm_weight: ti.types.ndarray(ti.f32, ndim=3),  # ti.f16
    itm_color: ti.types.ndarray(rgba32f, ndim=3),  # rgba16f
):
    for i, j, k in itm_signed_distance:
        itm_signed_distance[i, j, k] = 0.0
        itm_weight[i, j, k] = 0.0
        itm_color[i, j, k] = tm.vec4(0.0)


@ti.kernel
def _merge_sdf(
    itm_signed_distance: ti.types.ndarray(ti.f32, ndim=3),  # ti.f16
    itm_weight: ti.types.ndarray(ti.f32, ndim=3),  # ti.f16
    itm_color: ti.types.ndarray(rgba32f, ndim=3),  # rgba16f
    signed_distance: ti.types.ndarray(ti.f32, ndim=3),  # ti.f16
    weight: ti.types.ndarray(ti.f32, ndim=3),  # ti.f16
    color: ti.types.ndarray(rgba8, ndim=3),  # rgba8
    itm_sphere:ti.types.ndarray(tm.vec3, ndim=1),
    sphere: ti.types.ndarray(tm.vec3, ndim=1),
):
    for i, j, k in itm_signed_distance:
        pos = tm.vec3(i, j, k) * itm_sphere[1] + itm_sphere[0]

        idx = (pos - sphere[0]) / sphere[1]
        x0 = ti.cast(ti.floor(idx.x), ti.i32)
        y0 = ti.cast(ti.floor(idx.y), ti.i32)
        z0 = ti.cast(ti.floor(idx.z), ti.i32)
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        if x0 >= 0 and y0 >= 0 and z0 >= 0 and x1 < weight.shape[0] and y1 < weight.shape[1] and z1 < weight.shape[2]:
            # The grid point is not contained in the current volume.
            p0 = tm.ivec3(x0, y0, z0)
            p1 = tm.ivec3(x1, y1, z1)
            pd = idx - p0

            v_weight = 0.01
            if v_weight > 1e-5:
                itm_signed_distance[i, j, k] += v_weight 
                itm_color[i, j, k] += v_weight
                itm_weight[i, j, k] += v_weight


@ti.kernel
def _merge_sdf_finalize(
    out_signed_distance: ti.types.ndarray(ti.f32, ndim=3),  # ti.f16
    out_weight: ti.types.ndarray(ti.f32, ndim=3),  # ti.f16
    out_color: ti.types.ndarray(rgba8, ndim=3),  # rgba8
    itm_signed_distance: ti.types.ndarray(ti.f32, ndim=3),  # ti.f16
    itm_weight: ti.types.ndarray(ti.f32, ndim=3),  # ti.f16
    itm_color: ti.types.ndarray(rgba32f, ndim=3),  # rgba16f
):
    for i, j, k in out_signed_distance:
        total_weight = itm_weight[i, j, k]
        if total_weight > 1e-5:
            out_signed_distance[i, j,
                                k] = itm_signed_distance[i, j, k] / total_weight
            out_color[i, j, k] = tm.vec4(1)
            out_weight[i, j, k] = total_weight

#some magic here, don't change this export order
ti.aot.export(_merge_sdf_initialize)
ti.aot.export(_merge_sdf)
ti.aot.export(_merge_sdf_finalize)
