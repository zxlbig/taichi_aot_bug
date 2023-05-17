#include <cmath>
#include <iostream>
#include <numeric>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <taichi/cpp/taichi.hpp>
#include <limits>
#define RGBA8 4

constexpr int WIDTH = 512;
constexpr int HEIGHT = 512;
constexpr float DEPTH_VALUE = 1e29;

#ifdef __APPLE__
constexpr TiArch RUNTIME_ARCH = TI_ARCH_METAL;
#else
constexpr TiArch RUNTIME_ARCH = TI_ARCH_VULKAN;
#endif


struct MeshyApp {
  // This is different from what used in python script since compiled shaders are compatible with
  // dynamic ndarray shape
    void run_merge_sdf() {
        ti::Runtime runtime(TI_ARCH_VULKAN);
        ti::AotModule sdf_module = runtime.load_aot_module("assets/sdf.tcm");
        ti::Kernel merge_sdf_init = sdf_module.get_kernel("_merge_sdf_initialize");
        ti::Kernel merge_sdf = sdf_module.get_kernel("_merge_sdf");
        ti::Kernel merge_sdf_finalize = sdf_module.get_kernel("_merge_sdf_finalize");
        std::vector<uint32_t> size_vec{64, 64, 64};
        ti::NdArray<float> itm_signed_distance =
            runtime.allocate_ndarray<float>(size_vec, {1}, true);
        ti::NdArray<float> signed_distance = runtime.allocate_ndarray<float>(size_vec, {1}, true);
        ti::NdArray<float> out_signed_distance =
            runtime.allocate_ndarray<float>(size_vec, {1}, true);
        ti::NdArray<float> itm_weight = runtime.allocate_ndarray<float>(size_vec, {1}, true);
        ti::NdArray<float> weight = runtime.allocate_ndarray<float>(size_vec, {1}, true);
        ti::NdArray<float> out_weight = runtime.allocate_ndarray<float>(size_vec, {1}, true);
        ti::NdArray<float> itm_color = runtime.allocate_ndarray<float>(size_vec, {4}, true);
        ti::NdArray<uint8_t> color = runtime.allocate_ndarray<uint8_t>(size_vec, {4}, true);
        ti::NdArray<float> out_color = runtime.allocate_ndarray<float>(size_vec, {1}, true);
        ti::NdArray<float> itm_sphere = runtime.allocate_ndarray<float>({2}, {3}, true);
        ti::NdArray<float> sphere = runtime.allocate_ndarray<float>({2}, {3}, true);

        merge_sdf_init.clear_args();
        merge_sdf_init.push_arg(itm_signed_distance);
        merge_sdf_init.push_arg(itm_weight);
        merge_sdf_init.push_arg(itm_color);
        merge_sdf_init.launch();
        runtime.wait();

        merge_sdf.clear_args();
        merge_sdf.push_arg(itm_signed_distance);
        merge_sdf.push_arg(itm_weight);
        merge_sdf.push_arg(itm_color);
        merge_sdf.push_arg(signed_distance);
        merge_sdf.push_arg(weight);
        merge_sdf.push_arg(color);
        merge_sdf.push_arg(itm_sphere);
        merge_sdf.push_arg(sphere);
        merge_sdf.launch();
        runtime.wait();

        merge_sdf_finalize.clear_args();
        merge_sdf_finalize.push_arg(out_signed_distance);
        merge_sdf_finalize.push_arg(out_weight);
        merge_sdf_finalize.push_arg(out_color);
        merge_sdf_finalize.push_arg(itm_signed_distance);
        merge_sdf_finalize.push_arg(itm_weight);
        merge_sdf_finalize.push_arg(itm_color);
        merge_sdf_finalize.launch();
        runtime.wait();
    }

    void run_merge_sdf_simplify() {
        ti::Runtime runtime(TI_ARCH_VULKAN);
        ti::AotModule sdf_module = runtime.load_aot_module("assets/sdf.tcm");
        ti::Kernel merge_sdf_init = sdf_module.get_kernel("_merge_sdf_initialize");
        ti::Kernel merge_sdf = sdf_module.get_kernel("_merge_sdf");
        ti::Kernel merge_sdf_finalize = sdf_module.get_kernel("_merge_sdf_finalize");
        std::vector<uint32_t> size_vec{64, 64, 64};
        ti::NdArray<float> itm_signed_distance =
            runtime.allocate_ndarray<float>(size_vec, {1}, true);
    
        ti::NdArray<float> itm_weight = runtime.allocate_ndarray<float>(size_vec, {1}, true);
        ti::NdArray<float> itm_color = runtime.allocate_ndarray<float>(size_vec, {4}, true);
        ti::NdArray<uint8_t> color = runtime.allocate_ndarray<uint8_t>(size_vec, {4}, true);
        ti::NdArray<float> sphere = runtime.allocate_ndarray<float>({2}, {3}, true);

       /* merge_sdf_init.clear_args();
        merge_sdf_init.push_arg(itm_signed_distance);
        merge_sdf_init.push_arg(itm_weight);
        merge_sdf_init.push_arg(itm_color);
        merge_sdf_init.launch();
        runtime.wait();*/

        merge_sdf.clear_args();
        merge_sdf.push_arg(itm_signed_distance);
        merge_sdf.push_arg(itm_weight);
        merge_sdf.push_arg(itm_color);
        merge_sdf.push_arg(itm_signed_distance);
        merge_sdf.push_arg(itm_weight);
        merge_sdf.push_arg(color);
        merge_sdf.push_arg(sphere);
        merge_sdf.push_arg(sphere);
        merge_sdf.launch();
        runtime.wait();

        /*merge_sdf_finalize.clear_args();
        merge_sdf_finalize.push_arg(itm_signed_distance);
        merge_sdf_finalize.push_arg(itm_weight);
        merge_sdf_finalize.push_arg(color);
        merge_sdf_finalize.push_arg(itm_signed_distance);
        merge_sdf_finalize.push_arg(itm_weight);
        merge_sdf_finalize.push_arg(itm_color);
        merge_sdf_finalize.launch();
        runtime.wait();*/
    }
};

int main(int argc, const char** argv) {
    MeshyApp app;
    //M3MeshData mesh_data;
    /*app.render_hemisphere_front_m3();
    app.run_dbg_make_sphere_m3();*/
    app.run_merge_sdf_simplify();
    std::cout << "after render" << std::endl;
  return 0;
}
