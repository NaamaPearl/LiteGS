#include <torch/extension.h>
#include "binning.h"
#include "compact.h"
#include "raster.h"
#include "transform.h"



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("create_viewproj_forward", &create_viewproj_forward);
	m.def("create_viewproj_backward", &create_viewproj_backward);
	m.def("create_table", &create_table);
	m.def("tileRange", &tileRange);
	// near_plane: Gaussians at a view-space depth up to it are not visible (LiteGS's fixed 0.2 by default).
	// cull_by_extent: keep a Gaussian whose screen extent overlaps the image (LiteGS: its centre within +-1.3 NDC)
	m.def("get_allocate_size", &get_allocate_size, pybind11::arg("ndc"), pybind11::arg("view_space_z"),
		pybind11::arg("inv_cov2d"), pybind11::arg("opacity"), pybind11::arg("height"), pybind11::arg("width"),
		pybind11::arg("tile_size_h"), pybind11::arg("tile_size_w"), pybind11::arg("valid_length"),
		pybind11::arg("near_plane") = 0.2, pybind11::arg("cull_by_extent") = false);
	m.def("rasterize_forward", &rasterize_forward);
	m.def("rasterize_forward_packed", &rasterize_forward_packed);
	// fp32_grad_sum: sum the colour and opacity gradients in fp32 (LiteGS: fp16, which can overflow)
	m.def("rasterize_backward", &rasterize_backward, pybind11::arg("sorted_points"), pybind11::arg("start_index"),
		pybind11::arg("packed_params"), pybind11::arg("specific_tiles"), pybind11::arg("final_transmitance"),
		pybind11::arg("last_contributor"), pybind11::arg("d_img"), pybind11::arg("d_trans_img"),
		pybind11::arg("d_depth_img"), pybind11::arg("grad_inv_scaler"), pybind11::arg("img_h"), pybind11::arg("img_w"),
		pybind11::arg("tile_h"), pybind11::arg("tile_w"), pybind11::arg("enable_statistic"),
		pybind11::arg("fp32_grad_sum") = false);
	m.def("jacobianRayspace", &jacobianRayspace);
	m.def("createTransformMatrix_forward", &createTransformMatrix_forward);
	m.def("createTransformMatrix_backward", &createTransformMatrix_backward);
	m.def("world2ndc_forward", &world2ndc_forward);
	m.def("world2ndc_backword", &world2ndc_backword);
	m.def("mvp_transform_forward", &mvp_transform_forward);
	m.def("mvp_transform_backward", &mvp_transform_backward);
	m.def("createCov2dDirectly_forward", &createCov2dDirectly_forward);
	m.def("createCov2dDirectly_backward", &createCov2dDirectly_backward);
	m.def("sh2rgb_forward", &sh2rgb_forward);
	m.def("sh2rgb_backward", &sh2rgb_backward);
	m.def("eigh_and_inv_2x2matrix_forward", &eigh_and_inv_2x2matrix_forward);
	m.def("inv_2x2matrix_backward", &inv_2x2matrix_backward);
	m.def("cull_compact_activate", &cull_compact_activate);
	m.def("activate_backward", &activate_backward);
	m.def("adamUpdate", &adamUpdate);
	m.def("frustum_culling_aabb", &frustum_culling_aabb);
	m.def("gpu_driven_pipeline_sparse_op", &gpu_driven_pipeline_sparse_op);
}