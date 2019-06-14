#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <vector>

namespace at {
	namespace native {
		static inline bool within_bounds_2d(int64_t h, int64_t w, int64_t H, int64_t W) {
			return h >= 0 && h < H && w >= 0 && w < W;
		}
		template<typename scalar_t>
		Tensor depthwise_grid_sampler_cpu_impl(const Tensor& input, const Tensor& grid) {
			int64_t N = input.size(0);
			int64_t C = input.size(1);
			int64_t inp_H = input.size(2);
			int64_t inp_W = input.size(3);
			int64_t out_C = grid.size(1);
			int64_t out_H = grid.size(2);
			int64_t out_W = grid.size(3);
			auto output = at::empty({ N, C, out_H, out_W }, input.options());

			int64_t inp_sN = input.stride(0);
			int64_t inp_sC = input.stride(1);
			int64_t inp_sH = input.stride(2);
			int64_t inp_sW = input.stride(3);

			int64_t grid_sN = grid.stride(0);
			int64_t grid_sC = grid.stride(1);
			int64_t grid_sH = grid.stride(2);
			int64_t grid_sW = grid.stride(3);
			int64_t grid_sCoor = grid.stride(4);

			int64_t out_sN = output.stride(0);
			int64_t out_sC = output.stride(1);
			int64_t out_sH = output.stride(2);
			int64_t out_sW = output.stride(3);

			scalar_t *inp_ptr = input.data<scalar_t>();
			scalar_t *out_ptr = output.data<scalar_t>();
			scalar_t *grid_ptr = grid.data<scalar_t>();
			// loop over each output pixel
			at::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
				for (int64_t n = start; n < end; ++n) {
					scalar_t *grid_ptr_N = grid_ptr + n * grid_sN;
					for (int64_t c = 0; c < out_C; ++c) {
						scalar_t *inp_ptr_NC = inp_ptr + n * inp_sN + c * inp_sC;
						for (int64_t h = 0; h < out_H; ++h) {
							for (int64_t w = 0; w < out_W; ++w) {
								// get the corresponding input x, y, z co-ordinates from grid
								scalar_t *grid_ptr_NCHW = grid_ptr_N + c * grid_sC + h * grid_sH + w * grid_sW;
								scalar_t ix = *grid_ptr_NCHW;
								scalar_t iy = grid_ptr_NCHW[grid_sCoor];

								// normalize ix, iy, iz from [-1, 1] to [0, inp_W-1] & [0, inp_H-1] & [0, inp_D-1]
								ix = ((ix + 1) / 2) * (inp_W - 1);
								iy = ((iy + 1) / 2) * (inp_H - 1);

								// get corner pixel values from (x, y)
								// for 4d, we used north-east-south-west
								int64_t ix_nw = static_cast<int64_t>(std::floor(ix));
								int64_t iy_nw = static_cast<int64_t>(std::floor(iy));

								int64_t ix_ne = ix_nw + 1;
								int64_t iy_ne = iy_nw;

								int64_t ix_sw = ix_nw;
								int64_t iy_sw = iy_nw + 1;

								int64_t ix_se = ix_nw + 1;
								int64_t iy_se = iy_nw + 1;

								// get surfaces to each neighbor:
								scalar_t nw = (1 - ix + ix_nw) * (1 - iy + iy_nw);
								scalar_t ne = (ix - ix_nw) * (1 - iy + iy_nw);
								scalar_t sw = (1 - ix + ix_nw) * (iy - iy_nw);
								scalar_t se = (ix - ix_nw) * (iy - iy_nw);

								// calculate bilinear weighted pixel value and set output pixel
								scalar_t *out_ptr_NCHW = out_ptr + n * out_sN + c * out_sC + h * out_sH + w * out_sW;
								*out_ptr_NCHW = static_cast<scalar_t>(0);
								if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
									*out_ptr_NCHW += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
								}
								if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
									*out_ptr_NCHW += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
								}
								if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
									*out_ptr_NCHW += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
								}
								if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
									*out_ptr_NCHW += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
								}
							}
						}
					}
				}
			});
			return output;
		}

		template<typename scalar_t>
		std::tuple<Tensor, Tensor>
			depthwise_grid_sampler_backward_cpu_impl(const Tensor& grad_output,
				const Tensor& input, const Tensor& grid) {
			auto grad_input = at::zeros_like(input);
			auto grad_grid = at::empty_like(grid);
			// If interpolation mode is Nearest, then grad_grid is not filled in the
			// loop below.
			return std::make_tuple(grad_input, grad_grid);
		}
		Tensor depthwise_grid_sampler_cpu(const Tensor& input, const Tensor& grid) {
			return AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "depthwise_grid_sampler_cpu", [&] {
				return depthwise_grid_sampler_cpu_impl<scalar_t>(input, grid);
			});
		}
		std::tuple<Tensor, Tensor> depthwise_grid_sampler_backward_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& grid) {
			return AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "depthwise_grid_sampler_backward_cpu", [&] {
				return depthwise_grid_sampler_backward_cpu_impl<scalar_t>(grad_output, input, grid);
			});
		}
		PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
			m.def("forward", &depthwise_grid_sampler_cpu, "depthwise grid sampler forward");
			m.def("backward", &depthwise_grid_sampler_backward_cpu, "depthwise grid sampler backward");
		}
	}
}