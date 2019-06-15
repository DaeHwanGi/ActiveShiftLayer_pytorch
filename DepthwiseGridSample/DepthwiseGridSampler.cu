//#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/macros/Macros.h>

namespace at { namespace native {

using namespace at::cuda::detail;

	static __forceinline__ __device__
	bool within_bounds_2d(int h, int w, int H, int W) {
		return h >= 0 && h < H && w >= 0 && w < W;
	}

	template<typename scalar_t>
	static __forceinline__ __device__
	void safe_add_2d(scalar_t *data, int h, int w,
					int sH, int sW, int H, int W,
					scalar_t delta) {
		if (within_bounds_2d(h, w, H, W)) {
			atomicAdd(data + h * sH + w * sW, delta);
		}
	}

	template <typename scalar_t>
	C10_LAUNCH_BOUNDS_1(1024)
		__global__ void depthwise_grid_sampler_kernel(
			const int nthreads,
			TensorInfo<scalar_t, int> input,
			TensorInfo<scalar_t, int> grid,
			TensorInfo<scalar_t, int> output) {

		int C = input.sizes[1];
		int inp_H = input.sizes[2];
		int inp_W = input.sizes[3];

		int out_C = grid.sizes[1];
		int out_H = grid.sizes[2];
		int out_W = grid.sizes[3];

		int inp_sN = input.strides[0];
		int inp_sC = input.strides[1];
		int inp_sH = input.strides[2];
		int inp_sW = input.strides[3];

		int grid_sN = grid.strides[0];
		int grid_sC = grid.strides[1];
		int grid_sH = grid.strides[2];
		int grid_sW = grid.strides[3];
		int grid_sCoor = grid.strides[4];

		int out_sN = output.strides[0];
		int out_sC = output.strides[1];
		int out_sH = output.strides[2];
		int out_sW = output.strides[3];

		CUDA_KERNEL_LOOP(index, nthreads) {
			const int w = index % out_W;
			const int h = (index / out_W) % out_H;
			const int c = (index / (out_H * out_W)) % out_C;
			const int n = index / (out_C * out_H * out_W);
			const int grid_offset = n * grid_sN + c * grid_sC + h * grid_sH + w * grid_sW;

			// get the corresponding input x, y, z co-ordinates from grid
			scalar_t ix = grid.data[grid_offset];
			scalar_t iy = grid.data[grid_offset + grid_sCoor];

			// normalize ix, iy, iz from [-1, 1] to [0, inp_W-1] & [0, inp_H-1] & [0, inp_D-1]
			float ixf = ((ix + 1.f) / 2) * (inp_W - 1);
			float iyf = ((iy + 1.f) / 2) * (inp_H - 1);

			ix = static_cast<scalar_t>(ixf);
			iy = static_cast<scalar_t>(iyf);

			// get corner pixel values from (x, y, z)
			int ix_nw = static_cast<int>(::floor(ix));
			int iy_nw = static_cast<int>(::floor(iy));

			int ix_ne = ix_nw + 1;
			int iy_ne = iy_nw;

			int ix_sw = ix_nw;
			int iy_sw = iy_nw + 1;

			int ix_se = ix_nw + 1;
			int iy_se = iy_nw + 1;

			// get surfaces to each neighbor:
			scalar_t nw = (1 - ix + ix_nw) * (1 - iy + iy_nw);
			scalar_t ne = (ix - ix_nw) * (1 - iy + iy_nw);
			scalar_t sw = (1 - ix + ix_nw) * (iy - iy_nw);
			scalar_t se = (ix - ix_nw) * (iy - iy_nw);

			auto inp_ptr_NC = input.data + n * inp_sN + c * inp_sC;
			auto out_ptr_NCHW = output.data + n * out_sN + c * out_sC + h * out_sH + w * out_sW;

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

	template <typename scalar_t>
	C10_LAUNCH_BOUNDS_1(1024)
		__global__ void depthwise_grid_sampler_backward_kernel(
			const int nthreads,
			TensorInfo<scalar_t, int> grad_output,
			TensorInfo<scalar_t, int> input,
			TensorInfo<scalar_t, int> grid,
			TensorInfo<scalar_t, int> grad_input,	// initialized to zeros
			TensorInfo<scalar_t, int> grad_grid) {	// initialized to empty

		int C = input.sizes[1];
		int inp_H = input.sizes[2];
		int inp_W = input.sizes[3];

		int out_C = grid.sizes[1];
		int out_H = grid.sizes[2];
		int out_W = grid.sizes[3];

		int inp_sN = input.strides[0];
		int inp_sC = input.strides[1];
		int inp_sH = input.strides[2];
		int inp_sW = input.strides[3];

		int grid_sN = grid.strides[0];
		int grid_sC = grid.strides[1];
		int grid_sH = grid.strides[2];
		int grid_sW = grid.strides[3];
		int grid_sCoor = grid.strides[4];

		int gOut_sN = grad_output.strides[0];
		int gOut_sC = grad_output.strides[1];
		int gOut_sH = grad_output.strides[2];
		int gOut_sW = grad_output.strides[3];

		int gInp_sN = grad_input.strides[0];
		int gInp_sC = grad_input.strides[1];
		int gInp_sH = grad_input.strides[2];
		int gInp_sW = grad_input.strides[3];

		int gGrid_sN = grad_grid.strides[0];
		int gGrid_sC = grad_grid.strides[1];
		int gGrid_sH = grad_grid.strides[2];
		int gGrid_sW = grad_grid.strides[3];

		CUDA_KERNEL_LOOP(index, nthreads) {
			const int w = index % out_W;
			const int h = (index / out_W) % out_H;
			const int c = (index / (out_H * out_W)) % out_C;
			const int n = index / (out_C * out_H * out_W);
			const int grid_offset = n * grid_sN + c * grid_sC + h * grid_sH + w * grid_sW;

			// get the corresponding input x, y, z co-ordinates from grid
			scalar_t ix = grid.data[grid_offset];
			scalar_t iy = grid.data[grid_offset + grid_sCoor];

			// normalize ix, iy, iz from [-1, 1] to [0, inp_W-1] & [0, inp_H-1] & [0, inp_D-1]
			float ixf = ((ix + 1.f) / 2) * (inp_W - 1);
			float iyf = ((iy + 1.f) / 2) * (inp_H - 1);

			// multipliers for gradients on ix, iy, and iz
			// E.g.,  0 for out-of-bound indices when GridSamplerPadding::Border
			scalar_t gix_mult, giy_mult;
			gix_mult = static_cast<scalar_t>(1);
			giy_mult = static_cast<scalar_t>(1);

			ix = static_cast<scalar_t>(ixf);
			iy = static_cast<scalar_t>(iyf);

			// get corner pixel values from (x, y)
			int ix_nw = static_cast<int>(::floor(ix));
			int iy_nw = static_cast<int>(::floor(iy));

			int ix_ne = ix_nw + 1;
			int iy_ne = iy_nw;

			int ix_sw = ix_nw;
			int iy_sw = iy_nw + 1;

			int ix_se = ix_nw + 1;
			int iy_se = iy_nw + 1;


			// get surfaces to each neighbor:
			scalar_t nw = (1 - ix + ix_nw) * (1 - iy + iy_nw);
			scalar_t ne = (ix - ix_nw) * (1 - iy + iy_nw);
			scalar_t sw = (1 - ix + ix_nw) * (iy - iy_nw);
			scalar_t se = (ix - ix_nw) * (iy - iy_nw);

			scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0);
			scalar_t *gOut_ptr_NCHW = grad_output.data + n * gOut_sN + c * gOut_sC + h * gOut_sH + w * gOut_sW;
			scalar_t *gGrid_ptr_NCHW = grad_grid.data + c * gGrid_sC + h * gGrid_sH + w * gGrid_sW;
			scalar_t *gInp_ptr_NC = grad_input.data + n * gInp_sN + c * gInp_sC;
			scalar_t *inp_ptr_NC = input.data + n * inp_sN + c * inp_sC;

			scalar_t gOut = *gOut_ptr_NCHW;

			// calculate and set grad_input
			safe_add_2d(gInp_ptr_NC, iy_nw, ix_nw, gInp_sH, gInp_sW, inp_H, inp_W, nw * gOut);
			safe_add_2d(gInp_ptr_NC, iy_ne, ix_ne, gInp_sH, gInp_sW, inp_H, inp_W, ne * gOut);
			safe_add_2d(gInp_ptr_NC, iy_sw, ix_sw, gInp_sH, gInp_sW, inp_H, inp_W, sw * gOut);
			safe_add_2d(gInp_ptr_NC, iy_se, ix_se, gInp_sH, gInp_sW, inp_H, inp_W, se * gOut);

			// calculate grad_grid
			if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
				scalar_t nw_val = inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW];
				gix -= nw_val * (1 - iy + iy_nw) * gOut;
				giy -= nw_val * (1 - ix + ix_nw) * gOut;
			}
			if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
				scalar_t ne_val = inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW];
				gix += ne_val * (1 - iy + iy_nw) * gOut;
				giy -= ne_val * (ix - ix_nw) * gOut;
			}
			if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
				scalar_t sw_val = inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW];
				gix -= sw_val * (iy - iy_nw) * gOut;
				giy += sw_val * (1 - ix + ix_nw) * gOut;
			}
			if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
				scalar_t se_val = inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW];
				gix += se_val * (iy - iy_nw) * gOut;
				giy += se_val * (ix - ix_nw) * gOut;
			}

			// un-normalize grad_grid values back to [-1, 1] constraints
			gix = gix * (inp_W - 1) / 2;
			giy = giy * (inp_H - 1) / 2;

			// assuming grad_grid is contiguous
			gGrid_ptr_NCHW[0] = gix_mult * gix;
			gGrid_ptr_NCHW[1] = giy_mult * giy;

		}
	}


	// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
	Tensor depthwise_grid_sampler_cuda_forward(const Tensor& input, const Tensor& grid,
								int64_t interpolation_mode, int64_t padding_mode) {
	  auto N = input.size(0);
	  auto C = grid.size(1);
	  auto H = grid.size(2);
	  auto W = grid.size(3);
	  auto output = at::empty({N, C, H, W}, input.options());
	  int count = static_cast<int>(N * C * H * W);
	  if (count > 0) {
		AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "depthwise_grid_sampler_cuda_forward", [&] {
		  depthwise_grid_sampler_kernel<scalar_t>
			<<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
			  count,
			  getTensorInfo<scalar_t, int>(input),
			  getTensorInfo<scalar_t, int>(grid),
			  getTensorInfo<scalar_t, int>(output));
		});
	  }
	  return output;
	}


	// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
	std::tuple<Tensor, Tensor>
	depthwise_grid_sampler_cuda_backward(const Tensor& grad_output, const Tensor& input, const Tensor& grid) {
	  auto N = input.size(0);
	  auto C = grid.size(1);
	  auto H = grid.size(2);
	  auto W = grid.size(3);
	  auto grad_input = at::zeros_like(input);
	  auto grad_grid = at::empty_like(grid);
	  int count = static_cast<int>(N * C * H * W);
	  if (count > 0) {
		AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "depthwise_grid_sampler_cuda_backward", [&] {
		  depthwise_grid_sampler_backward_kernel<scalar_t>
			<<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
			  count,
			  getTensorInfo<scalar_t, int>(grad_output),
			  getTensorInfo<scalar_t, int>(input),
			  getTensorInfo<scalar_t, int>(grid),
			  getTensorInfo<scalar_t, int>(grad_input),
			  getTensorInfo<scalar_t, int>(grad_grid));
		});
	  }
	  return std::make_tuple(grad_input, grad_grid);
	}

}}  // namespace at::native
