#include <torch/extension.h>

#include <ATen/Parallel.h>
#include <vector>

static inline bool within_bounds_2d(int64_t h, int64_t w, int64_t H, int64_t W) {
	return h >= 0 && h < H && w >= 0 && w < W;
}
template<typename scalar_t>
static inline void safe_add_2d(scalar_t *data, int64_t h, int64_t w,
	int64_t sH, int64_t sW,
	int64_t H, int64_t W,
	scalar_t delta) {
	if (within_bounds_2d(h, w, H, W)) {
		data[h * sH + w * sW] += delta;
	}
}
template<typename scalar_t>
torch::Tensor activeshift2d_forward(const torch::Tensor& input, const torch::Tensor& theta) {
	int64_t N = input.size(0);
	int64_t C = input.size(1);
	int64_t inp_H = input.size(2);
	int64_t inp_W = input.size(3);

	int64_t inp_sN = input.stride(0);
	int64_t inp_sC = input.stride(1);
	int64_t inp_sH = input.stride(2);
	int64_t inp_sW = input.stride(3);

	int64_t theta_sC = theta.stride(0);
	int64_t theta_sCoor = theta.stride(1);

	auto output = at::empty({ N, C, inp_H, inp_W }, input.options());
	int64_t out_C = output.size(1);
	int64_t out_H = output.size(2);
	int64_t out_W = output.size(3);

	int64_t out_sN = output.stride(0);
	int64_t out_sC = output.stride(1);
	int64_t out_sH = output.stride(2);
	int64_t out_sW = output.stride(3);

	scalar_t *inp_ptr = input.data<scalar_t>();
	scalar_t *out_ptr = output.data<scalar_t>();
	scalar_t *theta_ptr = theta.data<scalar_t>();
	// loop over each output pixel
	at::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
		for (int64_t n = start; n < end; ++n) {
			for (int64_t c = 0; c < out_C; ++c) {
				scalar_t *inp_ptr_NC = inp_ptr + n * inp_sN + c * inp_sC;
				scalar_t *theta_ptr_C = theta_ptr + c * theta_sC;
				scalar_t alpha = *theta_ptr_C;
				scalar_t beta = theta_ptr_C[theta_sCoor];
				int64_t floor_alpha = static_cast<int64_t>(std::floor(alpha));
				int64_t floor_beta = static_cast<int64_t>(std::floor(beta));
				for (int64_t h = 0; h < out_H; ++h) {
					for (int64_t w = 0; w < out_W; ++w) {
						// get corner pixel values from (x, y)
						int64_t ix_nw = w + floor_beta;
						int64_t iy_nw = h + floor_alpha;

						int64_t ix_ne = ix_nw + 1;
						int64_t iy_ne = iy_nw;

						int64_t ix_sw = ix_nw;
						int64_t iy_sw = iy_nw + 1;

						int64_t ix_se = ix_nw + 1;
						int64_t iy_se = iy_nw + 1;

						// get surfaces to each neighbor:
						scalar_t nw = (1 - alpha + floor_alpha) * (1 - beta + floor_beta);
						scalar_t ne = (alpha - floor_alpha) * (1 - beta + floor_beta);
						scalar_t sw = (1 - alpha + floor_alpha) * (beta - floor_beta);
						scalar_t se = (alpha - floor_alpha) * (beta - floor_beta);

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
std::tuple<torch::Tensor, torch::Tensor> activeshift2d_backward(
	const torch::Tensor& grad_output,
	const torch::Tensor& input,
	const torch::Tensor& theta) {
	auto grad_input = at::zeros_like(input);
	auto grad_theta = at::zeros_like(theta);
	int64_t N = input.size(0);
	int64_t C = input.size(1);
	int64_t inp_H = input.size(2);
	int64_t inp_W = input.size(3);

	int64_t inp_sN = input.stride(0);
	int64_t inp_sC = input.stride(1);
	int64_t inp_sH = input.stride(2);
	int64_t inp_sW = input.stride(3);

	int64_t theta_sC = theta.stride(0);
	int64_t theta_sCoor = theta.stride(1);

	int64_t gOut_sN = grad_output.stride(0);
	int64_t gOut_sC = grad_output.stride(1);
	int64_t gOut_sH = grad_output.stride(2);
	int64_t gOut_sW = grad_output.stride(3);

	int64_t gInp_sN = grad_input.stride(0);
	int64_t gInp_sC = grad_input.stride(1);
	int64_t gInp_sH = grad_input.stride(2);
	int64_t gInp_sW = grad_input.stride(3);

	int64_t gTheta_sC = grad_theta.stride(0);
	int64_t gTheta_sCoor = grad_theta.stride(1);
	scalar_t *inp_ptr = input.data<scalar_t>();
	scalar_t *theta_ptr = theta.data<scalar_t>();
	scalar_t *gOut_ptr = grad_output.data<scalar_t>();
	scalar_t *gInp_ptr = grad_input.data<scalar_t>();
	scalar_t *gTheta_ptr = grad_theta.data<scalar_t>();
	// loop over each output pixel
	at::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
		for (int64_t n = start; n < end; ++n) {
			scalar_t *grid_ptr_N = grid_ptr + n * grid_sN;
			scalar_t *inp_ptr_N = inp_ptr + n * inp_sN;
			scalar_t *gGrid_ptr_N = gGrid_ptr + n * gGrid_sN;
			for (int64_t c = 0; c < out_C; ++c) {
				scalar_t *theta_ptr_C = theta_ptr + c * theta_sC;
				scalar_t *gTheta_ptc_C = gTheta_ptr + c * gTheta_sC;
				scalar_t alpha = *theta_ptr_C;
				scalar_t beta = theta_ptr_C[theta_sCoor];
				int64_t floor_alpha = static_cast<int64_t>(std::floor(alpha));
				int64_t floor_beta = static_cast<int64_t>(std::floor(beta));
				for (int64_t h = 0; h < out_H; ++h) {
					for (int64_t w = 0; w < out_W; ++w) {
						// get the corresponding input x, y, z co-ordinates from grid
						scalar_t *grid_ptr_NCHW = grid_ptr_N + c * grid_sC + h * grid_sH + w * grid_sW;
						scalar_t *gGrid_ptr_NDHW = gGrid_ptr_N + c * gGrid_sC + h * gGrid_sH + w * gGrid_sW;

						// get corner pixel values from (x, y)
						int64_t ix_nw = w + floor_beta;
						int64_t iy_nw = h + floor_alpha;

						int64_t ix_ne = ix_nw + 1;
						int64_t iy_ne = iy_nw;

						int64_t ix_sw = ix_nw;
						int64_t iy_sw = iy_nw + 1;

						int64_t ix_se = ix_nw + 1;
						int64_t iy_se = iy_nw + 1;

						// get surfaces to each neighbor:
						scalar_t nw = (1 - alpha + floor_alpha) * (1 - beta + floor_beta);
						scalar_t ne = (alpha - floor_alpha) * (1 - beta + floor_beta);
						scalar_t sw = (1 - alpha + floor_alpha) * (beta - floor_beta);
						scalar_t se = (alpha - floor_alpha) * (beta - floor_beta);

						scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0);
						scalar_t *gOut_ptr_NCHW = gOut_ptr + n * gOut_sN + c * gOut_sC + h * gOut_sH + w * gOut_sW;
						scalar_t *gInp_ptr_NC = gInp_ptr + n * gInp_sN + c * gInp_sC;
						scalar_t *inp_ptr_NC = inp_ptr_N + c * inp_sC;
						// calculate bilinear weighted pixel value and set output pixel
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

						// assuming grad_grid is contiguous
						*gTheta_ptc_C += giy; //alpha
						gTheta_ptc_C[gTheta_sCoor] += gix;
					}
				}

				*gTheta_ptc_C = *gTheta_ptc_C / N;
				gTheta_ptc_C[gTheta_sCoor] = gTheta_ptc_C[gTheta_sCoor] / N;
			}
		}
	});
	return std::make_tuple(grad_input, grad_grid);
}

torch::Tensor activeshift_forward(const torch::Tensor& input, const torch::Tensor& theta) {
	return AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "activeshift2d_forward", [&] {
		return activeshift2d_forward<scalar_t>(input, theta);
	});
}
std::tuple<torch::Tensor, torch::Tensor> activeshift_backward(
	const torch::Tensor& grad_output, 
	const torch::Tensor& input, 
	const torch::Tensor& theta) {
	return AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "activeshift2d_backward", [&] {
		return activeshift2d_backward<scalar_t>(grad_output, input, theta);
	});
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &activeshift_forward, "activeshift_forward");
  m.def("backward", &activeshift_backward, "activeshift_backward");
}
