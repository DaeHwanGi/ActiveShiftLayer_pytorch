#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <vector>

namespace at {
	namespace native {
		at::Tensor linspace_from_neg_one(const Tensor& grid, int64_t num_steps) {
			if (num_steps > 1) {
				return at::linspace(-1, 1, num_steps, grid.options());
			}
			else {
				return at::tensor(-1, grid.options());
			}
		}
		Tensor make_base_grid_4D(
			const Tensor& theta,
			int64_t N,
			int64_t C,
			int64_t H,
			int64_t W) {
			auto base_grid = at::empty({ C, H, W, 3 }, theta.options());

			base_grid.select(-1, 0).copy_(linspace_from_neg_one(theta, W));
			base_grid.select(-1, 1).copy_(linspace_from_neg_one(theta, H).unsqueeze_(-1));
			base_grid.select(-1, 2).fill_(1);
			return base_grid;
		}
		Tensor make_base_backward_grid_4D(
			const Tensor& theta,
			int64_t N,
			int64_t C,
			int64_t H,
			int64_t W) {
			auto base_grid = at::empty({ N, C, H, W, 3 }, theta.options());

			base_grid.select(-1, 0).copy_(linspace_from_neg_one(theta, W));
			base_grid.select(-1, 1).copy_(linspace_from_neg_one(theta, H).unsqueeze_(-1));
			base_grid.select(-1, 2).fill_(1);
			return base_grid;
		}
		Tensor affine_grid_generator_4D(
			const Tensor& theta,
			int64_t N,
			int64_t C,
			int64_t H,
			int64_t W) {
			Tensor base_grid = make_base_grid_4D(theta, N, C, H, W);
			auto grid = base_grid.view({ C, H * W, 3 });
			auto batch_grid = at::empty({ C, H * W, 2 }, theta.options());
			batch_grid = grid.bmm(theta.transpose(1,2));
			return batch_grid.repeat({ N,1,1,1 }).view({ N, C, H, W, 2 });
		}
		Tensor affine_grid_generator_4D_backward(
			const Tensor& grad_grid,
			int64_t N,
			int64_t C,
			int64_t H,
			int64_t W) {
			auto base_grid = make_base_backward_grid_4D(grad_grid, N, C, H, W);
			auto grad_theta = base_grid.view({ N, C, H * W, 3 }).transpose(2, 3);
			auto grad_grid_ = grad_grid.view({ N, C, H * W, 2 });
			auto out_grad = at::empty({ N, C, 3, 2 }, grad_grid.options());

			at::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
				for (int64_t n = start; n < end; ++n) {
					for (int64_t c = 0; c < C; ++c) {
						out_grad[n][c] = grad_theta[n][c].mm(grad_grid_[n][c]);
					}
				}
			});
			return out_grad.transpose(2, 3);
		}
		Tensor affine_grid_generator(const Tensor& theta, IntArrayRef size) {
			return affine_grid_generator_4D(theta, size[0], size[1], size[2], size[3]);
		}
		Tensor affine_grid_generator_backward(const Tensor& grad, IntArrayRef size) {
			return affine_grid_generator_4D_backward(grad, size[0], size[1], size[2], size[3]);
		}
		PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
			m.def("forward", &affine_grid_generator, "depthwise_affine_grid forward");
			m.def("backward", &affine_grid_generator_backward, "depthwise_affine_grid backward");
		}
	}
}
