import torch
import torch.nn as nn
import numpy as np

class shift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, theta):
        ctx.save_for_backward(x)
        B, C, H, W = x.size()
        theta = -theta
        for c in range(C):
            m, n = np.indices((H, W))
            m = torch.from_numpy(m).long()
            n = torch.from_numpy(n).long()
            floor_theta = torch.floor(theta[c, :]).long()
            delta_theta = theta[c, :] - floor_theta.float()

            x0 = n + floor_theta[1]
            x1 = n + floor_theta[1] + 1
            y0 = m + floor_theta[0]
            y1 = m + floor_theta[0] + 1

            x0 = torch.clamp(x0, 0, W - 1)[1:-1, 1:-1]
            x1 = torch.clamp(x1, 0, W - 1)[1:-1, 1:-1]
            y0 = torch.clamp(y0, 0, H - 1)[1:-1, 1:-1]
            y1 = torch.clamp(y1, 0, H - 1)[1:-1, 1:-1]

            z1 = x[:, c:c + 1, y0, x0]
            z2 = x[:, c:c + 1, y0, x1]
            z3 = x[:, c:c + 1, y1, x0]
            z4 = x[:, c:c + 1, y1, x1]

            x[:, c:c + 1, 1:-1, 1:-1] = z1 * (1 - delta_theta[0]) * (1 - delta_theta[1]) + z2 * (1 - delta_theta[0]) * \
                                        delta_theta[1] \
                                        + z3 * delta_theta[0] * (1 - delta_theta[1]) + z4 * delta_theta[0] * \
                                        delta_theta[1]
        return x[:, :, 1:-1, 1:-1]
    @staticmethod
    def backward(ctx, grad_output):
        pass

class ActiveShift2d(nn.Module):
    def __init__(self, in_channel, out_channel, bias=False):
        super(ActiveShift2d, self).__init__()
        self.theta_s = nn.Parameter(torch.FloatTensor(in_channel, 2))
        self.theta_s.data.uniform_(-1, 1)
        self.zeropad = nn.ZeroPad2d(1)
        self.depthwise = nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=bias, groups=in_channel)
        self.pointwise = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.zeropad(x)
        x = shift.apply(x, self.theta_s)
        x = self.depthwise(x)
        return self.pointwise(x)

if __name__ == '__main__':
    x = torch.FloatTensor(16, 3, 224, 224).cuda()
    theta_s = torch.FloatTensor(3, 2)
    layer = ActiveShift2d(3, 5).cuda()
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        x = shift.apply(x, theta_s)
    print(prof)
