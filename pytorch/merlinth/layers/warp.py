try:
    import optoth.warp
except:
    print('optoth could not be imported')

import torch

class WarpForward(torch.nn.Module):
    def forward(self, x, u):
        # we assume that the input does not have any channel dimension
        # x [batch, frames, M, N]
        # u [batch, frames, frames_all, M, N, 2]
        out_shape = u.shape[:-1]
        M, N = u.shape[-3:-1]
        x = torch.repeat_interleave(torch.unsqueeze(x, -3), repeats=u.shape[-4], dim=-3)
        x = torch.reshape(x, (-1, 1, M, N)) # [batch, frames * frames_all, 1, M, N]
        u = torch.reshape(u, (-1, M, N, 2)) # [batch, frames * frames_all, M, N, 2]
        Wx = optoth.warp.WarpFunction.apply(x, u)
        return torch.reshape(Wx, out_shape)

class WarpAdjoint(torch.nn.Module):
    def forward(self, x, u):
        # we assume that the input does not have any channel dimension
        # x [batch, frames, frames_all, M, N]
        # u [batch, frames, frames_all, M, N, 2]
        out_shape = u.shape[:-1]
        M, N = u.shape[-3:-1]
        x = torch.reshape(x, (-1, 1, M, N)) # [batch * frames * frames_all, 1, M, N]
        u = torch.reshape(u, (-1, M, N, 2)) # [batch * frames * frames_all, M, N, 2]
        x_warpT = optoth.warp.WarpTransposeFunction.apply(x, u)
        x_warpT = torch.reshape(x_warpT, out_shape)
        x_warpT = torch.sum(x_warpT, -3)
        return x_warpT
