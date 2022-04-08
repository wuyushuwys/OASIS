import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

from models.sync_batchnorm import SynchronizedBatchNorm2d


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class BatchNorm2dEval(nn.Module):
    """
    BatchNorm without affine
    """

    def __init__(self, num_channels, epsilon=1e-05):
        super(BatchNorm2dEval, self).__init__()
        self.epsilon = epsilon
        self.num_channels = num_channels

    def forward(self, input: torch.Tensor):
        assert len(input.shape) in (2, 4)
        if len(input.shape) == 2:
            mean = input.mean(dim=0)
            var = ((input - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = input.mean(dim=(0, 2, 3))
            var = ((input - mean) ** 2).mean(dim=(0, 2, 3)).view(1, self.num_channels, 1, 1)

        input_hat = (input - mean) / torch.sqrt(var + self.epsilon)


        return input_hat


class SPADE(nn.Module):
    def __init__(self, opt, norm_nc, label_nc):
        super().__init__()
        self.first_norm = get_norm_layer(opt, norm_nc)
        # self.first_norm = SynchronizedBatchNorm2d(norm_nc)
        # self.first_norm_eval = nn.BatchNorm2d(affine=True)
        ks = opt.spade_ks
        nhidden = 128
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):
        if self.training:
            normalized = self.first_norm(x)
        else:
            normalized = x
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out


def get_spectral_norm(opt):
    if opt.no_spectral_norm:
        return Identity()
    else:
        return spectral_norm


def get_norm_layer(opt, norm_nc):
    if opt.param_free_norm == 'instance':
        return nn.InstanceNorm2d(norm_nc, affine=False)
    # if opt.param_free_norm == 'syncbatch':
    #     return SynchronizedBatchNorm2d(norm_nc, affine=False)
    if opt.param_free_norm == 'batch':
        return nn.BatchNorm2d(norm_nc, affine=False)
    else:
        raise ValueError('%s is not a recognized param-free norm type in SPADE'
                         % opt.param_free_norm)


if __name__ == "__main__":
    import torch

    model = BatchNorm2dEval(64)
    x = torch.rand([1, 64, 64, 64])
    y = model(x)
    print(y.shape)

    torch.onnx.export(model, x, 'test.onnx')
