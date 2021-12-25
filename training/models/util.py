import torch
import torch.nn as nn

from models.sync_batchnorm import DataParallelWithCallback


def put_on_multi_gpus(model, opt):
    if opt.gpu_ids != "-1":
        gpus = list(map(int, opt.gpu_ids.split(",")))
        model = DataParallelWithCallback(model, device_ids=gpus).cuda()
    else:
        model.module = model
    assert len(opt.gpu_ids.split(",")) == 0 or opt.batch_size % len(opt.gpu_ids.split(",")) == 0
    return model


def generate_labelmix(label, fake_image, real_image):
    target_map = torch.argmax(label, dim=1, keepdim=True)
    all_classes = torch.unique(target_map)
    for c in all_classes:
        target_map[target_map == c] = torch.randint(0, 2, (1,), device=label.device)
    target_map = target_map.float()
    mixed_image = target_map * real_image + (1 - target_map) * fake_image
    return mixed_image, target_map


class AdaptiveInstanceNorm1d(nn.Module):
    # stylegan
    # https://github.com/rosinality/style-based-gan-pytorch/blob/master/model.py
    def __init__(self, in_features, style_features, transpose=False):
        super().__init__()
        self.transpose = transpose

        self.norm = nn.InstanceNorm1d(in_features)
        self.affine = Linear(in_features=style_features, out_features=in_features * 2)

        self.affine.bias.data[:in_features] = 1  # initial gamma is 1
        self.affine.bias.data[in_features:] = 0

    def forward(self, input, style):
        # input : (batch_size, in_features, length)
        # style : (batch_size, style_features)

        if self.transpose:
            input = torch.transpose(input, 1, 2)

        style = self.affine(style).unsqueeze(2)  # (batch_size, 2*in_features, 1)
        gamma, beta = style.chunk(2, dim=1)  # (batch_size, in_features, length), (batch_size, in_features, 1)

        out = self.norm(input)
        out = gamma * out + beta  # (batch_size, in_features, length)

        if self.transpose:
            out = torch.transpose(out, 1, 2)

        return out


class PixelwiseAdaptiveInstanceNorm2d(nn.Module):
    # normalize over channels
    def __init__(self, in_channels, style_features):
        super().__init__()

        self.norm = nn.LayerNorm(in_channels, elementwise_affine=False)

        self.affine = nn.Linear(in_features=style_features, out_features=in_channels * 2)

        self.affine.bias.data[:in_channels] = 1  # initial gamma is 1
        self.affine.bias.data[in_channels:] = 0

    def forward(self, input, style):
        # input : (batch_size, channels, height, width)
        # style : (batch_size, style_features)

        style = self.affine(style).unsqueeze(2).unsqueeze(3)  # (batch_size, 2*in_channels, 1, 1)
        gamma, beta = style.chunk(2, dim=1)

        x = input
        x = torch.transpose(x, 1, 3)  # (batch_size, width, height,  in_channels)
        x = self.norm(x)
        x = torch.transpose(x, 1, 3)  # (batch_size, in_channels, height, width)

        out = gamma * x + beta  # (batch_size, in_channels, height, width)

        return out


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, in_channels, style_features, modulated=False):
        super().__init__()

        self.modulated = modulated
        if modulated:
            self.norm = None
        else:
            self.norm = nn.InstanceNorm2d(in_channels)

        self.affine = nn.Linear(in_features=style_features, out_features=in_channels * 2)

        self.affine.bias.data[:in_channels] = 1  # initial gamma is 1
        self.affine.bias.data[in_channels:] = 0

    def forward(self, input, style):
        style = self.affine(style).unsqueeze(2).unsqueeze(3)  # (batch_size, 2*in_channels, 1, 1)
        gamma, beta = style.chunk(2, dim=1)

        x = input
        if self.norm is not None:
            x = self.norm(x)
        out = gamma * x + beta  # (batch_size, in_channels, height, width)

        return out


### Dirac-GAN
class R1(nn.Module):
    """
    Implementation of the R1 GAN regularization.
    """

    def __init__(self):
        """
        Constructor method
        """
        # Call super constructor
        super(R1, self).__init__()

    def forward(self, prediction_real: torch.Tensor, real_sample: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the regularization
        :param prediction_real: (torch.Tensor) Prediction of the discriminator for a batch of real images
        :param real_sample: (torch.Tensor) Batch of the corresponding real images
        :return: (torch.Tensor) Loss value
        """
        # Calc gradient
        grad_real = torch.autograd.grad(outputs=prediction_real.sum(), inputs=real_sample, create_graph=True)[0]
        # Calc regularization
        regularization_loss: torch.Tensor = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
        return regularization_loss


### BigGAN
def ortho(model, strength=1e-4, blacklist=[]):
  with torch.no_grad():
    for param in model.parameters():
      # Only apply this to parameters with at least 2 axes, and not in the blacklist
      if len(param.shape) < 2 or any([param is item for item in blacklist]):
        continue
      w = param.view(param.shape[0], -1)
      grad = (2 * torch.mm(torch.mm(w, w.t()) 
              * (1. - torch.eye(w.shape[0], device=w.device)), w))
      param.grad.data += strength * grad.view(param.shape)
