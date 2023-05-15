import torch
import torch.nn as nn
from fastai.layers import *
from torch import Tensor


def conv_block(c_in, c_out, ks, num_groups=None, **conv_kwargs):
  if not num_groups: num_groups = int(c_in / 2) if c_in % 2 == 0 else None
  return nn.Sequential(nn.GroupNorm(num_groups, c_in),
                       nn.ReLU(),
                       nn.Conv3d(c_in, c_out, ks, **conv_kwargs))


def upsize(c_in, c_out, ks=1, scale=2):
  return nn.Sequential(nn.Conv3d(c_in, c_out, ks),
                       nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=True))


class Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv3d(1, 32, 3, stride=1, padding=1)
    self.conv_block1 = conv_block(32, 64, 3, num_groups=8, stride=2, padding=1)
    self.conv_block2 = conv_block(64, 64, 3, num_groups=8, stride=1, padding=1)
    self.conv_block3 = conv_block(64, 128, 3, num_groups=8, stride=2, padding=1)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv_block1(x)
    x = self.conv_block2(x)
    x = self.conv_block3(x)
    return x


class Decoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.upsize1 = upsize(128, 64)
    self.upsize2 = upsize(64, 32)
    self.conv1 = nn.Conv3d(32, 1, 1)
    self.sigmoid1 = torch.nn.Sigmoid()

  def forward(self, x):
    x = self.upsize1(x)
    x = self.upsize2(x)
    x = self.conv1(x)
    x = self.sigmoid1(x)
    return x


class Autoencoder(nn.Module):
  def __init__(self):
    super(Autoencoder, self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()

  def forward(self, x):
    encode = self.encoder(x)
    decode = self.decoder(encode)
    return encode, decode


class SoftDiceLoss(nn.Module):
  "Soft dice loss based on a measure of overlap between prediction and ground truth"

  def __init__(self, epsilon=1e-6, c=3):
    super().__init__()
    self.epsilon = epsilon
    self.c = 3

  def forward(self, x: Tensor, y: Tensor):
    intersection = 2 * ((x * y).sum())
    union = (x ** 2).sum() + (y ** 2).sum()
    return 1 - ((intersection / (union + self.epsilon)) / self.c)


class KLDivergence(nn.Module):
  "KL divergence between the estimated normal distribution and a prior distribution"

  def __init__(self):
    super().__init__()

  def forward(self, z_mean: Tensor, z_log_var: Tensor):
    z_var = z_log_var.exp()
    return (1 / self.N) * ((z_mean ** 2 + z_var ** 2 - z_log_var ** 2 - 1).sum())


class L2Loss(nn.Module):
  "Measuring the `Euclidian distance` between prediction and ground truh using `L2 Norm`"

  def __init__(self):
    super().__init__()

  def forward(self, x: Tensor, y: Tensor):
    # return ((x - y) ** 2).sum()
    return torch.mean((x - y) ** 2)


class LambdaLR:
  def __init__(self, n_epochs, offset, decay_start_epoch):
    assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
    self.n_epochs = n_epochs
    self.offset = offset
    self.decay_start_epoch = decay_start_epoch

  def step(self, epoch):
    return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


if __name__ == "__main__":
  img3d = torch.randn(4, 1, 112, 112, 80)
  encoder3d = Encoder()
  decoder3d = Decoder()
  vec3d = encoder3d(img3d)
  new_img3d = decoder3d(vec3d)
  print(vec3d.shape)
  print(new_img3d.shape)

  autoencoder = Autoencoder()
  vec3d, new_img3d = autoencoder(img3d)
  print(vec3d.shape)
  print(new_img3d.shape)
