import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders.la_heart import LAHeart, RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from networks.autoencoder import Autoencoder, SoftDiceLoss, L2Loss, LambdaLR
from networks.vnet_sdf_dkt import VNet, DiceLoss_DSV, CrossEntropyLoss_DSV
from utils import ramps, losses, losses2, metrics
from utils.util import compute_sdf

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2018LA_Seg_Training_Set/', help='data_path')
parser.add_argument('--exp', type=str, default='SDC-SSL', help='model_name')
parser.add_argument('--max_iterations', type=int, default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=16, help='labeled number')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--consistency_weight', type=float, default=0.2,
                    help='balance factor to control supervised loss and consistency loss')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--beta', type=float, default=0.3, help='balance factor to control regional and sdm loss')
parser.add_argument('--consistency_type', type=str, default="kl", help='consistency_type')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "_{}labels/".format(args.labelnum)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if not args.deterministic:
  cudnn.benchmark = True
  cudnn.deterministic = False
else:
  cudnn.benchmark = False  # True #
  cudnn.deterministic = True  # False #

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)


def get_current_consistency_weight(epoch):
  return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


if __name__ == "__main__":
  # make logger file
  if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)
  if os.path.exists(snapshot_path + '/code'):
    shutil.rmtree(snapshot_path + '/code')
  shutil.copytree('.', snapshot_path + '/code',
                  shutil.ignore_patterns(['.git', '__pycache__']))

  logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                      format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
  logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
  logging.info(str(args))


  def create_model(ema=False):
    # Network definition
    net = VNet(n_channels=1, n_classes=num_classes - 1,
               normalization='batchnorm', has_dropout=True)
    model = net.cuda()
    if ema:
      for param in model.parameters():
        param.detach_()
    return model


  model_seg = create_model()
  model_sdm = create_model()

  db_train = LAHeart(base_dir=train_data_path,
                     split='train',  # train/val split
                     transform=transforms.Compose([
                       RandomRotFlip(),
                       RandomCrop(patch_size),
                       ToTensor(),
                     ]))

  labelnum = args.labelnum  # default 16
  labeled_idxs = list(range(labelnum))
  unlabeled_idxs = list(range(labelnum, 80))
  batch_sampler = TwoStreamBatchSampler(
    labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)


  def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


  trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                           worker_init_fn=worker_init_fn)

  model_seg.train()
  model_sdm.train()

  optimizer_seg = optim.SGD(model_seg.parameters(), lr=base_lr,
                            momentum=0.9, weight_decay=0.0001)
  optimizer_sdm = optim.SGD(model_sdm.parameters(), lr=base_lr,
                            momentum=0.9, weight_decay=0.0001)
  ce_loss = BCEWithLogitsLoss()
  mse_loss = MSELoss()

  if args.consistency_type == 'mse':
    consistency_criterion = losses.softmax_mse_loss
  elif args.consistency_type == 'kl':
    consistency_criterion = losses.softmax_kl_loss
  else:
    assert False, args.consistency_type

  writer = SummaryWriter(snapshot_path + '/log')
  logging.info("{} itertations per epoch".format(len(trainloader)))

  iter_num = 0
  max_epoch = max_iterations // len(trainloader) + 1
  lr_ = base_lr
  best_performance = 0.0
  # autoencoder begin
  autoencoder = Autoencoder()
  autoencoder = autoencoder.cuda()
  autoencoder.train()
  shape_criterion = L2Loss()
  reconstruction_criterion = SoftDiceLoss()
  optimizer_autoencoder = torch.optim.Adam(autoencoder.parameters(), lr=0.0002, betas=(0.5, 0.999))
  lr_scheduler_autoencoder = torch.optim.lr_scheduler.LambdaLR(
    optimizer_autoencoder, lr_lambda=LambdaLR(max_epoch, 0, max_epoch // 2).step
  )
  # autoencoder end
  iterator = tqdm(range(max_epoch), ncols=70)
  for epoch_num in iterator:
    time1 = time.time()
    for i_batch, sampled_batch in enumerate(trainloader):
      time2 = time.time()
      # print('fetch data cost {}'.format(time2-time1))
      volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
      volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

      # DSV
      outputs_sdm, _, out_dsv_sdm = model_sdm(volume_batch)
      _, outputs_seg, out_dsv_seg = model_seg(volume_batch)
      softmask_seg = torch.sigmoid(outputs_seg)

      # calculate the loss
      with torch.no_grad():
        gt_sdm = compute_sdf(label_batch.cpu().numpy(), outputs_sdm.shape)
        gt_sdm = torch.from_numpy(gt_sdm).float().cuda()
      loss_sdm = torch.norm(outputs_sdm - gt_sdm, 1) / torch.numel(outputs_sdm)

      loss_ce = ce_loss(
        outputs_seg[:labeled_bs, 0, ...], label_batch[:labeled_bs].float())
      loss_dice = losses.dice_loss(
        softmask_seg[:labeled_bs, 0, :, :, :], label_batch[:labeled_bs] == 1)
      loss_seg = 0.5 * (loss_ce + loss_dice)

      sdm_to_mask = torch.sigmoid(-1500 * outputs_sdm)

      loss_sdm_dice = losses.dice_loss(
        sdm_to_mask[:labeled_bs, 0, :, :, :], label_batch[:labeled_bs] == 1)

      # boundary distance loss
      with torch.no_grad():
        pseudo_sdm = compute_sdf(sdm_to_mask.cpu().numpy(), outputs_sdm.shape)
        pseudo_sdm = torch.from_numpy(pseudo_sdm).float().cuda()
      bd_loss = losses2.boundary_loss(softmask_seg, pseudo_sdm)

      consistency_loss = torch.mean((sdm_to_mask - softmask_seg) ** 2) - 0.1 * bd_loss
      consistency_weight = get_current_consistency_weight(iter_num // 150)

      loss_dsv_sdm_dice = DiceLoss_DSV(out_dsv_sdm, label_batch, labeled_bs=labeled_bs)
      loss_dsv_sdm_ce = CrossEntropyLoss_DSV(out_dsv_sdm, label_batch, labeled_bs=labeled_bs)
      loss_dsv_sdm = 0.05 * (loss_dsv_sdm_dice + loss_dsv_sdm_ce)

      loss_dsv_seg_dice = DiceLoss_DSV(out_dsv_seg, label_batch, labeled_bs=labeled_bs)
      loss_dsv_seg_ce = CrossEntropyLoss_DSV(out_dsv_seg, label_batch, labeled_bs=labeled_bs)
      loss_dsv_seg = 0.05 * (loss_dsv_seg_dice + loss_dsv_seg_ce)

      # autoencoder
      pred_input = softmask_seg.clone()
      pred_input = pred_input.float()

      label_input = sdm_to_mask.clone()
      label_input = label_input.float()

      pred_shape_vector, pred_output = autoencoder(pred_input.detach())
      label_shape_vector, label_output = autoencoder(label_input.detach())
      loss_shape = shape_criterion(pred_shape_vector, label_shape_vector)
      loss_shape = 5.0 * loss_shape
      # autoencoder

      # model loss
      model_sdm_loss = loss_sdm_dice + consistency_weight * consistency_loss + loss_dsv_sdm + consistency_weight * loss_shape
      model_seg_loss = loss_seg + consistency_weight * consistency_loss + loss_dsv_seg + consistency_weight * loss_shape

      optimizer_seg.zero_grad()
      optimizer_sdm.zero_grad()

      model_seg_loss.backward(retain_graph=True)
      model_sdm_loss.backward(retain_graph=True)

      optimizer_seg.step()
      optimizer_sdm.step()

      # autoencoder
      optimizer_autoencoder.zero_grad()
      pred_input_ = softmask_seg.clone()
      pred_input_ = pred_input_.float()

      label_input_ = sdm_to_mask.clone()
      label_input_ = label_input_.float()

      loss_reconstruction_pred = reconstruction_criterion(pred_output, pred_input_.detach())
      loss_reconstruction_label = reconstruction_criterion(label_output, label_input_.detach())

      model_autoencoder_loss = loss_shape + loss_reconstruction_pred + loss_reconstruction_label
      model_autoencoder_loss.backward()
      optimizer_autoencoder.step()

      dc = metrics.dice(torch.argmax(
        softmask_seg[:labeled_bs], dim=1), label_batch[:labeled_bs])

      iter_num = iter_num + 1
      writer.add_scalar('info/lr', lr_, iter_num)
      writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)

      writer.add_scalar('loss/loss_seg', model_seg_loss, iter_num)
      writer.add_scalar('loss/loss_sdm', model_sdm_loss, iter_num)
      writer.add_scalar('loss/consistency_loss', consistency_loss, iter_num)
      writer.add_scalar('loss/bd_loss', bd_loss, iter_num)

      # model_seg
      writer.add_scalar('model_seg/model_seg_loss', model_seg_loss, iter_num)
      writer.add_scalar('model_seg/loss_seg', loss_seg, iter_num)
      writer.add_scalar('model_seg/loss_ce', loss_ce, iter_num)
      writer.add_scalar('model_seg/loss_dice', loss_dice, iter_num)

      # dsv_model_seg
      writer.add_scalar('dsv_model_seg/loss_dsv_seg', loss_dsv_seg, iter_num)
      writer.add_scalar('dsv_model_seg/loss_dsv_seg_dice', loss_dsv_seg_dice, iter_num)
      writer.add_scalar('dsv_model_seg/loss_dsv_seg_ce', loss_dsv_seg_ce, iter_num)

      # model_sdm
      writer.add_scalar('model_sdm/model_sdm_loss', model_sdm_loss, iter_num)
      writer.add_scalar('model_sdm/loss_sdm_dice', loss_sdm_dice, iter_num)

      # dsv_model_sdm
      writer.add_scalar('dsv_model_sdm/loss_dsv_sdm', loss_dsv_sdm, iter_num)
      writer.add_scalar('dsv_model_sdm/loss_dsv_sdm_dice', loss_dsv_sdm_dice, iter_num)
      writer.add_scalar('dsv_model_sdm/loss_dsv_sdm_ce', loss_dsv_sdm_ce, iter_num)

      # autoencoder
      writer.add_scalar('autoencoder/loss_shape', loss_shape, iter_num)
      writer.add_scalar('autoencoder/model_autoencoder_loss', model_autoencoder_loss, iter_num)
      writer.add_scalar('autoencoder/loss_reconstruction_pred', loss_reconstruction_pred, iter_num)
      writer.add_scalar('autoencoder/loss_reconstruction_label', loss_reconstruction_label, iter_num)

      logging.info(
        'iteration %d : loss_seg : %f, loss_sdm: %f, loss_consistency: %f, loss_dice: %f, loss_shape: %f, model_autoencoder_loss: %f' %
        (iter_num, model_seg_loss.item(), model_sdm_loss, consistency_loss.item(), loss_dice.item(), loss_shape.item(),
         model_autoencoder_loss.item()))

      if iter_num % 30 == 0:
        image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
          3, 0, 1, 2).repeat(1, 3, 1, 1)
        grid_image = make_grid(image, 5, normalize=True)
        writer.add_image('train/Image', grid_image, iter_num)

        image = softmask_seg[0, 0:1, :, :, 20:61:10].permute(
          3, 0, 1, 2).repeat(1, 3, 1, 1)
        grid_image = make_grid(image, 5, normalize=False)
        writer.add_image('train/Predicted_label', grid_image, iter_num)

        image = sdm_to_mask[0, 0:1, :, :, 20:61:10].permute(
          3, 0, 1, 2).repeat(1, 3, 1, 1)
        grid_image = make_grid(image, 5, normalize=False)
        writer.add_image('train/sdm_to_mask', grid_image, iter_num)

        image = outputs_sdm[0, 0:1, :, :, 20:61:10].permute(
          3, 0, 1, 2).repeat(1, 3, 1, 1)
        grid_image = make_grid(image, 5, normalize=False)
        writer.add_image('train/outputs_sdm', grid_image, iter_num)

        image = label_batch[0, :, :, 20:61:10].unsqueeze(
          0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
        grid_image = make_grid(image, 5, normalize=False)
        writer.add_image('train/GroundTruth_label', grid_image, iter_num)

        image = pred_input[0, 0:1, :, :, 20:61:10].permute(
          3, 0, 1, 2).repeat(1, 3, 1, 1)
        grid_image = make_grid(image, 5, normalize=False)
        writer.add_image('autoencoder/pred_input', grid_image, iter_num)

        image = label_input[0, 0:1, :, :, 20:61:10].permute(
          3, 0, 1, 2).repeat(1, 3, 1, 1)
        grid_image = make_grid(image, 5, normalize=False)
        writer.add_image('autoencoder/label_input', grid_image, iter_num)

        image = pred_output[0, 0:1, :, :, 20:61:10].permute(
          3, 0, 1, 2).repeat(1, 3, 1, 1)
        grid_image = make_grid(image, 5, normalize=False)
        writer.add_image('autoencoder/pred_output', grid_image, iter_num)

        image = label_output[0, 0:1, :, :, 20:61:10].permute(
          3, 0, 1, 2).repeat(1, 3, 1, 1)
        grid_image = make_grid(image, 5, normalize=False)
        writer.add_image('autoencoder/label_output', grid_image, iter_num)

      # change lr
      if iter_num % 2500 == 0:
        lr_ = base_lr * 0.1 ** (iter_num // 2500)
        for param_group in optimizer_seg.param_groups:
          param_group['lr'] = lr_
        for param_group in optimizer_sdm.param_groups:
          param_group['lr'] = lr_

      if iter_num % 1000 == 0:
        save_mode_path = os.path.join(
          snapshot_path, 'iter_' + str(iter_num))
        torch.save(model_seg.state_dict(), save_mode_path + '_seg.pth')
        logging.info("save model to {}".format(save_mode_path))

      if iter_num >= max_iterations:
        break
      time1 = time.time()
    if iter_num >= max_iterations:
      iterator.close()
      break
    lr_scheduler_autoencoder.step()
  writer.close()
