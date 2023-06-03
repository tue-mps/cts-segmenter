import argparse
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch

from policynet.config import Config
from policynet.data_ade20k import TokenSharingPolicyDatasetADE20K
from policynet.data_pcontext import TokenSharingPolicyDatasetPContext
from policynet.data_cityscapes import TokenSharingPolicyDatasetCityscapes
from policynet.net import PolicyNet


def train(args):
  # Configuration settings
  cfg = Config()

  print(args)

  if args.num_iterations:
    cfg.num_iterations = args.num_iterations
  if args.batch_size_train:
    cfg.batch_size_train = args.batch_size_train
  if args.lr:
    cfg.lr = args.lr
  if args.exp_name:
    logdir = os.path.join(cfg.logdir, args.exp_name)
  else:
    logdir = os.path.join(cfg.logdir, 'experiment')
  if not os.path.exists(logdir):
    os.makedirs(logdir, exist_ok=False)
  cfg.hflip = bool(args.hflip)
  cfg.dataset = args.dataset
  cfg.optimizer = args.optimizer
  txt_write = open(os.path.join(logdir, 'log.txt'), 'w')

  writer = SummaryWriter(log_dir=logdir)

  # Load dataset
  if cfg.dataset == 'ade20k':
    dataset = TokenSharingPolicyDatasetADE20K
  elif cfg.dataset == 'cityscapes':
    dataset = TokenSharingPolicyDatasetCityscapes
  elif cfg.dataset == 'pcontext':
    dataset = TokenSharingPolicyDatasetPContext
  else:
    raise NotImplementedError

  train_dataset = dataset(cfg,
                          split='training',
                          mode='training',
                          apply_hflip=cfg.hflip)

  train_dataloader = DataLoader(train_dataset,
                                batch_size=cfg.batch_size_train,
                                shuffle=True,
                                num_workers=cfg.num_workers,
                                drop_last=True)

  eval_dataset = dataset(cfg,
                         split='validation',
                         mode='validation',
                         apply_hflip=False)

  eval_dataloader = DataLoader(eval_dataset,
                               batch_size=cfg.batch_size_test,
                               shuffle=True,
                               num_workers=cfg.num_workers,
                               drop_last=False)

  # Initialize network
  model = PolicyNet()
  model.train()
  if cfg.enable_cuda:
    model = model.cuda()

  # Initialize optimizer
  if cfg.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.lr_momentum, weight_decay=cfg.weight_decay)
  elif cfg.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
  elif cfg.optimizer == 'AdamW':
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
  else:
    raise ValueError('Only SGD, Adam or AdamW optimizers implemented')

  # Loop over images
  running_loss = 0.0
  running_loss_tb = 0.0
  i = 0
  print("Starting training...")
  while i < cfg.num_iterations:
    for (imgs, labels, image_ids) in train_dataloader:
      if i > cfg.num_iterations:
        break
      if cfg.enable_cuda:
        imgs = imgs.cuda()
        labels = labels.cuda()

      optimizer.zero_grad()
      out = model(imgs, labels)
      loss = out['loss']

      running_loss += out['loss'].item()
      running_loss_tb += out['loss'].item()

      # Apply back-propagation
      loss.backward()
      # Take one step with the optimizer
      optimizer.step()

      if i % cfg.log_iterations == 0:
        if i == 0:
          loss_avg = running_loss
        else:
          loss_avg = running_loss/cfg.log_iterations
        print("Iteration {} - Loss: {}".format(i, round(loss_avg, 5)))
        txt_write.write("Iteration {} - Loss: {} \n".format(i, round(loss_avg, 5)))
        running_loss = 0.0

      if i % cfg.summary_iterations == 0:
        if i == 0:
          loss_summary = running_loss_tb
        else:
          loss_summary = running_loss_tb/cfg.summary_iterations
        writer.add_scalar('Loss/train', loss_summary, i)
        running_loss_tb = 0.0

      # Every eval_iterations steps, run evaluation on both the training and validation set
      if i % cfg.eval_iterations == 0 and not i == 0:
        print("Evaluating...")
        model.eval()
        correct = 0
        total = 0

        for data in eval_dataloader:
          images, labels, image_ids = data
          if cfg.enable_cuda:
            images, labels = images.cuda(), labels.cuda()
          outputs = model(images)
          logits = outputs['logits']
          # The class with the highest score is the prediction
          _, predicted = torch.max(logits.data, 1)
          not_ignore = labels != 255
          total += labels[not_ignore].numel()
          correct += (predicted[not_ignore] == labels[not_ignore]).sum().item()

        accuracy = round((correct/total*100), 1)
        writer.add_scalar('Accuracy/val', accuracy, i)
        print("Val accuracy at step {}: {}".format(i, accuracy))

        for data in train_dataloader:
          images, labels, image_ids = data
          if cfg.enable_cuda:
            images, labels = images.cuda(), labels.cuda()
          outputs = model(images)
          logits = outputs['logits']
          # The class with the highest score is the prediction
          _, predicted = torch.max(logits.data, 1)
          not_ignore = labels != 255
          total += labels[not_ignore].numel()
          correct += (predicted[not_ignore] == labels[not_ignore]).sum().item()

        accuracy = round((correct/total*100), 1)
        writer.add_scalar('Accuracy/train', accuracy, i)
        print("Train accuracy at step {}: {}".format(i, accuracy))

        model.train()
        print("Continuing training...")

      i += 1

  print("Finished training.")
  save_path = os.path.join(logdir, 'model.pth')
  torch.save(model.state_dict(), save_path)
  print("Saved trained model as {}.".format(save_path))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--num_iterations", type=int, help="number of iterations")
  parser.add_argument("--batch_size_train", type=int, help="training batch size")
  parser.add_argument("--hflip", type=int, help="apply horizontal flip (0 or 1)", default=1)
  parser.add_argument("--lr", type=float, help="learning rate")
  parser.add_argument("--exp_name", type=str, help="experiment name")
  parser.add_argument("--optimizer", type=str, help="optimizer", default="AdamW")
  parser.add_argument("--dataset", type=str, help="network", default="ade20k")

  args = parser.parse_args()
  if args.num_iterations:
    print("Num iterations:", args.num_iterations)
  if args.batch_size_train:
    print("Training batch size:", args.batch_size_train)
  if args.lr:
    print("Learning rate:", args.lr)
  if args.exp_name:
    print("Experiment name:", args.exp_name)
  train(args)
