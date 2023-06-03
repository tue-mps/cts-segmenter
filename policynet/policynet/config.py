from dataclasses import dataclass


@dataclass
class Config:
    batch_size_train = 8
    batch_size_test = 10
    lr = 5e-5
    lr_momentum = 0.9
    weight_decay = 1e-3
    gt_dir_ade20k = "ade20k/ADEChallengeData2016/"
    gt_dir_cityscapes = "cityscapes/"
    gt_dir_pascal = "pcontext/VOCdevkit/VOC2010/"
    num_iterations = 50000
    log_iterations = 100
    summary_iterations = 20
    eval_iterations = 10000
    hflip = True
    optimizer = 'AdamW'
    enable_cuda = True
    logdir = "./logdir/"
    dataset = 'ade20k'
    num_workers = 16
