import torch
import timm
import torch.nn as nn


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model('efficientnet_lite0', pretrained=True, features_only=True)
        self.head = nn.Conv2d(320, 2, 1)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, labels=None):
        feats = self.backbone(x)
        x = self.head(feats[-1])

        out_dict = {'logits': x}

        if self.training:
            loss = self.loss(x, labels)
            out_dict['loss'] = loss

        return out_dict

    def loss(self, logits, labels):
        loss = self.criterion(logits, labels)

        return loss

