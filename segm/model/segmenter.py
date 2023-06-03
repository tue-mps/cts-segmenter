import torch
import torch.nn as nn
import torch.nn.functional as F

from segm.model.utils import padding, unpadding
from segm.model.policy_net import PolicyNet


class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
        model_cfg,
    ):
        super().__init__()
        self.n_cls = n_cls
        if encoder is not None:
            self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder
        self.model_cfg = model_cfg
        self.policy_net = None

        self.policynet_ckpt = self.model_cfg["policynet_ckpt"]

        if self.model_cfg["policy_method"] == 'policy_net':
            self.policy_net = PolicyNet()
            if self.policynet_ckpt is not None:
                try:
                    self.policy_net.load_state_dict(torch.load(self.policynet_ckpt))
                except:
                    Warning('PolicyNet checkpoint path {} cannot be found, so cannot be loaded. This is problematic during training, because the network requires the PolicyNet to be initialized. It should not be a problem during inference, when the PolicyNet weights are embedded in the network weights, and should be loaded later. Verify the quality of the predictions to be sure.')
            else:
                raise Warning('PolicyNet checkpoint path is None, so the weights of the policynet are intialized randomly. Check that this is desired behavior.')
            for p in self.policy_net.parameters():
                p.requires_grad = True

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, im):
        if self.policy_net is not None:
            policy_pred = self.policy_net(im)
        else:
            policy_pred = None

        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        x, policy_code, policy_indices = self.encoder(im, policy_pred, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        masks = self.decoder(x, (H, W), policy_code)
        masks = F.interpolate(masks, size=(H, W), mode="bilinear", align_corners=False)
        masks = unpadding(masks, (H_ori, W_ori))

        return masks, policy_pred, policy_indices

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)
