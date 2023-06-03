# Content-aware Token Sharing Policy Network

Part of "Content-aware Token Sharing for Efficient Semantic Segmentation with Vision Transformers", Chenyang Lu, Daan de Geus, Gijs Dubbelman, _CVPR_ 2023.

## Policy network
This policy network (PolicyNet) is trained to identify what superpatches (2x2 neighboring patches) contain the same semantic class, and can therefore share a single token. It achieves this by turning the problem into a binary classification problem, see below.

## Training
The PolicyNet can be trained for each dataset that has semantic labels. In this code, we support ADE20K, Cityscapes and Pascal-Context. To train the network for ADE20K, use the following command:

```bash
cd policynet

python train.py \
  --exp_name 'policynet_efficientnet_ade20k' \
  --dataset 'ade20k'
```

For Pascal-Context, use `--dataset 'pcontext'`. For Cityscapes, use `--dataset 'cityscapes'`. Further hyperparameters can be specified in the [config](policynet/config.py).

## Using learned policy for token sharing
When the policy network is trained, the `model.pth` checkpoint will be saved in the `logdir` defined in `config.py`. The learned model can now be used to determine the token sharing policy for ViT-based segmentation networks. In the code that we provide for [CTS with Segmenter](../README.md), we provide the path PolicyNet checkpoint so that it can be loaded into the PolicyNet that is used within the model. For more information, see the [README](../README.md).