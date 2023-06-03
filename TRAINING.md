# Training

Below, we provide the training commands for training Segmenter + CTS on the following datasets:
* [ADE20K](#ade20k)
* [Cityscapes](#cityscapes)
* [Pascal-Context](#pascal-context)

Different backbone possibilities:
* ViT-Ti: `vit_tiny_patch16_384`
* ViT-S: `vit_small_patch16_384`
* ViT-B: `vit_base_patch16_384`
* ViT-L: `vit_large_patch16_384`

For more configuration options, see [segm/config.yml](segm/config.yml) and [segm/train.py](segm/train.py).

## ADE20K

With CTS, 30% token reduction (612 not shared, 336 shared), and ViT-S backbone:

```bash
python -m segm.train --log-dir runs/ade20k_segmenter_small_patch16_cts_612_103 \
                     --dataset ade20k \
                     --backbone vit_small_patch16_384 \
                     --decoder mask_transformer \
                     --policy-method policy_net \
                     --num-tokens-notshared 612 \
                     --num-tokens-shared 103 \
                     --policynet-ckpt 'policynet/logdir/policynet_efficientnet_ade20k/model.pth'
```

Without CTS, 0% token reduction (1024 not shared), and ViT-S backbone:

```bash
python -m segm.train --log-dir runs/ade20k_segmenter_small_patch16_nosharing_1024_0 \
                     --dataset ade20k \
                     --backbone vit_small_patch16_384 \
                     --decoder mask_transformer \
                     --policy-method no_sharing \
                     --num-tokens-notshared 1024 \
                     --num-tokens-shared 0
```

## Cityscapes


With CTS, 44% token reduction (960 not shared, 336 shared), and ViT-S:
```bash
python -m segm.train --log-dir runs/cityscapes_segmenter_small_patch16_cts_960_336 \
                     --dataset cityscapes \
                     --backbone vit_small_patch16_384 \
                     --decoder mask_transformer \
                     --policy-method policy_net \
                     --num-tokens-notshared 960 \
                     --num-tokens-shared 336 \
                     --policynet-ckpt 'policynet/logdir/policynet_efficientnet_cityscapes/model.pth'
```

Without CTS, 0% token reduction (2304 not shared), and ViT-S:
```bash
python -m segm.train --log-dir runs/cityscapes_segmenter_small_patch16_nosharing_2304_0 \
                     --dataset cityscapes \
                     --backbone vit_small_patch16_384 \
                     --decoder mask_transformer \
                     --policy-method no_sharing \
                     --num-tokens-notshared 2304 \
                     --num-tokens-shared 0 
```

## Pascal Context

With CTS, 30% token reduction (540 not shared, 90 shared), and ViT-S:
```bash
python -m segm.train --log-dir runs/pcontext_segmenter_small_patch16_cts_540_90 \
                     --dataset pascal_context \
                     --backbone vit_small_patch16_384 \
                     --decoder mask_transformer \
                     --policy-method policy_net \
                     --num-tokens-notshared 540 \
                     --num-tokens-shared 90 \
                     --policynet-ckpt 'policynet/logdir/policynet_efficientnet_pcontext/model.pth'
```

Without CTS, 0% token reduction (900 not shared), and ViT-S:
```bash
python -m segm.train --log-dir runs/pcontext_segmenter_small_patch16_nosharing_900_0 \
                     --dataset pascal_context \
                     --backbone vit_small_patch16_384 \
                     --decoder mask_transformer \
                     --policy-method no_sharing \
                     --num-tokens-notshared 900 \
                     --num-tokens-shared 0 
```
