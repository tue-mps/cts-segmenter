import numpy as np
import math
import torch
import torch.nn
import torch.nn.functional as F
from einops import rearrange

# for evaluation of all types of policy generation approaches, the output of this function is not bounded by the
# grouping schedules, all consistent potential 2*2 patches will be True.
def policy_gt_gen_for_eval(seg_map, patch_size):
    B, H, W = seg_map.size()
    assert B == 1
    seg_map = seg_map[0]
    group_if_all_ignore = True  # If true, a patch is grouped if it contains only the ignore label 0 (otherwise set to ignore)
    # ignore_if_one_class_plus_ignore = False  # If true, a patch is ignored if it contains one class + ignore label 0 (otherwise set to 'don't group')
    # ignore_if_one_class_plus_ignore_at_edges = True
    patch_groups = np.zeros((H//patch_size//2, H//patch_size//2), dtype=np.uint8)
    for i in range(patch_groups.shape[0]):
        for j in range(patch_groups.shape[1]):
            patch = seg_map[i * patch_size*2:i * patch_size*2 + patch_size*2, j * patch_size*2:j * patch_size*2 + patch_size*2]
            unique = np.unique(patch)
            patch_groups[i, j] = unique.shape[0] == 1

            if not group_if_all_ignore:
                if unique.shape[0] == 1:
                    if np.unique(patch)[0] == 0:  # 0 is the ignore label in GT
                        patch_groups[i, j] = 255  # 255 is the ignore label in the new patch grouping GT

            # if ignore_if_one_class_plus_ignore:
            #     if unique.shape[0] == 2:
            #         if 0 in unique:
            #             patch_groups[i, j] = 255
            #
            # if ignore_if_one_class_plus_ignore_at_edges:
            #     if i in [0, H//patch_size//2] or j in [0, W//patch_size//2]:
            #         if unique.shape[0] == 2:
            #             if 0 in unique:
            #                 patch_groups[i, j] = 255

    return None, patch_groups


def policy_indices_by_policynet_pred(images, patch_size, policy_schedule, policynet_pred):
    B, C, H, W = images.size()
    assert H % patch_size == 0 and W % patch_size == 0
    clue = policynet_pred['logits'].to(torch.float)

    base_grid_H, base_grid_W = H // (patch_size * 2), W // (patch_size * 2)
    num_scale_1, num_scale_2 = policy_schedule

    group_scores = torch.softmax(clue, dim=1)[:, 1]

    selected_msk_scale_1_per_img = list()
    selected_msk_scale_2_per_img = list()

    group_scores = rearrange(group_scores, 'b h w-> b (h w)')
    group_scores_sorted, group_scores_idx = torch.sort(group_scores, descending=True, dim=1)

    for b in range(B):
        grouped_mask = torch.zeros((base_grid_H, base_grid_W)).bool()
        grouped_mask = rearrange(grouped_mask, 'h w-> (h w)')

        group_scores_idx_selected = group_scores_idx[b, 0: num_scale_2]

        grouped_mask[group_scores_idx_selected] = True
        grouped_mask = grouped_mask.view((base_grid_H, base_grid_W))

        selected_msk_scale_2_per_img.append(grouped_mask)

        grouped_mask_large = F.interpolate(grouped_mask.float().unsqueeze(0).unsqueeze(0),
                                           size=(base_grid_H*2, base_grid_W*2),
                                           mode='nearest').squeeze(0).squeeze(0).bool()
        selected_msk_scale_1_per_img.append(torch.logical_not(grouped_mask_large))

    selected_msk_scale_1 = torch.stack(selected_msk_scale_1_per_img, dim=0)
    selected_msk_scale_2 = torch.stack(selected_msk_scale_2_per_img, dim=0)

    assert num_scale_1 == torch.div(torch.sum(selected_msk_scale_1), B, rounding_mode='floor')
    assert num_scale_2 == torch.div(torch.sum(selected_msk_scale_2), B, rounding_mode='floor')

    return selected_msk_scale_1, selected_msk_scale_2


def policy_indices_no_sharing(images, patch_size):
    B, C, H, W = images.size()
    assert H % patch_size ==0 and W % patch_size == 0
    base_grid_H, base_grid_W = H // patch_size, W // patch_size

    selected_msk_scale_2 = torch.zeros((B, base_grid_H // 2, base_grid_W // 2), dtype=torch.bool)
    selected_msk_scale_1 = torch.ones((B, base_grid_H, base_grid_W), dtype=torch.bool)

    return (selected_msk_scale_1, selected_msk_scale_2)


def images_to_patches(images, patch_size, policy_indices):
    # group_quota should be a tuple of integers (base, scale_2, scale_4, potentially more)
    B, C, H, W = images.size()
    assert H % patch_size ==0 and W % patch_size == 0
    base_grid_H, base_grid_W = H // patch_size, W // patch_size
    
    # prepare all possible patches at different scale
    # base level, no grouping and rescale
    patch_scale_1 = rearrange(images, 'b c (gh ps_h) (gw ps_w) -> b gh gw c ps_h ps_w', gh=base_grid_H, gw=base_grid_W)
    scale_value_1 = torch.ones([B, base_grid_H, base_grid_W, 1])
    patch_code_scale_1  = torch.cat([scale_value_1,
                                torch.linspace(0, base_grid_H-1, base_grid_H).view(-1, 1, 1).expand_as(scale_value_1),
                                torch.linspace(0, base_grid_W-1, base_grid_W).view(1, -1, 1).expand_as(scale_value_1)], dim=3)

    # group 2*2 patches and resize them to the defined patch size
    patch_scale_2 = rearrange(F.interpolate(images, scale_factor=0.5, mode='bilinear',
                                            align_corners=False, recompute_scale_factor=False),
                              'b c (gh ps_h) (gw ps_w) -> b gh gw c ps_h ps_w', gh=base_grid_H//2, gw=base_grid_W//2)
    patch_code_scale_2 = torch.clone(patch_code_scale_1)[:, ::2, ::2, :]
    patch_code_scale_2[:, :, :, 0] = 2

    (selected_msk_scale_1, selected_msk_scale_2) = policy_indices

    patch_code_scale_2_selected = patch_code_scale_2[selected_msk_scale_2]
    patch_code_scale_2_selected = rearrange(patch_code_scale_2_selected, '(b np) c -> b np c', b=B)
    patch_scale_2_selected = patch_scale_2[selected_msk_scale_2]
    patch_scale_2_selected = rearrange(patch_scale_2_selected, '(b np) c h w -> b np c h w', b=B)

    patch_code_scale_1_selected = patch_code_scale_1[selected_msk_scale_1]
    patch_code_scale_1_selected = rearrange(patch_code_scale_1_selected, '(b np) c -> b np c', b=B)
    patch_scale_1_selected = patch_scale_1[selected_msk_scale_1]
    patch_scale_1_selected = rearrange(patch_scale_1_selected, '(b np) c h w -> b np c h w', b=B)

    patches_total = torch.cat([patch_scale_1_selected, patch_scale_2_selected], dim=1)
    patch_code_total = torch.cat([patch_code_scale_1_selected, patch_code_scale_2_selected], dim=1)

    patches_total = rearrange(patches_total, 'b np c ps_h ps_w -> b c ps_h (np ps_w)')

    return patches_total, patch_code_total


def patches_to_images(patches, policy_code, grid_size):
    batch_size, dim_patch, patch_size, ps_times_num_patch = patches.size()
    num_patch = ps_times_num_patch // patch_size
    num_grid_h, num_grid_w = grid_size # grid size is based on the original base patch size
    patches = rearrange(patches, 'b c hp (np wp) ->b np c hp wp', np=num_patch)
    num_total_grid = num_grid_h * num_grid_w

    scale_value = policy_code[:, :, 0]
    grid_coords = policy_code[:, :, 1:]

    # process patches that stay at the original scale
    scale_1_idx = scale_value == 1

    patch_scale_1 = patches[scale_1_idx]
    patch_scale_1 = rearrange(patch_scale_1, '(b np) c h w -> b np c h w', b=batch_size)
    grid_coord_1 = grid_coords[scale_1_idx].unsqueeze(1)
    grid_coord_1 = rearrange(grid_coord_1, '(b np) ng c -> b (np ng) c', b=batch_size)

    # process patches that are downsize by the factor of 2
    scale_2_idx = scale_value == 2
    # transform 1*1 original grid coord to 2*2 (because it will be upsampled by the factor of 2)
    grid_coord_2 = grid_coords[scale_2_idx].unsqueeze(1) # coords are stacked across batch dim
    grid_coord_2 = torch.cat([grid_coord_2, grid_coord_2 + torch.tensor([[0, 1]]),
                              grid_coord_2 + torch.tensor([[1, 0]]), grid_coord_2 + torch.tensor([[1, 1]])], dim=1)
    grid_coord_2 = rearrange(grid_coord_2, '(b np) ng c -> b (np ng) c', b=batch_size) # create batch dim

    # find and enlarege the patches that are downsize before, and break it into 4 pieces
    patch_scale_2 = patches[scale_2_idx]
    patch_scale_2 = F.interpolate(patch_scale_2, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=False) # stacked across batch dim
    patch_scale_2 = rearrange(patch_scale_2, 'b c (h1 h) (w1 w) -> b (h1 w1) c h w', h1=2, w1=2)
    patch_scale_2 = rearrange(patch_scale_2, '(b np) ng c ps_h ps_w  -> b (np ng) c ps_h ps_w', b=batch_size)

    # combine all the restored patches (of the same size and scale) together, total size should be 'num_total_grid'
    # even one shuffle the patches and grid value, it will be sorted later anyway
    patches_uni = torch.cat([patch_scale_1, patch_scale_2], dim=1)
    grid_coord_uni = torch.cat([grid_coord_1, grid_coord_2], dim=1)

    # sort the patches according to grid universal positional value, different samples batch-level have offset values
    grid_uni_value = grid_coord_uni[:, :, 0] * num_grid_w + grid_coord_uni[:, :, 1]
    batch_offset = torch.linspace(0, batch_size-1, batch_size).view(batch_size, 1).expand_as(grid_uni_value)*num_total_grid
    grid_sort_global = batch_offset + grid_uni_value
    grid_sort_global = grid_sort_global.view(-1)
    patch_uni_global = rearrange(patches_uni, 'b np c h w -> (b np) c h w')
    indices_global = torch.argsort(grid_sort_global)
    patch_uni_global = patch_uni_global[indices_global]

    patch_uni_global = rearrange(patch_uni_global, '(b np) c h w  -> b np c h w', b=batch_size)
    images = rearrange(patch_uni_global, 'b (hp wp) c h w -> b c (hp h) (wp w)', hp=num_grid_h, wp=num_grid_w)

    return images
