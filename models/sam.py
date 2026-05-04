import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from .mmseg.models.sam import ImageEncoderViT, MaskDecoder, TwoWayTransformer

logger = logging.getLogger(__name__)
from .iou_loss import IOU
from typing import Any, Optional, Tuple
from .sam2.modeling.backbones.image_encoder import ImageEncoder
from .sam2.modeling.backbones.hieradet import Hiera
from .sam2.modeling.backbones.image_encoder import FpnNeck
from torch.nn.init import trunc_normal_

def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.BatchNorm2d:
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)

class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss

def _iou_loss(pred, target, valid=None):
    """Soft IOU loss. valid is an optional bool mask (B,H,W) of pixels to include."""
    pred = torch.sigmoid(pred)  # (B, C, H, W)
    # Build one-hot target: (B, C, H, W)
    B, C, H, W = pred.shape
    t = target.clone()
    if valid is not None:
        t[~valid] = 0   # treat ignored pixels as background for IOU
    t_onehot = torch.zeros_like(pred)
    t_onehot.scatter_(1, t.unsqueeze(1).clamp(0, C - 1), 1.0)
    if valid is not None:
        mask = valid.unsqueeze(1).float()
        pred = pred * mask
        t_onehot = t_onehot * mask
    inter = (pred * t_onehot).sum(dim=(2, 3))
    union = (pred + t_onehot).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / (union + 1e-6))
    return iou.mean()

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W


@register('sam')
class SAM(nn.Module):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None,
        num_classes = None,
        sam_mask_decoder_extra_args=None,
        use_high_res_features_in_sam=True,
        directly_add_no_mem_embed=True,
        iou_prediction_use_sigmoid=False,
        pred_obj_scores: bool = True,
        pred_obj_scores_mlp: bool = False,
        use_obj_ptrs_in_encoder=False,
        fixed_no_obj_ptr: bool = False,
        use_multimask_token_for_obj_ptr: bool = True,
                 ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = encoder_mode['embed_dim']
        em = encoder_mode  # shorthand
        self.image_encoder = ImageEncoder(
            trunk=Hiera(
                img_size=em.get('img_size', 1024),
                embed_dim=em.get('embed_dim', 144),
                # backbone_channel_list is coarse->fine [1152,576,288,144];
                # Hiera's embed_dims expects fine->coarse [144,288,576,1152]
                embed_dims=list(reversed(em.get('backbone_channel_list', [1152, 576, 288, 144]))),
                num_heads=em.get('num_heads', 2),
                stages=tuple(em.get('stages', [2, 6, 36, 4])),
                window_spec=tuple(em.get('window_spec', [8, 4, 16, 8])),
                global_att_blocks=tuple(em.get('global_att_blocks', [23, 33, 43])),
                window_pos_embed_bkg_spatial_size=tuple(em.get('window_pos_embed_bkg_spatial_size', [7, 7])),
                # adapter / prompt settings
                scale_factor=em.get('scale_factor', 32),
                prompt_type=em.get('prompt_type', 'highpass'),
                tuning_stage=str(em.get('tuning_stage', '1234')),
                input_type=em.get('input_type', 'fft'),
                freq_nums=em.get('freq_nums', 0.25),
                handcrafted_tune=em.get('handcrafted_tune', True),
                embedding_tune=em.get('embedding_tune', True),
                adaptor=em.get('adaptor', 'adaptor'),
            ),
            neck=FpnNeck(
                d_model=em.get('d_model', 256),
                backbone_channel_list=em.get('backbone_channel_list', [1152, 576, 288, 144]),
                fpn_top_down_levels=em.get('fpn_top_down_levels', [2, 3]),
                fpn_interp_model=em.get('fpn_interp_model', 'nearest'),
            ),
            img_size=em.get('img_size', 1024),
        )
        # _bb_feat_sizes and image_embedding_size are computed dynamically
        # at runtime from the actual input tensor (see _compute_feat_sizes).
        # Initialise with inp_size as a sensible default; overwritten each forward pass.
        self._bb_feat_sizes = [(inp_size//4,)*2, (inp_size//8,)*2, (inp_size//16,)*2]
        self.image_embedding_size = inp_size // 16
        self.prompt_embed_dim = encoder_mode['prompt_embed_dim']
        self.class_prompt_embed = nn.Embedding(num_classes - 1, self.prompt_embed_dim)
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid
        self.pred_obj_scores = pred_obj_scores
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
            trunc_normal_(self.no_obj_ptr, std=0.02)
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=1,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            num_classes=num_classes,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
        )

        if 'evp' in encoder_mode['name']:
            for k, p in self.encoder.named_parameters():
                if "prompt" not in k and "mask_decoder" not in k and "prompt_encoder" not in k:
                    p.requires_grad = False

        self.loss_mode = loss
        if self.loss_mode == 'bce':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()

        elif self.loss_mode == 'bbce':
            self.criterionBCE = BBCEWithLogitLoss()

        elif self.loss_mode == 'iou':
            # ignore_index=255 handles pixels masked by ignore_bg=true in the dataloader
            # and also naturally handles class-imbalance by excluding background
            self.criterionBCE = torch.nn.CrossEntropyLoss(ignore_index=255)
            self.criterionIOU = IOU()

        self.pe_layer = PositionEmbeddingRandom(encoder_mode['prompt_embed_dim'] // 2)
        self.inp_size = inp_size
        self.no_mask_embed = nn.Embedding(1, encoder_mode['prompt_embed_dim'])

        # FIX 1: Register no_mem_embed as a proper trainable parameter instead of
        # creating a throwaway nn.Parameter inside forward() on every step.
        # Previously this was `torch.nn.Parameter(torch.zeros(...))` inside forward/infer,
        # which is NOT registered in the module and therefore never trained.
        if self.directly_add_no_mem_embed:
            self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, 256))

    def _compute_feat_sizes(self, inp: torch.Tensor):
        """Compute FPN spatial sizes and image_embedding_size from actual input shape.
        Called at the start of forward() and infer() so any input resolution works.
        Hiera: patch_embed stride=4, then 3x stride-2 pool -> levels at //4,//8,//16,//32.
        scalp=1 removes the coarsest (//32), leaving 3 levels: //4, //8, //16.
        """
        h = inp.shape[-2]
        s0, s1, s2 = h // 4, h // 8, h // 16
        self._bb_feat_sizes      = [(s0, s0), (s1, s1), (s2, s2)]
        self.image_embedding_size = s2

    def set_input(self, input, gt_mask):
        self.input = input.to(self.device)
        self.gt_mask = gt_mask.to(self.device)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _encode_image(self, input: torch.Tensor):
        """Shared image encoding logic used by both forward() and infer()."""
        features = self.image_encoder(input)
        if self.use_high_res_features_in_sam:
            features["backbone_fpn"][0] = self.mask_decoder.conv_s0(
                features["backbone_fpn"][0]
            )
            features["backbone_fpn"][1] = self.mask_decoder.conv_s1(
                features["backbone_fpn"][1]
            )
        features = features.copy()
        assert len(features["backbone_fpn"]) == len(features["vision_pos_enc"])
        assert len(features["backbone_fpn"]) >= self.num_feature_levels

        feature_maps      = features["backbone_fpn"][-self.num_feature_levels:]
        vision_pos_embeds = features["vision_pos_enc"][-self.num_feature_levels:]

        feat_sizes        = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        vision_feats      = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        # FIX 2: Use the registered self.no_mem_embed parameter instead of creating
        # a new unregistered nn.Parameter on every call (which was never trained).
        if self.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.no_mem_embed.to(vision_feats[-1].device)

        bs = input.shape[0]
        feats = [
            feat.permute(1, 2, 0).reshape(bs, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
        ][::-1]

        return {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

    def forward(self):
        bs = self.input.shape[0]
        self._compute_feat_sizes(self.input)

        # Class-conditioned sparse embeddings: one learnable token per foreground class.
        # These are the prompts that teach the decoder to separate categories.
        cls_tokens = self.class_prompt_embed.weight.unsqueeze(0).expand(bs, -1, -1)
        sparse_embeddings = cls_tokens  # (bs, num_classes-1, prompt_embed_dim)

        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        self._features = self._encode_image(self.input)
        high_res_features = self._features["high_res_feats"]

        # FIX 3: Use multimask_output=True in forward() to match infer().
        # With multimask_output=False the decoder collapses to 1 output channel,
        # which is then squeeze(dim=1)-d away in postprocess_masks — the model
        # never produces one logit map per class, so cross-class competition is broken.
        low_res_masks, iou_predictions, sam_output_tokens, object_score_logits = self.mask_decoder(
            image_embeddings=self._features["image_embed"],
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,   # <-- was False; must match infer()
            repeat_image=False,
            high_res_features=high_res_features,
        )

        # Debug: fires only on the first forward pass so you can verify decoder output shape.
        if not getattr(self, '_shape_printed', False):
            print(f'[SAM] decoder low_res_masks raw shape : {tuple(low_res_masks.shape)}')
            masks_dbg = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
            print(f'[SAM] pred_mask shape after postprocess: {tuple(masks_dbg.shape)}')
            print(f'[SAM] num_classes (from class_prompt_embed): {self.class_prompt_embed.num_embeddings + 1}')
            self._shape_printed = True
            self.pred_mask = masks_dbg
            return

        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        self.pred_mask = masks

    def infer(self, input):
        bs = input.shape[0]
        self._compute_feat_sizes(input)

        # FIX 4: Use the same class-conditioned sparse embeddings as forward().
        # Previously infer() used an empty tensor here, so the decoder had no
        # category signal at inference time — categories that overlapped with
        # background were never pushed apart.
        cls_tokens = self.class_prompt_embed.weight.unsqueeze(0).expand(bs, -1, -1)
        sparse_embeddings = cls_tokens  # (bs, num_classes-1, prompt_embed_dim)

        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        self._features = self._encode_image(input)
        high_res_features = self._features["high_res_feats"]

        low_res_masks, iou_predictions, sam_output_tokens, object_score_logits = self.mask_decoder(
            image_embeddings=self._features["image_embed"],
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        return masks

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size,
        original_size,
    ) -> torch.Tensor:
        """
        Upscale low-res decoder masks to the original image size.

        masks : (B, 1, H, W) or (B, C, H, W) from the decoder.
        input_size / original_size : int or (H, W) tuple — both are accepted.

        Returns (B, C, H, W) float logits at original_size resolution.
        """
        # Normalise size args to (H, W) tuples so F.interpolate never receives
        # a bare int for a 4-D tensor.
        def _to_hw(s):
            return (s, s) if isinstance(s, int) else tuple(s)

        enc_hw  = _to_hw(self.image_encoder.img_size)
        inp_hw  = _to_hw(input_size)
        orig_hw = _to_hw(original_size)

        # The SAM2 decoder returns (B, num_tokens, num_multimask_outputs, H, W)
        # when class-prompt tokens are used as sparse embeddings.
        # e.g. batch=2, 4 class tokens, 1 mask output → (2, 4, 1, 128, 128)
        # Flatten tokens × mask_outputs into a single class-channel dim → (B, C, H, W)
        if masks.dim() == 5:
            B, T, M, H, W = masks.shape
            masks = masks.view(B, T * M, H, W)   # (B, num_classes, H, W)

        # Shouldn't normally be anything other than 4D at this point, but guard anyway.
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        masks = F.interpolate(masks, enc_hw, mode="bilinear", align_corners=False)
        masks = masks[..., : inp_hw[0], : inp_hw[1]]
        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks

    def postprocess_masks2(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
        """
        Perform PostProcessing on output masks.
        """
        from .sam2.utils.misc import get_connected_components

        masks = masks.float()
        if self.max_hole_area > 0:
            mask_flat = masks.flatten(0, 1).unsqueeze(1)
            labels, areas = get_connected_components(mask_flat <= self.mask_threshold)
            is_hole = (labels > 0) & (areas <= self.max_hole_area)
            is_hole = is_hole.reshape_as(masks)
            masks = torch.where(is_hole, self.mask_threshold + 10.0, masks)

        if self.max_sprinkle_area > 0:
            labels, areas = get_connected_components(mask_flat > self.mask_threshold)
            is_hole = (labels > 0) & (areas <= self.max_sprinkle_area)
            is_hole = is_hole.reshape_as(masks)
            masks = torch.where(is_hole, self.mask_threshold - 10.0, masks)

        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks

    def backward_G(self):
        """Calculate cross-entropy + soft IOU loss, respecting ignore pixels."""
        self.loss_G = self.criterionBCE(self.pred_mask, self.gt_mask)
        if self.loss_mode == 'iou':
            # Mask out ignore pixels (255) before computing IOU loss
            valid = (self.gt_mask != 255)
            if valid.any():
                self.loss_G = self.loss_G + _iou_loss(
                    self.pred_mask, self.gt_mask, valid
                )
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward_G()
        self.optimizer.step()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requires_grad for all the networks to avoid unnecessary computations.
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad