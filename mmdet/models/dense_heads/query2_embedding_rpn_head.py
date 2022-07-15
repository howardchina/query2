import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from mmdet.models.builder import HEADS
from ...core import bbox_cxcywh_to_xyxy


@HEADS.register_module()
class Query2EmbeddingRPNHead(BaseModule):
    """RPNHead in the `Sparse R-CNN <https://arxiv.org/abs/2011.12450>`_ .

    Unlike traditional RPNHead, this module does not need FPN input, but just
    decode `init_proposal_bboxes` and expand the first dimension of
    `init_proposal_bboxes` and `init_proposal_features` to the batch_size.

    Args:
        num_proposals (int): Number of init_proposals. Default 100.
        proposal_feature_channel (int): Channel number of
            init_proposal_feature. Defaults to 256.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_proposals=100,
                 dim_global=7,
                 proposal_feature_channel=256,
                 use_global_forward=True,
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(Query2EmbeddingRPNHead, self).__init__(init_cfg)
        self.num_proposals = num_proposals
        self.dim_global = dim_global
        self.proposal_feature_channel = proposal_feature_channel
        self.use_global_forward = use_global_forward
        self._init_layers()

    def _init_layers(self):
        """Initialize a sparse set of proposal boxes and proposal features."""
        self.init_proposal_bboxes = nn.Embedding(self.num_proposals, 4)
        self.init_proposal_features = nn.Embedding(
            self.num_proposals, self.proposal_feature_channel)
        self.global_features_embed = nn.Embedding(
            self.dim_global, self.proposal_feature_channel)
        self.init_global_bbox = nn.Embedding(
            1, 4)

    def init_weights(self):
        """Initialize the init_proposal_bboxes as normalized.

        [c_x, c_y, w, h], and we initialize it to the size of  the entire
        image.
        """
        super(Query2EmbeddingRPNHead, self).init_weights()
        nn.init.constant_(self.init_proposal_bboxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_bboxes.weight[:, 2:], 1)
        nn.init.constant_(self.init_global_bbox.weight[:, :2], 0.5)
        nn.init.constant_(self.init_global_bbox.weight[:, 2:], 1)

    def _decode_init_proposals(self, imgs, img_metas):
        """Decode init_proposal_bboxes according to the size of images and
        expand dimension of init_proposal_features to batch_size.

        Args:
            imgs (list[Tensor]): List of FPN features.
            img_metas (list[dict]): List of meta-information of
                images. Need the img_shape to decode the init_proposals.

        Returns:
            Tuple(Tensor):

                - proposals (Tensor): Decoded proposal bboxes,
                  has shape (batch_size, num_proposals, 4).
                - init_proposal_features (Tensor): Expanded proposal
                  features, has shape
                  (batch_size, num_proposals, proposal_feature_channel).
                - imgs_whwh (Tensor): Tensor with shape
                  (batch_size, 4), the dimension means
                  [img_width, img_height, img_width, img_height].
        """
        proposals = self.init_proposal_bboxes.weight.clone()
        proposals = bbox_cxcywh_to_xyxy(proposals)
        global_box = self.init_global_bbox.weight.clone()
        global_box = bbox_cxcywh_to_xyxy(global_box)
        num_imgs = len(imgs[0])
        imgs_whwh = []
        for meta in img_metas:
            h, w, _ = meta['img_shape']
            imgs_whwh.append(imgs[0].new_tensor([[w, h, w, h]]))
        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        imgs_whwh = imgs_whwh[:, None, :]

        # imgs_whwh has shape (batch_size, 1, 4)
        # The shape of proposals change from (num_proposals, 4)
        # to (batch_size ,num_proposals, 4)
        proposals = proposals * imgs_whwh
        global_box = global_box * imgs_whwh

        init_proposal_features = self.init_proposal_features.weight.clone()
        init_proposal_features = init_proposal_features[None].expand(
            num_imgs, *init_proposal_features.size())
        return proposals, global_box, init_proposal_features, imgs_whwh

    def forward_dummy(self, img, anatomy, img_metas):
        """Dummy forward function.

        Used in flops calculation.
        """
        if self.use_global_forward:
            anatomy = torch.cat(anatomy, 0).reshape(-1, 1)# (bs, 1)
            global_features_embed = self.global_features_embed(anatomy)
        else:
            num_imgs = len(img[0])
            global_features_embed = self.global_features_embed.weight.clone()
            global_features_embed = global_features_embed[None].expand(
                num_imgs, *global_features_embed.size())
        
        proposals, global_box, init_proposal_features, imgs_whwh = self._decode_init_proposals(img, img_metas)
        return proposals, init_proposal_features, global_box, global_features_embed, imgs_whwh

    def forward_train(self, img, anatomy, img_metas):
        """Forward function in training stage."""
        if self.use_global_forward:
            anatomy = torch.cat(anatomy, 0).reshape(-1, 1)# (bs, 1)
            global_features_embed = self.global_features_embed(anatomy)
        else:
            num_imgs = len(img[0])
            global_features_embed = self.global_features_embed.weight.clone()
            global_features_embed = global_features_embed[None].expand(
                num_imgs, *global_features_embed.size())
        
        proposals, global_box, init_proposal_features, imgs_whwh = self._decode_init_proposals(img, img_metas)
        return proposals, init_proposal_features, global_box, global_features_embed, imgs_whwh

    def simple_test_rpn(self, img, anatomy, img_metas):
        """Forward function in testing stage."""
        if self.use_global_forward: 
            anatomy = torch.cat(anatomy, 0).reshape(-1, 1)# (bs, 1)
            global_features_embed = self.global_features_embed(anatomy)
        else:
            num_imgs = len(img[0])
            global_features_embed = self.global_features_embed.weight.clone()
            global_features_embed = global_features_embed[None].expand(
                num_imgs, *global_features_embed.size())
        
        proposals, global_box, init_proposal_features, imgs_whwh = self._decode_init_proposals(img, img_metas)
        return proposals, init_proposal_features, global_box, global_features_embed, imgs_whwh

    def aug_test_rpn(self, imgs, anatomies, img_metas):
        aug_proposal_boxes = []
        aug_proposal_features = []
        aug_global_features = []
        aug_global_box = []
        aug_imgs_whwh = []
        for img, anatony, img_meta in zip(imgs, anatomies, img_metas):
            proposal_boxes, global_box, proposal_features, global_features, imgs_whwh = self.simple_test_rpn(img, anatony, img_meta)
            aug_proposal_boxes.append(proposal_boxes)
            aug_proposal_features.append(proposal_features)
            aug_global_features.append(global_features)
            aug_global_box.append(global_box)
            aug_imgs_whwh.append(imgs_whwh)
        return aug_proposal_boxes, aug_proposal_features, aug_global_box, aug_global_features, aug_imgs_whwh
