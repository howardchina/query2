from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from mmcv.runner import auto_fp16
import torch
from mmdet.core.visualization import imshow_det_bboxes
import mmcv
import numpy as np


@DETECTORS.register_module()
class QueryGlob(TwoStageDetector):
    r"""Implementation of `CondQueryInst: Conditioned Parallelly Supervised Mask Query for
     Instance Segmentation <https://arxiv.org/abs/xxxx.xxxxxx>`, based on 
     QueryInst detector. """
    
    def __init__(self, *args, **kwargs):
        super(QueryGlob, self).__init__(*args, **kwargs)
        assert self.with_rpn, 'QueryInst do not support external proposals'
        
    @auto_fp16(apply_to=('img', 'anatomy'))
    def forward(self, img, anatomy, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, anatomy, img_metas, **kwargs)
        else:
            return self.forward_test(img, anatomy, img_metas, **kwargs)
                
    def forward_train(self,
                      img,
                      anatomy,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            anatomy (List[Tensor], optional): override rpn proposals with
                custom global proposals. for each image with shape (1).
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (List[Tensor], optional) : Segmentation masks for
                each box. But we don't support it in this architecture.
            proposals (List[Tensor], optional): override rpn proposals with
                custom proposals. Use when `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        assert proposals is None, 'QueryInst does not support' \
                                  ' external proposals'
        # assert gt_masks is not None, 'QueryInst needs mask groundtruth annotations' \
        #                           ' for instance segmentation'

        x = self.extract_feat(img)
        proposal_boxes, proposal_features, proposal_global_features, imgs_whwh = \
            self.rpn_head.forward_train(x, anatomy, img_metas)
        roi_losses = self.roi_head.forward_train(
            x,
            proposal_boxes,
            proposal_features,
            proposal_global_features,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_masks=gt_masks,
            imgs_whwh=imgs_whwh)
        return roi_losses
    
    def forward_test(self, imgs, anatomies, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], anatomies[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, anatomies, img_metas, **kwargs)
        
    def simple_test(self, img, anatomy, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        proposal_boxes, proposal_features, proposal_global_features, imgs_whwh = \
            self.rpn_head.simple_test_rpn(x, anatomy, img_metas)
        results = self.roi_head.simple_test(
            x,
            proposal_boxes,
            proposal_features,
            proposal_global_features,
            img_metas,
            imgs_whwh=imgs_whwh,
            rescale=rescale)
        return results

    def aug_test(self, imgs, anatomies, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_boxes, proposal_features, proposal_global_features, imgs_whwh = \
            self.rpn_head.aug_test_rpn(x, anatomies, img_metas)
        results = self.roi_head.aug_test(
            x,
            proposal_boxes,
            proposal_features,
            proposal_global_features,
            img_metas,
            aug_imgs_whwh=imgs_whwh,
            rescale=rescale)
        return results

    def forward_dummy(self, img, anatomy=None):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        # backbone
        x = self.extract_feat(img)
        # rpn
        num_imgs = len(img)
        dummy_img_metas = [
            dict(img_shape=(800, 1333, 3)) for _ in range(num_imgs)
        ]
        anatomy = [
            img.new(1).long() for _ in range(num_imgs)
        ]
        proposal_boxes, proposal_features, proposal_global_features, imgs_whwh = \
            self.rpn_head.simple_test_rpn(x, anatomy, dummy_img_metas)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposal_boxes,
                                               proposal_features,
                                               proposal_global_features,
                                               dummy_img_metas)
        return roi_outs

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            if len(result) == 3: # show glob
                bbox_result, segm_result, glob_result = result
            else:
                bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        globs = None
        if glob_result is not None and len(labels) > 0:  # non empty
            globs = mmcv.concat_list(glob_result)
            if isinstance(globs[0], torch.Tensor):
                globs = torch.stack(globs, dim=0).detach().cpu().numpy()
            else:
                globs = np.stack(globs, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file,
            globs=globs)

        if not (show or out_file):
            return img