import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .coco import CocoDataset
from .pipelines import Compose


@DATASETS.register_module()
class StanfordCarDataset(CocoDataset):

    CLASSES = ('AM General Hummer SUV 2000',
        'Acura RL Sedan 2012',
        'Acura TL Sedan 2012',
        'Acura TL Type-S 2008',
        'Acura TSX Sedan 2012',
        'Acura Integra Type R 2001',
        'Acura ZDX Hatchback 2012',
        'Aston Martin V8 Vantage Convertible 2012',
        'Aston Martin V8 Vantage Coupe 2012',
        'Aston Martin Virage Convertible 2012',
        'Aston Martin Virage Coupe 2012',
        'Audi RS 4 Convertible 2008',
        'Audi A5 Coupe 2012',
        'Audi TTS Coupe 2012',
        'Audi R8 Coupe 2012',
        'Audi V8 Sedan 1994',
        'Audi 100 Sedan 1994',
        'Audi 100 Wagon 1994',
        'Audi TT Hatchback 2011',
        'Audi S6 Sedan 2011',
        'Audi S5 Convertible 2012',
        'Audi S5 Coupe 2012',
        'Audi S4 Sedan 2012',
        'Audi S4 Sedan 2007',
        'Audi TT RS Coupe 2012',
        'BMW ActiveHybrid 5 Sedan 2012',
        'BMW 1 Series Convertible 2012',
        'BMW 1 Series Coupe 2012',
        'BMW 3 Series Sedan 2012',
        'BMW 3 Series Wagon 2012',
        'BMW 6 Series Convertible 2007',
        'BMW X5 SUV 2007',
        'BMW X6 SUV 2012',
        'BMW M3 Coupe 2012',
        'BMW M5 Sedan 2010',
        'BMW M6 Convertible 2010',
        'BMW X3 SUV 2012',
        'BMW Z4 Convertible 2012',
        'Bentley Continental Supersports Conv. Convertible 2012',
        'Bentley Arnage Sedan 2009',
        'Bentley Mulsanne Sedan 2011',
        'Bentley Continental GT Coupe 2012',
        'Bentley Continental GT Coupe 2007',
        'Bentley Continental Flying Spur Sedan 2007',
        'Bugatti Veyron 16.4 Convertible 2009',
        'Bugatti Veyron 16.4 Coupe 2009',
        'Buick Regal GS 2012',
        'Buick Rainier SUV 2007',
        'Buick Verano Sedan 2012',
        'Buick Enclave SUV 2012',
        'Cadillac CTS-V Sedan 2012',
        'Cadillac SRX SUV 2012',
        'Cadillac Escalade EXT Crew Cab 2007',
        'Chevrolet Silverado 1500 Hybrid Crew Cab 2012',
        'Chevrolet Corvette Convertible 2012',
        'Chevrolet Corvette ZR1 2012',
        'Chevrolet Corvette Ron Fellows Edition Z06 2007',
        'Chevrolet Traverse SUV 2012',
        'Chevrolet Camaro Convertible 2012',
        'Chevrolet HHR SS 2010',
        'Chevrolet Impala Sedan 2007',
        'Chevrolet Tahoe Hybrid SUV 2012',
        'Chevrolet Sonic Sedan 2012',
        'Chevrolet Express Cargo Van 2007',
        'Chevrolet Avalanche Crew Cab 2012',
        'Chevrolet Cobalt SS 2010',
        'Chevrolet Malibu Hybrid Sedan 2010',
        'Chevrolet TrailBlazer SS 2009',
        'Chevrolet Silverado 2500HD Regular Cab 2012',
        'Chevrolet Silverado 1500 Classic Extended Cab 2007',
        'Chevrolet Express Van 2007',
        'Chevrolet Monte Carlo Coupe 2007',
        'Chevrolet Malibu Sedan 2007',
        'Chevrolet Silverado 1500 Extended Cab 2012',
        'Chevrolet Silverado 1500 Regular Cab 2012',
        'Chrysler Aspen SUV 2009',
        'Chrysler Sebring Convertible 2010',
        'Chrysler Town and Country Minivan 2012',
        'Chrysler 300 SRT-8 2010',
        'Chrysler Crossfire Convertible 2008',
        'Chrysler PT Cruiser Convertible 2008',
        'Daewoo Nubira Wagon 2002',
        'Dodge Caliber Wagon 2012',
        'Dodge Caliber Wagon 2007',
        'Dodge Caravan Minivan 1997',
        'Dodge Ram Pickup 3500 Crew Cab 2010',
        'Dodge Ram Pickup 3500 Quad Cab 2009',
        'Dodge Sprinter Cargo Van 2009',
        'Dodge Journey SUV 2012',
        'Dodge Dakota Crew Cab 2010',
        'Dodge Dakota Club Cab 2007',
        'Dodge Magnum Wagon 2008',
        'Dodge Challenger SRT8 2011',
        'Dodge Durango SUV 2012',
        'Dodge Durango SUV 2007',
        'Dodge Charger Sedan 2012',
        'Dodge Charger SRT-8 2009',
        'Eagle Talon Hatchback 1998',
        'FIAT 500 Abarth 2012',
        'FIAT 500 Convertible 2012',
        'Ferrari FF Coupe 2012',
        'Ferrari California Convertible 2012',
        'Ferrari 458 Italia Convertible 2012',
        'Ferrari 458 Italia Coupe 2012',
        'Fisker Karma Sedan 2012',
        'Ford F-450 Super Duty Crew Cab 2012',
        'Ford Mustang Convertible 2007',
        'Ford Freestar Minivan 2007',
        'Ford Expedition EL SUV 2009',
        'Ford Edge SUV 2012',
        'Ford Ranger SuperCab 2011',
        'Ford GT Coupe 2006',
        'Ford F-150 Regular Cab 2012',
        'Ford F-150 Regular Cab 2007',
        'Ford Focus Sedan 2007',
        'Ford E-Series Wagon Van 2012',
        'Ford Fiesta Sedan 2012',
        'GMC Terrain SUV 2012',
        'GMC Savana Van 2012',
        'GMC Yukon Hybrid SUV 2012',
        'GMC Acadia SUV 2012',
        'GMC Canyon Extended Cab 2012',
        'Geo Metro Convertible 1993',
        'HUMMER H3T Crew Cab 2010',
        'HUMMER H2 SUT Crew Cab 2009',
        'Honda Odyssey Minivan 2012',
        'Honda Odyssey Minivan 2007',
        'Honda Accord Coupe 2012',
        'Honda Accord Sedan 2012',
        'Hyundai Veloster Hatchback 2012',
        'Hyundai Santa Fe SUV 2012',
        'Hyundai Tucson SUV 2012',
        'Hyundai Veracruz SUV 2012',
        'Hyundai Sonata Hybrid Sedan 2012',
        'Hyundai Elantra Sedan 2007',
        'Hyundai Accent Sedan 2012',
        'Hyundai Genesis Sedan 2012',
        'Hyundai Sonata Sedan 2012',
        'Hyundai Elantra Touring Hatchback 2012',
        'Hyundai Azera Sedan 2012',
        'Infiniti G Coupe IPL 2012',
        'Infiniti QX56 SUV 2011',
        'Isuzu Ascender SUV 2008',
        'Jaguar XK XKR 2012',
        'Jeep Patriot SUV 2012',
        'Jeep Wrangler SUV 2012',
        'Jeep Liberty SUV 2012',
        'Jeep Grand Cherokee SUV 2012',
        'Jeep Compass SUV 2012',
        'Lamborghini Reventon Coupe 2008',
        'Lamborghini Aventador Coupe 2012',
        'Lamborghini Gallardo LP 570-4 Superleggera 2012',
        'Lamborghini Diablo Coupe 2001',
        'Land Rover Range Rover SUV 2012',
        'Land Rover LR2 SUV 2012',
        'Lincoln Town Car Sedan 2011',
        'MINI Cooper Roadster Convertible 2012',
        'Maybach Landaulet Convertible 2012',
        'Mazda Tribute SUV 2011',
        'McLaren MP4-12C Coupe 2012',
        'Mercedes-Benz 300-Class Convertible 1993',
        'Mercedes-Benz C-Class Sedan 2012',
        'Mercedes-Benz SL-Class Coupe 2009',
        'Mercedes-Benz E-Class Sedan 2012',
        'Mercedes-Benz S-Class Sedan 2012',
        'Mercedes-Benz Sprinter Van 2012',
        'Mitsubishi Lancer Sedan 2012',
        'Nissan Leaf Hatchback 2012',
        'Nissan NV Passenger Van 2012',
        'Nissan Juke Hatchback 2012',
        'Nissan 240SX Coupe 1998',
        'Plymouth Neon Coupe 1999',
        'Porsche Panamera Sedan 2012',
        'Ram C/V Cargo Van Minivan 2012',
        'Rolls-Royce Phantom Drophead Coupe Convertible 2012',
        'Rolls-Royce Ghost Sedan 2012',
        'Rolls-Royce Phantom Sedan 2012',
        'Scion xD Hatchback 2012',
        'Spyker C8 Convertible 2009',
        'Spyker C8 Coupe 2009',
        'Suzuki Aerio Sedan 2007',
        'Suzuki Kizashi Sedan 2012',
        'Suzuki SX4 Hatchback 2012',
        'Suzuki SX4 Sedan 2012',
        'Tesla Model S Sedan 2012',
        'Toyota Sequoia SUV 2012',
        'Toyota Camry Sedan 2012',
        'Toyota Corolla Sedan 2012',
        'Toyota 4Runner SUV 2012',
        'Volkswagen Golf Hatchback 2012',
        'Volkswagen Golf Hatchback 1991',
        'Volkswagen Beetle Hatchback 2012',
        'Volvo C30 Hatchback 2012',
        'Volvo 240 Sedan 1993',
        'Volvo XC90 SUV 2007',
        'smart fortwo Convertible 2012')

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        self.data_infos = self.load_annotations(self.ann_file)

        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)
    
    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels. 
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
#             if ann['area'] <= 0 or w < 1 or h < 1:
            if w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)


        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore)

        return ann

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results
    
    def _glob2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        glob_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, glb = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)
                
                # global results
                data = dict()
                data['image_id'] = img_id
                data['category_id'] = self.cat_ids[label]
                data['score'] = float(glb[0, label])
                glob_json_results.append(data)    
                
        return bbox_json_results, glob_json_results

    def _eval_global(self, results, topk=1):
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
            return_single = True
        else:
            return_single = False

        maxk = max(topk) 
        
        global_preds = []
        global_labels = []
        for idx in range(len(results)):
            _, global_pred = results[idx]
            global_labels.append(int(self.get_ann_info(idx)['labels']))
            global_preds.append(global_pred.detach().cpu())
        global_labels = torch.tensor(global_labels)
        global_preds = torch.cat(global_preds).reshape((-1, len(self.CLASSES)))
        
        pred_value, pred_label = global_preds.topk(maxk, dim=1)
        pred_label = pred_label.t()
        correct = pred_label.eq(global_labels.view(1, -1).expand_as(pred_label))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / global_preds.size(0)))
        return res[0] if return_single else res        
    
    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple) and len(results[0]) == 2:
            json_results = self._glob2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['glob'] = f'{outfile_prefix}.glob.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['glob'])
            
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast', 'glob']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue
            
            if metric == 'glob':
                global_results = self._eval_global(results)
                key = 'accuracy'
                val = f'{global_results}'
                eval_results[key] = val
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                cocoDt = cocoGt.loadRes(predictions)
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
    
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['anatomy'] = []
        
