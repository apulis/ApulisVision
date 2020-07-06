import logging
import os
import os.path as osp
import sys
#import tempfile
import json
import mmcv
import numpy as np
import pandas as pd

from .density import generate_density_map, generate_density_map_csp, rpc_category_to_super_category

sys.path.append('../../../mmdetection/')

from mmdet.core import eval_map, eval_recalls
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.pipelines import Compose
from mmdet.datasets.builder import DATASETS

import rpctool
import boxx
import pickle
import itertools
from collections import defaultdict
from operator import itemgetter
from datetime import datetime
from tabulate import tabulate

@DATASETS.register_module
class RPC_Dataset(CustomDataset):
    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 rendered=False,
                 select_pseudo_lavel_with_ground_truth=False,
                 csp=False,
                 match_error=0,
                 n_class_density_map=0,
                 export_result_dir=None,
                 use_density_map=False,
                 generate_pseudo_label=False,):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        
        ######## edit here
        self.filter_empty_gt = filter_empty_gt
        self.rendered = rendered 
        self.csp = csp
        self.match_error=match_error
        self.n_class_density_map = n_class_density_map
        self.export_result_dir = export_result_dir
        self.generate_pseudo_label = generate_pseudo_label
        self.select_pseudo_lavel_with_ground_truth = select_pseudo_lavel_with_ground_truth
        ########
        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        self.img_infos = self.load_annotations(self.ann_file)
        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None
        # filter images too small
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # processing pipeline
        self.pipeline = Compose(pipeline)

    with open(os.path.join(os.path.dirname(__file__), "rpc.txt"), "r") as f:
        CLASSES = [x.strip() for x in f.readlines()]


    def get_cAcc(self, result, level):
        LEVELS = ('easy', 'medium', 'hard', 'averaged')
        index = LEVELS.index(level)
        return float(result.loc[index, 'cAcc'].strip('%'))
            
    def check_best_result(self, output_folder, result, result_str, filename):
        current_cAcc = self.get_cAcc(result, "averaged")
        best_path = osp.join(output_folder, 'best_result.txt')
        if osp.exists(best_path):
            with open(best_path) as f:
                best_cAcc = float(f.readline().strip())
            if current_cAcc >= best_cAcc:
                best_cAcc = current_cAcc
                with open(best_path, 'w') as f:
                    f.write(str(best_cAcc) + '\n' + filename + '\n' + result_str)
        else:
            best_cAcc = current_cAcc
            with open(best_path, 'w') as f:
                f.write(str(current_cAcc) + '\n' + filename + '\n' + result_str)
        return best_cAcc

        
    def load_annotations(self, ann_file):
        self.ann_file = ann_file        
        self.cat_ids = list(range(1,201))
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        
        with open(self.ann_file) as fid:
            data = json.load(fid)

        annotations = defaultdict(list)
        images = []
        for image in data['images']:
            image["filename"] = image["file_name"]
            del image["file_name"]
            images.append(image)
            
        for ann in data['annotations']:
            bbox = ann['bbox']
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            annotations[ann['image_id']].append((ann['category_id'], x, y, w, h))

        self.img_infos = images
        
        self.annotations = dict(annotations)
        
        self.img_ids = list(self.annotations.keys())
        return self.img_infos
        

    def get_ann_info(self, idx):
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_raw_labels = []
        
        img_id = self.img_infos[idx]['id']
        img_size = self.img_infos[idx]['width']
        
        if self.img_infos[idx]['width'] != self.img_infos[idx]['height']:
            print("Width and Height do not match in image id '{}'".format(img_id))
            print((self.img_infos[idx]['width'], self.img_infos[idx]['height']))
        #img_scale = 800 / self.img_infos[idx]['width']
        ann_info = self.annotations[img_id]
        
        for ann in ann_info:
            category_id, x1, y1, w, h = ann
            if w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            #bbox_resized = [x * img_scale for x in bbox]
            
            if True:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[category_id])
                gt_raw_labels.append(category_id)
                #gt_masks_ann.append(ann['segmentation'])

                
        ############## Generate Density Map #################
        if self.n_class_density_map > 0:
            self.density_map_stride = 1.0 / 8
            self.density_min_sigma = 1.0

            size = int(self.density_map_stride * 800)
            n_class_density_map = self.n_class_density_map

            super_categories = [rpc_category_to_super_category(category, n_class_density_map) for category in gt_raw_labels]
            
            if self.csp:
                density_map = generate_density_map_csp(super_categories, gt_bboxes,
                                                    scale=size / img_size,
                                                    size=size, num_classes=n_class_density_map,
                                                    min_sigma=self.density_min_sigma)
            else:
                
                density_map = generate_density_map(super_categories, gt_bboxes,
                                                    scale=size / img_size,
                                                    size=size, num_classes=n_class_density_map,
                                                    min_sigma=self.density_min_sigma)    
        else:
            density_map = np.zeros((1, 100, 100), dtype=np.float32)
            
        ############### End of Density Map ##################
        
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
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            density_map=density_map)
        return ann


    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = self.annotations.keys()
        for i, img_info in enumerate(self.img_infos):
            if self.filter_empty_gt and img_info['id'] not in ids_with_ann:
                continue
            if img_info['id'] == 3610:
                print("Filtered Image ID 3610.")
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                
        print("Filters applied here, used {} valid images".format(len(valid_inds)), flush=True)
        
        return valid_inds
    
    def format_results(self, results, outfile_prefix, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))      
        
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = '{}.{}.json'.format(outfile_prefix, 'bbox')
            result_files['proposal'] = '{}.{}.json'.format(
                outfile_prefix, 'bbox')
            mmcv.dump(json_results, result_files['bbox'])
            return result_files, None
            
        elif isinstance(results[0], tuple):
            ## Important here
            json_results = self._bbox_counter2json(results)
            result_files['bbox'] = '{}.{}.json'.format(outfile_prefix, 'bbox')
            result_files['proposal'] = '{}.{}.json'.format(
                outfile_prefix, 'bbox')
            result_files['counter'] = '{}.{}.pkl'.format(outfile_prefix, 'counter')
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['counter'])
            return result_files, json_results[1]
            
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = '{}.{}.json'.format(
                outfile_prefix, 'proposal')
            mmcv.dump(json_results, result_files['proposal'])
            return result_files, None
        else:
            raise TypeError('invalid type of results')
        
    
    def xyxy2xywh(self, bbox):
        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0] + 1,
            _bbox[3] - _bbox[1] + 1,
        ]

    def _proposal2json(self, results):
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
    
    def _bbox_counter2json(self, results):
        bbox_json_results = []
        counter_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, counter, density_map = results[idx]
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

                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]                        
                    data['counter'] = counter
                    data['density_map'] = density_map
                    counter_json_results.append(data)
        return bbox_json_results, counter_json_results
    
    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 jsonfile_prefix=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ["bbox", 'mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=self.CLASSES,
                logger=logger)
            eval_results['bbox_mAP_0.5'] = mean_ap
            
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results['recall@{}@{}'.format(num, iou)] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results['AR@{}'.format(num)] = ar[i]
                    
        elif metric == 'bbox': 
            if jsonfile_prefix is None:  # in test model, there will be jsonfile_prefix, but in val mode, you have to assign where to export result
                jsonfile_prefix = osp.join(self.export_result_dir, "result.pkl")
                output_folder = self.export_result_dir
            else:
                output_folder = jsonfile_prefix[:-len(osp.basename(jsonfile_prefix))] + "val_result"
            
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                
                
            if False:
                # Optional mAP 0.5 eval, disable for now
                assert isinstance(iou_thr, float)
                results_bbox = [result[0] for result in results]
                mean_ap, _ = eval_map(
                    results_bbox,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    dataset=self.CLASSES,
                    logger=logger)
                eval_results['bbox_mAP_0.5'] = mean_ap
            
            ### bbox part
            result_files, counter_result = self.format_results(results, jsonfile_prefix)  
            # This step used "format_results" to generate bbox result and counter result
            
            if isinstance(results[0], tuple):
                has_density_map = True
            else:
                has_density_map = False
                
            ## Use RPC tools to get metrics
            
            with open(result_files["bbox"]) as fid:
                res_js = json.load(fid)
                
            with open(self.ann_file) as fid:
                gt_js = json.load(fid)
            
            result = rpctool.evaluate(res_js, gt_js)
            
            if has_density_map: 
                ######## Start counter head evaluation and pseudo label generation
                num_density_classes = 1

                # Change this threshold can change a lot with bbox pred result
                threshold = 0.95
                pred_density_cat_counts = np.zeros((num_density_classes,), dtype=np.int32)

                pseudo_label = []
                ids = []
                gt = []
                bbox = []
                pred = []
                mae = 0
                mae_bbox = 0 
                density_correct = 0
                bbox_correct = 0 
                bbox_density_match = 0
                count = 0
                n_images_selected = 0 
                n_images_selected_bbox_correct = 0

                gt_density_cat_counts_dict = {}
                for image_id, dict_list in itertools.groupby(gt_js['annotations'], key=itemgetter('image_id')):
                    gt_density_cat_counts = np.zeros((num_density_classes,), dtype=np.int32)
                    for single_dict in dict_list:
                        density_category = 0
                        gt_density_cat_counts[density_category] += 1
                    gt_density_cat_counts_dict[image_id] = gt_density_cat_counts
            

            
                for image_id, dict_list in itertools.groupby(counter_result, key=itemgetter('image_id')):
                    box_density_cat_counts = np.zeros((num_density_classes,), dtype=np.int32)

                    flag = True
                    single_image_data = []
                    for single_dict in dict_list:
                        if flag: 
                            #pred_density_cat_counts = np.round(np.sum(single_dict["density_map"][0][0].cpu().numpy())).astype(np.int32)
                            pred_density_cat_counts = np.round(single_dict["counter"][0][0]).astype(np.int32)
                            flag = False

                        if single_dict["score"] > threshold:
                            single_det = {}
                            density_category = 0
                            #density_category = rpc_category_to_super_category(label, num_density_classes)
                            #box_all_cat_counts[label] += 1
                            box_density_cat_counts[density_category] += 1

                            single_det['image_id'] = single_dict["image_id"]
                            single_det['category_id'] = single_dict["category_id"]
                            single_det['bbox'] = single_dict["bbox"]
                            single_image_data.append(single_det)

                    ids.append(image_id)
                    gt.append(gt_density_cat_counts_dict[image_id][0])
                    bbox.append(box_density_cat_counts[0])
                    pred.append(pred_density_cat_counts)

                    if np.all(box_density_cat_counts == pred_density_cat_counts):
                        density_correct += 1

                    if np.all(box_density_cat_counts == gt_density_cat_counts_dict[image_id]):
                        bbox_correct += 1

                    if np.sum(np.abs(box_density_cat_counts[0] - pred_density_cat_counts)) <= self.match_error:
                        bbox_density_match += 1

                    mae += np.sum(np.abs(gt_density_cat_counts_dict[image_id] - pred_density_cat_counts))
                    mae_bbox += np.sum(np.abs(gt_density_cat_counts_dict[image_id] - box_density_cat_counts))
                    count +=1

                    if self.select_pseudo_lavel_with_ground_truth:
                        is_valid = np.all(box_density_cat_counts == gt_density_cat_counts_dict[image_id])
                    else:
                        is_valid = np.sum(np.abs(box_density_cat_counts[0] - pred_density_cat_counts)) <= self.match_error

                    if is_valid:
                        pseudo_label.extend(single_image_data)
                        n_images_selected += 1
                        is_box_correct = np.all(box_density_cat_counts == gt_density_cat_counts_dict[image_id])
                        if is_box_correct:
                             n_images_selected_bbox_correct += 1

                data_export = tabulate(pd.DataFrame({"id":ids, "gt":gt, "bbox":bbox, "pred":pred}), headers='keys', tablefmt='github', showindex=False)

                time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                filename_data = osp.join(output_folder, 'result_{}_detail.txt'.format(time_stamp))
                with open(filename_data, 'w') as fid:
                    fid.write(data_export)
                export_result = "\n Density MAE:{:3.5f}, BBox MAE:{:3.5f}, Match rate:{:3.5f}, Density Correct Rate:{:3.5f}, BBox Correct Rate:{:3.5f}, Total Images:{}".format(mae/count, mae_bbox/count, bbox_density_match/count, density_correct/count, bbox_correct/count, count)

                if self.generate_pseudo_label:
                    pseudo_label_js = gt_js
                    pseudo_label_js["annotations"] = pseudo_label
                    export_result = export_result + "\n Seletced {} bboxes from {} detections, total {} images, {} bbox correct.".format(len(pseudo_label), len(counter_result), 
                                                                                                                        n_images_selected, n_images_selected_bbox_correct)
                    # write pseudo label to file
                    filename = osp.join(output_folder, 'pseudo_label.json')
                    mmcv.dump(pseudo_label_js, filename)

                ######## end counter head evaluation and pseudo label generation
            
            # Start parsing evaluation result from rpc_tool's return value
            result_str = str(result)                                     
            if has_density_map:
                result_str += export_result

            filename = osp.join(output_folder, 'result_{}.txt'.format(time_stamp))                  
            with open(filename, 'w') as fid:
                fid.write(result_str)

            best_cAcc = self.check_best_result(output_folder, result, result_str, filename)
            print('Best cAcc: {}%'.format(best_cAcc))
            
            metric_items = ['averaged', 'hard', 'medium', 'easy']
            
            for i in range(len(metric_items)):
                key = '{}_{}'.format('cAcc', metric_items[i])
                val =  self.get_cAcc(result, metric_items[i])
                eval_results[key] = val

            if has_density_map:
                eval_results.update({
                    'Density_correct_Ratio': density_correct/count,
                    'Bbox_correct_Ratio': bbox_correct/count,
                    'Density MAE': mae /count,
                    'BBox MAE': mae_bbox/count,
                })
                
        print(result_str)
        return eval_results
    
@DATASETS.register_module
class RPC_Syn_Dataset(RPC_Dataset):             
    def load_annotations(self, ann_file):
        # Syn dataset's format is very different
        self.cat_ids = list(range(1,201))
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_infos = []
        self.ann_file = ann_file

        with open(self.ann_file) as fid:
            self.annotations = json.load(fid)
            
        for ann in self.annotations:
            if self.rendered:
                img_info = {'filename':ann["image_id"],
                 'width': 800,
                 'height': 800,
                 'id': None,}
            else:
                img_info = {'filename': ann["image_id"],
                 'width': 1815,
                 'height': 1815,
                 'id': None,}
            
            self.img_infos.append(img_info)
            
        if self.rendered:
            self.image_size = 800
            self.scale = 800 / 1815
        else:
            self.image_size = 1815
            self.scale = 1.0
            
        self.n_class_density_map = 1
        return self.img_infos
        

    def get_ann_info(self, idx):
        """ Format of syn/render dataset 
        └── /: list  2000
        ├── 0: dict  2
        │   ├── image_id: synthesized_image_0.jpg
        │   └── objects: list  13
        │       ├── 0: dict  3
        │       │   ├── bbox: list  4
        │       │   │   ├── 0: 205
        │       │   │   ├── 1: 606
        │       │   │   ├── 2: 129
        │       │   │   └── 3: 76
        │       │   ├── category_id: 141
        │       │   └── center_of_mass: list  2
        │       │       ├── 0: 270
        │       │       └── 1: 644    
        """
        
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_raw_labels = []
        
        ann_info = self.annotations[idx]["objects"]
        
        for ann in ann_info:
            x1, y1, w, h = ann["bbox"]
            category_id = ann["category_id"]
            if w < 1 or h < 1:
                continue
            bbox = [x1*self.scale, y1*self.scale, (x1 + w - 1)*self.scale, (y1 + h - 1)*self.scale]
            
            if True:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[category_id])
                #gt_masks_ann.append(ann['segmentation'])
                gt_raw_labels.append(category_id)
                
                
        ############## Generate Density Map #################
        
        if self.n_class_density_map > 0:
            self.density_map_stride = 1.0 / 8
            self.density_min_sigma = 1.0

            size = int(self.density_map_stride * 800)
            n_class_density_map = self.n_class_density_map

            super_categories = [rpc_category_to_super_category(category, n_class_density_map) for category in gt_raw_labels]
            
            if self.csp:
                density_map = generate_density_map_csp(super_categories, gt_bboxes,
                                                    scale=size / self.image_size,
                                                    size=size, num_classes=n_class_density_map,
                                                    min_sigma=self.density_min_sigma)
            else:
                
                density_map = generate_density_map(super_categories, gt_bboxes,
                                                    scale=size / self.image_size,
                                                    size=size, num_classes=n_class_density_map,
                                                    min_sigma=self.density_min_sigma)

        
        ############### End of Density Map ##################
        

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
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            density_map=density_map,)

        return ann

    def _filter_imgs(self, min_size=32):
        print("Use All {} images, no filters applied here.".format(len(self.img_infos)), flush=True)
        return list(range(len(self.img_infos)))