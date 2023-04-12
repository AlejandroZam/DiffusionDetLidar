# ==========================================
# Modified by Shoufa Chen
# ===========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DiffusionDet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import itertools
import weakref
from typing import Any, Dict, List, Set
import logging
from collections import OrderedDict
import tkinter
import torch
from fvcore.nn.precise_bn import get_bn_modules
from nuscenes import NuScenes
import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, create_ddp_model, \
    AMPTrainer, SimpleTrainer, hooks
from detectron2.evaluation import COCOEvaluator, LVISEvaluator, verify_results
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.modeling import build_model
from nuscenes.utils.data_classes import LidarPointCloud,Box
from nuscenes.utils.geometry_utils import transform_matrix,view_points, BoxVisibility
from diffusiondet import DiffusionDetDatasetMapper, add_diffusiondet_config, DiffusionDetWithTTA
from diffusiondet.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer
from pyquaternion import Quaternion
from tqdm import tqdm
from detectron2 import model_zoo
import numpy as np
from detectron2.structures import BoxMode
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

# import cv2

categories = ['human.pedestrian.adult',
             'human.pedestrian.child',
             'human.pedestrian.wheelchair',
             'human.pedestrian.stroller',
             'human.pedestrian.personal_mobility',
             'human.pedestrian.police_officer',
             'human.pedestrian.construction_worker',
             'animal',
             'vehicle.car',
             'vehicle.motorcycle',
             'vehicle.bicycle',
             'vehicle.bus.bendy',
             'vehicle.bus.rigid',
             'vehicle.truck',
             'vehicle.construction',
             'vehicle.emergency.ambulance',
             'vehicle.emergency.police',
             'vehicle.trailer',
             'movable_object.barrier',
             'movable_object.trafficcone',
             'movable_object.pushable_pullable',
             'movable_object.debris',
             'static_object.bicycle_rack']

def xyz_convert_bev_points(bbox,max_dist=51,pixel_ratio = 10.0):
  h_size = int((max_dist*2)*int(pixel_ratio))
  w_size = int((max_dist*2)*int(pixel_ratio))
  origin = (h_size/2,w_size/2) # point of origin

  bbox_x = int(round(bbox[0] * pixel_ratio)) + int(origin[1])
  
  bbox_y = int(round(bbox[1]* pixel_ratio)) + int(origin[1])

  bbox_w = int(round(bbox[2]* pixel_ratio)) + int(origin[1]) - bbox_x

  bbox_h = int(round(bbox[3]* pixel_ratio)) + int(origin[1]) - bbox_y
    
  return [bbox_x,bbox_y,bbox_w,bbox_h]

def xywhz_convert_xyz_bbox(xywh,z):
    print('here')


def get_anno_stats(path="./", version='v1.0-mini', categories=None):

    assert(path[-1] == "/"), "Insert '/' in the end of path"
    nusc = NuScenes(version=version, dataroot=path, verbose=False)

    # Select all catecategories if not set
    if categories == None:
        categories = [data["name"] for data in nusc.category]
    assert(isinstance(categories, list)), "Categories type must be list"
    states_mat = [[0] * 3 for i in range(len(categories))]
    dataset_dicts = []
    idx = 0
    for i in tqdm(range(0, len(nusc.sample))):
        sample_data_annotation = nusc.sample[i]['anns']
        for anno in sample_data_annotation:
            my_annotation_metadata =  nusc.get('sample_annotation', anno)
            num = categories.index(my_annotation_metadata['category_name'])
        
            states_mat[num][0] = states_mat[num][0] +  float(my_annotation_metadata['size'][2])
            states_mat[num][1] = states_mat[num][1] + 1  
    temp_dict = {}
    for i in range(len(states_mat)):
        if states_mat[i][1] > 0:

            key = categories[i]
            val = states_mat[i][0]/states_mat[i][1]
        else:
            key = categories[i]
            val = 0
        temp_dict[key] = val

    return(temp_dict)

def show_anno_image():
    print('here')

def test_rotation(path="./", version='v1.0-mini', categories=None):

    assert(path[-1] == "/"), "Insert '/' in the end of path"
    nusc = NuScenes(version=version, dataroot=path, verbose=False)

    # Select all catecategories if not set
    if categories == None:
        categories = [data["name"] for data in nusc.category]
    assert(isinstance(categories, list)), "Categories type must be list"


def get_nuscenes_dicts(path="./", version='v1.0-mini', categories=None,hp_stats=None):
    """
    This is a helper fuction that create dicts from nuscenes to detectron2 format.
    Nuscenes annotation use 3d bounding box, but for detectron we need 2d bounding box.
    The simplest solution is get max x, min x, max y and min y coordinates from 3d bb and
    create 2d box. So we lost accuracy, but this is not critical.
    :param path: <string>. Path to Nuscenes dataset.
    :param version: <string>. Nuscenes dataset version name.
    :param categories <list<string>>. List of selected categories for detection.
        Get from https://www.nuscenes.org/data-annotation
        Categories names:
            ['human.pedestrian.adult',
             'human.pedestrian.child',
             'human.pedestrian.wheelchair',
             'human.pedestrian.stroller',
             'human.pedestrian.personal_mobility',
             'human.pedestrian.police_officer',
             'human.pedestrian.construction_worker',
             'animal',
             'vehicle.car',
             'vehicle.motorcycle',
             'vehicle.bicycle',
             'vehicle.bus.bendy',
             'vehicle.bus.rigid',
             'vehicle.truck',
             'vehicle.construction',
             'vehicle.emergency.ambulance',
             'vehicle.emergency.police',
             'vehicle.trailer',
             'movable_object.barrier',
             'movable_object.trafficcone',
             'movable_object.pushable_pullable',
             'movable_object.debris',
             'static_object.bicycle_rack']
    :return: <dict>. Return dict with data annotation in detectron2 format.
    """

    if hp_stats == None:

        print('need average heights for each catergory')

        return

    assert(path[-1] == "/"), "Insert '/' in the end of path"
    nusc = NuScenes(version=version, dataroot=path, verbose=False)

    # Select all catecategories if not set
    if categories == None:
        categories = [data["name"] for data in nusc.category]
    assert(isinstance(categories, list)), "Categories type must be list"



    dataset_dicts = []
    idx = 0
    for i in tqdm(range(0, len(nusc.sample))):
        sample_data_token = nusc.sample[i]['data']['LIDAR_TOP']


        # ann_record = nusc.get('sample_annotation',nusc.sample[i]['anns'][0])
        # print(ann_record)

        # nusc.render_annotation( nusc.sample[i]['anns'][0], margin= 10,
        #                   view = np.eye(4),
        #                   box_vis_level = BoxVisibility.ANY,
        #                   out_path = None,
        #                   extra_info = False)


        sd_record = nusc.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']

        sample_rec = nusc.get('sample', sd_record['sample_token'])
        chan = sd_record['channel']
        ref_chan = 'LIDAR_TOP'
        ref_sd_token = sample_rec['data'][ref_chan]
        ref_sd_record = nusc.get('sample_data', ref_sd_token)
        bounding_boxes = []
        pc, times = LidarPointCloud.from_file_multisweep(nusc, sample_rec, chan, ref_chan,nsweeps=10) # default is 1


        cs_record = nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', ref_sd_record['ego_pose_token'])

        ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                      rotation=Quaternion(cs_record["rotation"]))

        # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
        ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]

        rotation_vehicle_flat_from_vehicle = np.dot(
            Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
            Quaternion(pose_record['rotation']).inverse.rotation_matrix)
        
        vehicle_flat_from_vehicle = np.eye(4)
        vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
        viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
        box_vis_level = BoxVisibility
        use_flat_vehicle_coordinates = True
        _, boxes, _ = nusc.get_sample_data(ref_sd_token, box_vis_level=box_vis_level,
                                              use_flat_vehicle_coordinates=use_flat_vehicle_coordinates)      

        points = view_points(pc.points[:3, :], viewpoint, normalize=False)
        dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))

        # points_img =  view_points(pc.points[:3, :],  np.array(cs_record['camera_intrinsic']), normalize=False)


        # points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)


        max_dist = 51 # meters, max distance from origin
        pixel_ratio = 10.0 # pixels per meter ratio
        h_size = int((max_dist*2)*int(pixel_ratio))
        w_size = int((max_dist*2)*int(pixel_ratio))
        origin = (h_size/2,w_size/2) # point of origin
        max_lidar_height =4

        bev_mat = np.full((w_size,h_size,3),1.0,dtype=float)
        max=0

        z_min = np.min(points[2])
        z_max = np.max(points[2])
        mag_min = np.min(dists)
        mag_max = np.max(dists)

        for i in range(points.shape[1]):


        
            x_point_meters = points[0][i] + max_dist 
            x_point_pixels = int(round(points[0][i] * pixel_ratio)) + int(origin[1])
            y_point_meters = points[1][i] + max_dist 
            y_point_pixels = int(round(points[1][i] * pixel_ratio)) + int(origin[0])


            norm_z = (points[2][i] - z_min) / (z_max-z_min)
            norm_mag = (dists[i] - mag_min) / (mag_max-mag_min)
          


            dot = np.asarray([0.0,norm_z,norm_mag])

            if x_point_pixels < w_size and y_point_pixels < h_size and x_point_pixels >=0 and y_point_pixels >=0:
                bev_mat[x_point_pixels,y_point_pixels] = dot

        fig,ax = plt.subplots(1,1,figsize=(15,15))

        plt.imshow(bev_mat)

        # data = nusc.get('sample_data', sample_rec_cur['data']["CAM_FRONT"])

        data = sd_record

        record = {}

        record["file_name"] = sample_data_token
        record["image_id"] = idx
        record["height"] = h_size
        record["width"] = w_size
        idx += 1

        # Get boxes from front camera
        # _, boxes, camera_intrinsic = nusc.get_sample_data(sample_rec_cur['data']["CAM_FRONT"], BoxVisibility.ANY)
        # Get only necessary boxes
        boxes = [box for box in boxes if box.name in categories]
        # Go through all bounding boxes
        objs = []
        for box in boxes:

            z = box.center[2]

            h = box.wlh[2]

            #1.84m height from ground
            # sensor_max_height = 1.84
            # corners = view_points(box.corners(),np.eye(4),normalize=False)[:2,:]
            corners_alt = view_points(box.corners(),np.eye(4),normalize=False)

  



            xval_top = [corners_alt[0][4],corners_alt[0][5],corners_alt[0][0],corners_alt[0][1]]

            xval_bot = [corners_alt[0][7],corners_alt[0][6],corners_alt[0][3],corners_alt[0][2]]

            xval_top.sort()
            yval_top = [corners_alt[1][4],corners_alt[1][5],corners_alt[1][0],corners_alt[1][1]]

            yval_bot = [corners_alt[1][7],corners_alt[1][6],corners_alt[1][3],corners_alt[1][2]]

            yval_top.sort()

            # height_top = 
            # bot_mid = 

            max_x = xval_top[-1] 
            min_x = xval_top[0]  
            max_y = yval_top[-1] 
            min_y = yval_top[0] 

            c = xyz_convert_bev_points([min_x, min_y, max_x, max_y])

            # box_ = xywhz_convert_xyz_bbox()

            ax.add_patch(Rectangle((c[1],c[0]),c[3],c[2],fc ='none',ec = (0,1,0),lw = 0.5)  )

            
            # cent_z= ((z - h/2.0)/255 *3.0) + sensor_max_height
            # h_= h/255 *3.0
            # if 1:

            #   print('name: ',box.name)
            # #   print('box: ',box)
            #   print('z pos: ',z)
            #   print('height: ',h)
            #   print('cent_z: ',cent_z)
            #   print('h_: ',h_)
            #   # print('max z height: ', z+h/2 ) 
            #   # print('min z height: ', z-h/2 ) 
            #   print('coor: ', c)
            # # print('rot: ', box.)
    

            height = h/hp_stats[box.name]


            obj = {
                "bbox": [c[1],c[0],c[3],c[2]],
                "height":[height,z],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": categories.index(box.name),
                "iscrowd": 0
            }
            objs.append(obj)

        record["annotations"] = objs
    
        dataset_dicts.append(record)
        plt.show()
        exit()

    return dataset_dicts




class Trainer(DefaultTrainer):
    """ Extension of the Trainer class adapted to DiffusionDet. """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super(DefaultTrainer, self).__init__()  # call grandfather's `__init__` while avoid father's `__init()`
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        ########## EMA ############
        kwargs = {
            'trainer': weakref.proxy(self),
        }
        kwargs.update(may_get_ema_checkpointer(cfg, model))
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
            # trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        # setup EMA
        may_build_model_ema(cfg, model)
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if 'lvis' in dataset_name:
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        else:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DiffusionDetDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def ema_test(cls, cfg, model, evaluators=None):
        # model with ema weights
        logger = logging.getLogger("detectron2.trainer")
        if cfg.MODEL_EMA.ENABLED:
            logger.info("Run evaluation with EMA.")
            with apply_model_ema_and_restore(model):
                results = cls.test(cfg, model, evaluators=evaluators)
        else:
            results = cls.test(cfg, model, evaluators=evaluators)
        return results

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = DiffusionDetWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        if cfg.MODEL_EMA.ENABLED:
            cls.ema_test(cfg, model, evaluators)
        else:
            res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            EMAHook(self.cfg, self.model) if cfg.MODEL_EMA.ENABLED else None,  # EMA hook
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret


def setup(args):
    """
    Create configs and perform basic setups.
    """
    #cfg = get_cfg()
    # add_diffusiondet_config(cfg)
    # add_model_ema_configs(cfg)
    # cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    # cfg.freeze()
    # default_setup(cfg, args)

    #cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))


    # cfg.DATASETS.TRAIN = ("my_dataset_train",)
    # cfg.DATASETS.TEST = ("my_dataset_val")
    # cfg.DATALOADER.NUM_WORKERS = 2
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    #cfg.MODEL.WEIGHTS = '/content/DiffusionDet/diffdet_coco_swinbase.pth'  # Let training initialize from model zoo
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml")  # Let training initialize from model zoo
    #cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    #cfg.SOLVER.BASE_LR = 0.00020  # pick a good LR
    #cfg.SOLVER.MAX_ITER = 5000 #5000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    #cfg.SOLVER.STEPS = []        # do not decay learning rate
    #cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    #cfg.MODEL.ROI_HEADS.NUM_CLASSES =   # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    #os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    #trainer = DefaultTrainer(cfg) 
    #trainer.resume_or_load(resume=True)
    #trainer.train()


    #return cfg


def main(args):


    #dataset_dict = get_nuscenes_dicts(path="/data/sets/nuscenes/", version='v1.0-mini', categories=categories)
    stats = get_anno_stats(path="/data/sets/nuscenes/", version='v1.0-mini', categories=categories)
    dataset_dict = get_nuscenes_dicts(path="/data/sets/nuscenes/", version='v1.0-mini', categories=categories,hp_stats = stats)



    # test = test_rotation(path="/data/sets/nuscenes/", version='v1.0-mini', categories=categories)

    # cfg = setup(args)

    # if args.eval_only:
    #     model = Trainer.build_model(cfg)
    #     kwargs = may_get_ema_checkpointer(cfg, model)
    #     if cfg.MODEL_EMA.ENABLED:
    #         EMADetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(cfg.MODEL.WEIGHTS,
    #                                                                                           resume=args.resume)
    #     else:
    #         DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(cfg.MODEL.WEIGHTS,
    #                                                                                        resume=args.resume)
    #     res = Trainer.ema_test(cfg, model)
    #     if cfg.TEST.AUG.ENABLED:
    #         res.update(Trainer.test_with_TTA(cfg, model))
    #     if comm.is_main_process():
    #         verify_results(cfg, res)
    #     return res

    # trainer = Trainer(cfg)
    # trainer.resume_or_load(resume=args.resume)
    # return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
