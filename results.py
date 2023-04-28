# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import logging
import numpy as np
import os, sys
import tempfile
import time
import warnings
import cv2
import tqdm
from diffusiondet.register_coco import register_coco_instances
from diffusiondet.catalog import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.data.detection_utils import read_image

from detectron2.engine import DefaultPredictor
import math
from diffusiondet.util.birdview_detection_refiner import BirdviewDetectionRefiner
from diffusiondet.util.utils_3d import _draw_projection_obstacle_to_cam
from diffusiondet.util.object_3d import Object3d
from diffusiondet.util.utils_calib import Calibration
from diffusiondet.dataset_mapper import DatasetMapper

from diffusiondet.predictor import VisualizationDemo
from diffusiondet import add_diffusiondet_config, DiffusionDetWithTTA
from diffusiondet.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer

'''
This script allows the user to:
1. Obtain the annotations in KITTI format of one or multiple checkpoints, to be evaluated with an external evaluator like https://github.com/cguindel/eval_kitti
2. Visualize and save the images resulting in both BEV and 3D as well
3. Change the evaluation parameters and kitti_root by arguments
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Validation script for BirdNet+')
    parser.add_argument(
        '--config-file', help="Name of the configuration to use without extension", default='diffdetbev.coco.res50', type=str)
    parser.add_argument(
        '--ann_val', help="Validation file with the annotations in COCO format previously generated by the training script, without extension", default='validation_annotations_kitti_carpedcyc_RDHCVPr12', type=str)
    parser.add_argument(
        '--write', help="Write results in KITTI format", default=False, action="store_true")
    parser.add_argument(
        '--img2show', help="Show a fixed number of images, 0 to eliminate the visualization", default=0, type=int)
    parser.add_argument(
        '--save_img', help="Save images showed", default=False, action="store_true")
    parser.add_argument(
        '--eval_chkp', help="Starting from the second half of the checkpoints, the rest will be evaluated with a certain interval specified here, 1 to evaluate all of them", default=1, type=int)
    parser.add_argument(
        '--force_test', help="Name of the checkpoint to extract annotations or evaluate, empty disable this option", default='', type=str)
    parser.add_argument(
        '--score', help="Limitation for lower scores", default=0.01, type=float)
    parser.add_argument(
        '--nms', help="NMS IoU for the overlapping obstacles per class", default=0.3, type=float)
    parser.add_argument(
        '--kitti_root', help="Path of the KITTI dataset", default='/content/DiffusionDetLidar/datasets/bv_kitti/testing', type=str)
    parser.add_argument(
        '--eval_only', help="Write results in KITTI format", default=False, action="store_true")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()



detectron2_root = '/content/DiffusionDetLidar'



# BEV parameters
bvres = 0.05
velodyne_h = 1.73
only_front = True
# BEV images
im_path = os.path.join(detectron2_root,'datasets/bv_kitti/image') 

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

# Viewpoint calculation
def getfrombins(cl,bins):
    bin_dist = np.linspace(-math.pi,math.pi,bins+1)
    bin_res = (bin_dist[1]-bin_dist[0])/2.
    bin = [bin_dist[i]-bin_res for i in range(len(bin_dist)-1)][cl] 
    return bin

idclass = { 0:'Car', 1:'Van', 2:'Truck', 3:'Pedestrian', 4:'Person_sitting',  5:'Cyclist', 6:'Tram', 7:'Misc', 8:'DontCare'}
classtoid3 = { 'Car':0,'Pedestrian':1,'Cyclist':2}
idclass3 = { 0:'Car', 1:'Pedestrian', 2:'Cyclist'}
def catName(category_id,nclass):

    nclass = 3
    # print('number of classes: ',nclass)
    if nclass > 3:
        _idclass = idclass
    elif nclass == 3:
        _idclass = idclass3
    strclass = _idclass.get(category_id, nclass)
    return strclass   

def prepareAnn(lbl, alpha, box, h=-1, w=-1, l=-1, x=-1000, y=-1000, z=-1000, ry=-10, score=None):
    ann = [
       lbl, 
       -1,
       -1,
       alpha,
       box[0],box[1],box[2],box[3],
       h,w,l,
       x,y,z,
       ry
    ]  
    if score is not None:
        ann.append(score)
    strAnn = ' '.join([str(x) for x in ann])
    obj3d = Object3d(strAnn)
     
    return ann, obj3d, strAnn

def prepare_for_coco_detection_KITTI(instance, output_folder, filename, write, kitti_calib_path, nclass, vp, bins, vp_res, hwrot, height_training):
    # Extract important information from instance class
    # print('instance',instance)
    boxes  = np.array(instance.get('pred_boxes').tensor)
    scores = np.array(instance.get('scores'))
    labels = np.array(instance.get('pred_classes'))
    # if 0:
    #     alpha = np.array([rad for rad in instance.get('viewpoint_residual')]) if vp else np.ones((labels.shape))*(-10.00)
    # else:
    #     alpha = np.array([getfrombins(cl,bins) for cl in instance.get('viewpoint')]) if vp else np.ones((labels.shape))*(-10.00)
    alpha = np.ones((labels.shape))*(-12.00)
    h = np.array([[h,g] for h,g in instance.get('pred_height')]) if height_training else np.array([-1,-1000]*labels.shape)

    # Image BV
    bv_image = cv2.imread(filename).astype(np.uint8)

    # print('image size: ',bv_image.size)

    if height_training:
        bv_ground = None
    else:
        # Ground BV
        bv_ground = np.fromfile(os.path.join(im_path,'ground_'+filename[-10:].split('.png')[0]+'.txt'),sep=' ')
        bv_ground = bv_ground.reshape(bv_image.shape[0],bv_image.shape[1],1)
    
    # Calibration for 3D
 
    calib_file = os.path.join(kitti_calib_path.replace('/testing',''),filename[-10:].split('.png')[0]+'.txt')

    # Refiner for 3D
    # print('pixel reslution: ', bvres)
    refiner = BirdviewDetectionRefiner(bv_image, bv_ground, bvres, velodyne_h, only_front)

    im_ann = []
    im_ann_obj = []
    if write:
        file_ann  = open(os.path.join(output_folder,filename[-10:].split('.png')[0]+'.txt'), 'w+')
    for k, box in enumerate(boxes):
        lbl = catName(labels[k],nclass)
        ann,obj3d,strAnn = prepareAnn(lbl,alpha[k],box,score=scores[k],h=h[k,0],z=h[k,1])
        # print('pre refinement: ',strAnn)
      
        

        if hwrot and height_training:
            # print('refine hwrot: ', hwrot)
            # print('refine height: ', height_training)
            refiner.refine_detection_rotated_wheight(obj3d)
        elif hwrot:
            # print('refine hwrot: ', hwrot)
            refiner.refine_detection_rotated(obj3d)
        else:
            # print('refine detection no hwrot or height: ')
            refiner.refine_detection(obj3d)
        if obj3d.height == -1:
            continue

        # Project points to camera frame coordinates
        calib = Calibration(calib_file)
        p = calib.project_velo_to_rect(np.array([[obj3d.location.x,obj3d.location.y,obj3d.location.z]]))

        # Change 2D bbox in BV getting 2D bbox in camera frame (projection)
        _,_,bbox2D = _draw_projection_obstacle_to_cam(obj3d, calib_file, bvres, only_front, False)
        if bbox2D == None:
            continue
        # Obtain alpha from yaw
        obj3d.alpha = obj3d.yaw -(-math.atan2(p[0][2],p[0][0]) - 1.5*math.pi)
        obj3d.alpha = obj3d.alpha%(2*math.pi)
        if obj3d.alpha > math.pi:
            obj3d.alpha -= 2*math.pi
        elif obj3d.alpha < -math.pi:
            obj3d.alpha += 2*math.pi

        # After refinement
        ann = [
               obj3d.kind_name, 
               obj3d.truncated,
               obj3d.occluded,
               round(obj3d.alpha,6),
               round(bbox2D[0],6),round(bbox2D[1],6),round(bbox2D[2],6),round(bbox2D[3],6),
               round(obj3d.height,6), round(obj3d.width,6), round(obj3d.length,6), 
               round(p[0][0],6), round(p[0][1],6), round(p[0][2],6), # Camera coordinates
               round(obj3d.yaw,6),
               obj3d.score, # DON'T ROUND IT
            ]

        im_ann.append(ann)
        im_ann_obj.append(obj3d)
        strAnn = ' '.join([str(x) for x in ann])
        # print('post refinement: ', strAnn)
        if write:
            file_ann.write(strAnn+'\n')
    if write:
        file_ann.close()
    # print(instance)
    return  im_ann, im_ann_obj, instance


# def get_2dbox_iou():


# def get_3dbox_iou();


def main(config_file, ann_val, write, img2show, save_img, eval_chkp, force_test, score_thresh , nms_thresh, kitti_root ,eval_only):
    # KITTI paths
    kitti_im_path = kitti_root+'/image_2'
    kitti_calib_path = kitti_root+'/calib'

    # LOGGER AND CONFIGURATION LOAD
    logger = logging.getLogger("detectron2.trainer")
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.DATASETS.TEST = ("kitti_val",)
    cfg.OUTPUT_DIR = '/content/output_swinbase'
    cfg.VIEWPOINT = False
    cfg.VIEWPOINT_RESIDUAL = False
    cfg.ROTATED_BOX_TRAINING = False

    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 
    # cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
    cfg.merge_from_file(os.path.join(detectron2_root,"configs/{}.yaml".format(config_file)))
    # cfg.freeze()
    default_setup(cfg, None)

    nclasses = cfg.MODEL.ROI_HEADS.NUM_CLASSES
    optional_arguments = []
    if cfg.VIEWPOINT:
        optional_arguments.append('viewpoint')
    if cfg.VIEWPOINT_RESIDUAL:
        optional_arguments.append('vp_res')
    if cfg.ROTATED_BOX_TRAINING:
        optional_arguments.append('bbox3D')
    if 1:
        optional_arguments.append('height')

    val_path = detectron2_root+"/datasets/bv_kitti/annotations/{}.json".format(ann_val)
    register_coco_instances("kitti_val", {}, val_path, detectron2_root+'/datasets/bv_kitti/image', extra_arguments=optional_arguments)
    calib_root_path = '/content/DiffusionDetLidar/datasets/bv_kitti/label'
    toeval = []
    models = os.listdir(cfg.OUTPUT_DIR)
    for model in models:
        if model.endswith('.pth') and not model=='model_final.pth':
            toeval.append(model)
    toeval.sort()
    toeval = toeval[:-1]
    if force_test:
        toeval = [e for e in toeval if force_test in e]
        f_eval = [folder.split('_')[1].split('.')[0] for folder in toeval]
    elif eval_chkp!=0:
        length = len(toeval)//2
        toeval = toeval[length::eval_chkp]
        toeval.append('model_final.pth')
        f_eval = [folder.split('_')[1].split('.')[0] for folder in toeval]
    else:
        toeval = ['model_final.pth']
        f_eval = ['final']
    toeval = ['model_final.pth']
    f_eval = ['final']
    print('Checkpoints to be evaluated: ',toeval)

    # dict to hold scores
    # tp fp fn tn



    eval_res = { 'Car':[0,0,0,0],'Pedestrian':[0,0,0,0],'Cyclist':[0,0,0,0]}



    for checkpoint, eval_folder in zip(toeval,f_eval):
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, checkpoint) 
        
        cfg.MODEL.DiffusionDet.L1_HEIGHT_WEIGHT= (5.0, 0.5, 10.0)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh 
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_thresh
        predictor = DefaultPredictor(cfg)
        #

        val_bv_dicts = DatasetCatalog.get("kitti_val")
        val_bv_meta = MetadataCatalog.get("kitti_val")

        obj_anns = []
        kitti_results = []
        c = 0

        sample_idx = range(img2show) if img2show != 0 else [-1]
        # print('sample indx: ',sample_idx)
        logger.info("Showing {} predictions".format(str(img2show)))
        ann_outdir = os.path.join(cfg.OUTPUT_DIR,'annotations',eval_folder)
        if not os.path.exists(ann_outdir):
            os.makedirs(ann_outdir)

        for image_id, d in enumerate(val_bv_dicts):
            c += 1
            file = os.path.join(ann_outdir,d["file_name"][-10:].split('.png')[0]+'.txt')

            # d['file_name'].replace('/home/alejo')

            im = cv2.imread(d["file_name"])
            print("Preparing prediction {}, from {}, image: {}".format(str(c),str(len(val_bv_dicts)),d["file_name"]))
            if not os.path.exists(file) or write:
                is_kitti_ann=False
                # Inference
                outputs = predictor(im)
                # print('outputs:',outputs)
                list_anns, obj_anns, instances = prepare_for_coco_detection_KITTI(outputs["instances"].to("cpu"), ann_outdir, d["file_name"], write, kitti_calib_path, nclasses, cfg.VIEWPOINT, cfg.VP_BINS, cfg.VIEWPOINT_RESIDUAL, cfg.ROTATED_BOX_TRAINING, cfg.HEIGHT_TRAINING)
                #ground truth
                gt_label_path = os.path.join(calib_root_path,d["file_name"][-10:].split('.png')[0]+'.txt')
                # print( 'gt label: ',gt_label_path)
                f = open(gt_label_path)
                gt_labels = f.readlines()

                f.close()

                gt_objs = []

                for x in gt_labels:
                  # print('raw label: ' ,x)
                  if x.split(' ')[0] ==  'Pedestrian' or x.split(' ')[0] == 'Cyclist' or x.split(' ')[0] == 'Car':
                    temp = Object3d(x)
                    # temp.print_object
                    gt_objs.append(temp)
                #print('ground truth: ')
                for gt in gt_objs:
                  gt.yaw = -10
                  #print('name: ',gt.kind_name,' trunc: ',gt.truncated,' occ: ',gt.occluded,' alpha: ',gt.alpha,' xmin: ',gt.xmin,' ymin: ',gt.ymin,' xmax: ',gt.xmax,' ymax: ',gt.ymax,' height: ',gt.height,' width: ',gt.width,' length: ',gt.length,' yaw: ',gt.yaw)
                #ground truth





                fixed_list_anns = []
                for idx,item in enumerate(list_anns):
                  # print(item)
                  if item[0] == 'Van':
                    list_anns[idx][0] = 'Pedestrian'
                    fixed_list_anns.append(list_anns[idx])
                  elif item[0] == 'Truck':
                    list_anns[idx][0] = 'Cyclist'
                    fixed_list_anns.append(list_anns[idx])
                  elif item[0] == 'Car':
                    fixed_list_anns.append(list_anns[idx])

                for obj in obj_anns:
                  if obj.kind_name == 'Van':
                    obj.kind_name = 'Pedestrian'
           
                  elif obj.kind_name == 'Truck':
                    obj.kind_name = 'Cyclist'

                kitti_results.append(fixed_list_anns)
            else:
                # print('why we here')
                is_kitti_ann=True
                with open(file,'r') as f:
                    list_anns = f.read().splitlines()
                kitti_results.append([anns.split(' ') for anns in list_anns] if list_anns else [])
                for ann in list_anns:
                    obj_anns.append(Object3d(ann))

            # print('c value: ', c)
            if c in sample_idx:
                # print('c in smaple idx')
                # Change BV aspect
                nonzero = np.where(im>0)
                im[nonzero]=255-im[nonzero]
                im=cv2.bitwise_not(im)

                kitti_im = cv2.imread(os.path.join(kitti_im_path,d["file_name"][-10:]))
                calib_file = os.path.join(kitti_calib_path.replace('/testing',''),d["file_name"][-10:].split('.png')[0]+'.txt')
                # Show obstacles

                # for j, gt in enumerate(gt_objs):
                #   print('ground tuth: ',j,' ')
                #   kitti_im, im, _ = _draw_projection_obstacle_to_cam(gt, calib_file, bvres, only_front, True, kitti_im, im, is_kitti_ann=True)
                #   gt.print_object()
                # tp fp fn tn


                for i, obj in enumerate(obj_anns):

                    #obj.print_object()
                    fn_flag = True
                    for gt in gt_objs:
                      if gt.kind_name == obj.kind_name and obj.score >= score_thresh:
                        fn_flag = False
                        eval_res[gt.kind_name][0] +=1
                        if save_img:
                          kitti_im, im, _ = _draw_projection_obstacle_to_cam(obj, calib_file, bvres, only_front, True, kitti_im, im, is_kitti_ann=is_kitti_ann)

                          im_outdir = os.path.join(cfg.OUTPUT_DIR,'images')
                          if not os.path.exists(im_outdir):
                              os.makedirs(im_outdir)
                          cv2.imwrite(os.path.join(im_outdir,'3D_'+d["file_name"][-10:]), kitti_im)
                          cv2.imwrite(os.path.join(im_outdir,'BEV_'+d["file_name"][-10:]), im)
                          
                      elif gt.kind_name == obj.kind_name and obj.score < score_thresh:
                        eval_res[gt.kind_name][1] +=1

                      

                    if fn_flag and obj.score >= score_thresh:
                      eval_res[gt.kind_name][2] +=1

   
            elif c > max(sample_idx) and not write:
                # print('c value bigger than max index: ',c)
                break
    print(eval_res)
    # tp / fn + tp
    # print('car recall: ',eval_res['Car'][0]/(eval_res['Car'][0] + eval_res['Car'][2])*100)
    # print('Pedestrian recall: ',eval_res['Pedestrian'][0]/(eval_res['Pedestrian'][0] + eval_res['Pedestrian'][2])*100)
    # print('Cyclist recall: ',eval_res['Cyclist'][0]/(eval_res['Cyclist'][0] + eval_res['Cyclist'][2])*100)
    # # tp/tp+fp
    # print('car AP: ',eval_res['Car'][0]/(eval_res['Car'][0] + eval_res['Car'][1])*100)
    # print('Pedestrian AP: ',eval_res['Pedestrian'][0]/(eval_res['Pedestrian'][0] + eval_res['Pedestrian'][1])*100)
    # print('Cyclist AP: ',eval_res['Cyclist'][0]/(eval_res['Cyclist'][0] + eval_res['Cyclist'][1])*100)
if __name__ == '__main__':
    args = parse_args()

    main(args.config_file, args.ann_val, args.write, args.img2show, args.save_img, args.eval_chkp, args.force_test, args.score, args.nms, args.kitti_root,args.eval_only)