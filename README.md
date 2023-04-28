# DiffusionDetLidar

In order to run use the colab notebook file

there will be comments before each cell to run and how to obtain any data but in a nutshell here are the steps


1. first you will need to setup detectron2, utilize req.txt to setup your environment: 
	example: conda create -n <environment-name> --file req.txt

2. clone my github repo containiing the code

3. You will need to download the pre-generated dataset into the colab note book and save it under ~/datasets (this will take a bit)
	these is a smaller file only containing ten images that can also be utilize

4. you will need to download both pretrained weights for resnet101 and swinbase
	this should be on a share link in onedrive, save them to ~/DiffusionDetLidar/ directory 

5. you will then need to make sure you run the commands in the ~/ directory in other words on the same level as DiffusionDetLidar directory

6. last in the results.py python file change the global variable 'detectron2_root'  on line 71 to the complete path of the DiffusionDetLidar directory

7. you can also load the trained weights into the model and save them at the same level as the DiffusionDetLidar Folder, the train weights must be in a folder call output_resnet101 or output_swinbase

8. to run eval on resnet101 

python DiffusionDetLidar/results.py --config-file diffdetbev.coco.res101 --weights_dir DiffusionDetLidar/output_resnet101 --force_test model_final.pth --eval_only --img2show 100 --score 0.04 --nms 0.5 --kitti_root /home/azamora/DiffusionDetLidar/datasets/bv_kitti/testing

to run eval on swinbase

python DiffusionDetLidar/results.py --config-file diffdetbev.coco.swinbase --weights_dir DiffusionDetLidar/output_swinbase --force_test model_final.pth --eval_only --img2show 100 --score 0.04 --nms 0.5 --kitti_root /home/azamora/DiffusionDetLidar/datasets/bv_kitti/testing


Note: im2show will run the evaluation on 1 to n, n being the numbder you specified


 