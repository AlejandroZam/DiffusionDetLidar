# DiffusionDetLidar

In order to run use the colab notebook file

there will be comments before each cell to run and how to obtain any data but in a nutshell here are the steps


1. first you will need to setup detectron2 on colab this can be done running the first two cells

2. clone my github repo containiing the code

3. You will need to download the pre-generated dataset into the colab note book and save it under /content/datasets (this will take a bit)

4. you will need to download both pretrained weights for resnet101 and swinbase

5. you will then need to make sure you cd into DiffusionDetLidar/diffusiondet/util folder 

6. now there will be several commands in the cells you can try. 

7. you can also load the trained weights into the model and save them at the same level as the DiffusionDetLidar Folder

-- !python /content/DiffusionDetLidar/train_bev_net.py --num-gpus 1 --config-file /content/DiffusionDetLidar/configs/diffdetbev.coco.res101.yaml
will run the training for resnet101

-- !python /content/DiffusionDetLidar/train_bev_net.py --num-gpus 1 --config-file /content/DiffusionDetLidar/configs/diffdetbev.coco.swinbase.yaml
will run the training for swinbase

-- !python /content/DiffusionDetLidar/results.py --config-file diffdetbev.coco.res101 --eval_only --force_test model_final.pth --img2show 3769 --score 0.04 --nms 0.5
  will run the evaluation for resnet101

-- !python /content/DiffusionDetLidar/results.py --config-file diffdetbev.coco.swinbase --eval_only --force_test model_final.pth --img2show 3769 --score 0.04 --nms 0.5
  will run the evaluation for swinbase
  
  
