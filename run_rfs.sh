#!/usr/bin/env bash

ROOT=/media/datasets/gpu17_models/WEN/checkpoints/rfs/faster_rcnn #<- Change this yourself

#------------------------------ Base-training ------------------------------- #
for split in 1 2 3
do
    python tools/train_net.py --num-gpus 3 \
        --config-file configs/RFS/base-training/R101_FPN_base_training_split${split}.yml \
        --opts OUTPUT_DIR ${ROOT}/faster_rcnn_R_101_FPN_base${split}
done

#------------------------------ Random initialize ------------------------------- #
for split in 1 2 3
do
    python tools/ckpt_surgery.py \
       --src1 ${ROOT}/faster_rcnn_R_101_FPN_base${split}/model_final.pth \
       --method randinit \
       --save-dir ${ROOT}/faster_rcnn_R_101_FPN_base${split}
done

#------------------------------ Fine-tuning ------------------------------- #
for split in 1 2 3
do
    for shot in 1 2 3 5 10  
    do
        echo split:$split shot:$shot
        CONFIG_PATH=configs/RFS/split${split}/${shot}shot_GPB_PFB_proloss.yml
        OUT_DIR=${ROOT}/split${split}_${shot}shot_GPB_PFB_proloss
        python3 -m tools.train_net --num-gpus 3 \
	        --config-file ${CONFIG_PATH} \
            --opts OUTPUT_DIR ${OUT_DIR}
        rm ${OUT_DIR}/last_checkpoint            
    done
done

#------------------------------ Evaluating ------------------------------- #
for split in 1 2 3
do
    for shot in 1 2 3 5 10  
    do
        echo split:$split shot:$shot
        CONFIG_PATH=configs/RFS/split${split}/${shot}shot_GPB_PFB_proloss.yml
        OUT_DIR=${ROOT}/split${split}_${shot}shot_GPB_PFB_proloss
        python3 -m tools.train_net --num-gpus 3 \
	        --config-file ${CONFIG_PATH} --eval-only \
            --opts MODEL.WEIGHTS ${ROOT}/model_final.pth \
               MODEL.MODEL.ROI_HEADS.NOVEL_MODULE.INIT_FEATURE_WEIGHT ${ROOT}/prototypes_feature_final.pkl           
    done
done

