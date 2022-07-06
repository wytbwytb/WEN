python tools/test_net.py --num-gpus 1 \
        --config-file configs/PASCAL_VOC/split2/5shot_GPB_PFB_proloss.yml \
        --eval-only \
        MODEL.WEIGHTS /media/datasets/gpu17_models/Ours/checkpoints/voc/faster_rcnn/split2_5shot_GPB_PFB_proloss/428.pth \
		MODEL.ROI_HEADS.NOVEL_MODULE.INIT_FEATURE_WEIGHT /media/datasets/gpu17_models/Ours/checkpoints/voc/faster_rcnn/split2_5shot_GPB_PFB_proloss/428.pkl
