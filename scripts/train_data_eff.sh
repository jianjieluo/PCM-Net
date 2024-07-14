############# COCO In-domain DATA EFFICENT #############

CUDA_VISIBLE_DEVICES=0 python train_net.py --num-gpus 1 \
    --config-file configs/mscoco_sd/vit_b32/train_data_eff.yaml \
    OUTPUT_DIR output/vit_b32/final_data_eff/rand_ratio_0.8 \
    DATALOADER.DATA_EFF_RATIO 0.8

CUDA_VISIBLE_DEVICES=0 python train_net.py --num-gpus 1 \
    --config-file configs/mscoco_sd/vit_b32/train_data_eff.yaml \
    OUTPUT_DIR output/vit_b32/final_data_eff/rand_ratio_0.6 \
    DATALOADER.DATA_EFF_RATIO 0.6

CUDA_VISIBLE_DEVICES=0 python train_net.py --num-gpus 1 \
    --config-file configs/mscoco_sd/vit_b32/train_data_eff.yaml \
    OUTPUT_DIR output/vit_b32/final_data_eff/rand_ratio_0.4 \
    DATALOADER.DATA_EFF_RATIO 0.4

CUDA_VISIBLE_DEVICES=0 python train_net.py --num-gpus 1 \
    --config-file configs/mscoco_sd/vit_b32/train_data_eff.yaml \
    OUTPUT_DIR output/vit_b32/final_data_eff/rand_ratio_0.2 \
    DATALOADER.DATA_EFF_RATIO 0.2

CUDA_VISIBLE_DEVICES=0 python train_net.py --num-gpus 1 \
    --config-file configs/mscoco_sd/vit_b32/train_data_eff.yaml \
    OUTPUT_DIR output/vit_b32/final_data_eff/rand_ratio_0.1 \
    DATALOADER.DATA_EFF_RATIO 0.1
