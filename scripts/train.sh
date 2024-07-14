############# MSCOCO-SD In-domain #############

# ViT-B/32
CUDA_VISIBLE_DEVICES=0 python train_net.py --num-gpus 1 \
    --config-file configs/mscoco_sd/vit_b32/train.yaml \
    OUTPUT_DIR output/mscoco_sd/vit_b32/pcm_net


# ViT-L/14
CUDA_VISIBLE_DEVICES=0 python train_net.py --num-gpus 1 \
    --config-file configs/mscoco_sd/vit_l14/train.yaml \
    OUTPUT_DIR output/mscoco_sd/vit_l14/pcm_net


############# Flickr30k In-domain #############

# ViT-B/32
CUDA_VISIBLE_DEVICES=0 python train_net.py --num-gpus 1 \
    --config-file configs/flickr_sd/vit_b32/train.yaml \
    OUTPUT_DIR output/flickr_sd/vit_b32/pcm_net


# ViT-L/14
CUDA_VISIBLE_DEVICES=0 python train_net.py --num-gpus 1 \
    --config-file configs/flickr_sd/vit_l14/train.yaml \
    OUTPUT_DIR output/flickr_sd/vit_l14/pcm_net

