
########################### Table 1 ###########################

# MSCOCO ViT-B32
CUDA_VISIBLE_DEVICES=0 python test_net.py --num-gpus 1 \
    --config-file configs/mscoco_sd/vit_b32/train.yaml \
    MODEL.WEIGHTS release_models/in_domain/mscoco/vit_b32/model_Epoch_00010_Iter_0169499.pth \
    OUTPUT_DIR output_final_eval/in_domain/mscoco/vit_b32


# MSCOCO ViT-L14
CDUA_VISIBLE_DEVICES=0 python test_net.py --num-gpus 1 \
    --config-file configs/mscoco_sd/vit_l14/train.yaml \
    MODEL.WEIGHTS release_models/in_domain/mscoco/vit_l14/model_Epoch_00006_Iter_0101699.pth \
    OUTPUT_DIR output_final_eval/in_domain/mscoco/vit_l14


# Flickr30k ViT-B32
CDUA_VISIBLE_DEVICES=0 python test_net.py --num-gpus 1 \
    --config-file configs/flickr_sd/vit_b32/train.yaml \
    MODEL.WEIGHTS release_models/in_domain/flickr/vit_b32/model_Epoch_00011_Iter_0049675.pth \
    OUTPUT_DIR output_final_eval/in_domain/flickr/vit_b32


# Flickr30k ViT-L14
CDUA_VISIBLE_DEVICES=0 python test_net.py --num-gpus 1 \
    --config-file configs/flickr_sd/vit_l14/train.yaml \
    MODEL.WEIGHTS release_models/in_domain/flickr/vit_l14/model_Epoch_00009_Iter_0040643.pth \
    OUTPUT_DIR output_final_eval/in_domain/flickr/vit_l14


########################### Table 2 ###########################

################# MSCOCO --> Flickr30k
# ViT-B32
CUDA_VISIBLE_DEVICES=0 python test_net.py --num-gpus 1 \
    --config-file configs/mscoco_sd/vit_b32/flickr30k_cross_domain_eval.yaml \
    MODEL.WEIGHTS release_models/cross_domain/coco_to_flickr/vit_b32/model_Epoch_00007_Iter_0118649.pth \
    OUTPUT_DIR output_final_eval/cross_domain/coco_to_flickr/vit_b32


# ViT-L14
CUDA_VISIBLE_DEVICES=0 python test_net.py --num-gpus 1 \
    --config-file configs/mscoco_sd/vit_l14/flickr30k_cross_domain_eval.yaml \
    MODEL.WEIGHTS release_models/cross_domain/coco_to_flickr/vit_l14/model_Epoch_00006_Iter_0101699.pth \
    OUTPUT_DIR output_final_eval/cross_domain/coco_to_flickr/vit_l14


################# Flickr30k --> MSCOCO

# ViT-B32
CUDA_VISIBLE_DEVICES=0 python test_net.py --num-gpus 1 \
    --config-file configs/flickr_sd/vit_b32/mscoco_cross_domain_eval.yaml \
    MODEL.WEIGHTS release_models/cross_domain/flickr_to_coco/vit_b32/model_Epoch_00010_Iter_0045159.pth \
    OUTPUT_DIR output_final_eval/cross_domain/flickr_to_coco/vit_b32


# ViT-L14
CUDA_VISIBLE_DEVICES=0 python test_net.py --num-gpus 1 \
    --config-file configs/flickr_sd/vit_l14/mscoco_cross_domain_eval.yaml \
    MODEL.WEIGHTS release_models/cross_domain/flickr_to_coco/vit_l14/model_Epoch_00009_Iter_0040643.pth \
    OUTPUT_DIR output_final_eval/cross_domain/flickr_to_coco/vit_l14
