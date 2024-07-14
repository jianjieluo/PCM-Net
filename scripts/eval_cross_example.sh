# MSCOCO -> Flickr30k
CUDA_VISIBLE_DEVICES=0 python test_net.py --num-gpus 1 \
    --config-file configs/mscoco_sd/vit_l14/flickr30k_cross_domain_eval.yaml \
    MODEL.WEIGHTS {path_to_mscoco_sd_model} \
    OUTPUT_DIR {path_to_output}


# Flickr30k -> MSCOCO
CUDA_VISIBLE_DEVICES=0 python test_net.py --num-gpus 1 \
    --config-file configs/flickr_sd/vit_l14/mscoco_cross_domain_eval.yaml \
    MODEL.WEIGHTS {path_to_flickr_sd_model} \
    OUTPUT_DIR {path_to_output}
