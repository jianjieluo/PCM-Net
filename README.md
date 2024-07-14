# Unleashing Text-to-Image Diffusion Prior for Zero-Shot Image Captioning [ECCV24]

This is the official repository for **Unleashing Text-to-Image Diffusion Prior for Zero-Shot Image Captioning [ECCV24]**, mainly for the proposed framework PCM-Net.

## Overview
Zero-shot image captioning is a challenging task where only text data is available for training. While recent advancements in text-to-image diffusion models have enabled the generation of synthetic image-caption pairs, these pairs often suffer from defective details in salient regions, leading to semantic misalignment.

PCM-Net introduces a Patch-wise Cross-modal feature Mixup (PCM) mechanism that adaptively mitigates unfaithful content in a fine-grained manner during training. This mechanism can be integrated into most encoder-decoder frameworks.

## PCM-Net Framework
![pcmnet](imgs/framework.jpg)

- **Salient Visual Concept Detection:** For each input image, salient visual concepts are detected based on image-text similarity in CLIP space.
- **Patch-wise Feature Fusion:** Selectively fuses patch-wise visual features with textual features of salient concepts, creating a mixed-up feature map with reduced defects.
- **Visual-Semantic Encoding:** A visual-semantic encoder refines the feature map, which is then used by the sentence decoder for generating captions.
- **CLIP-weighted Cross-Entropy Loss:** A novel loss function prioritizes high-quality image-text pairs over low-quality ones, enhancing model training with synthetic data.


## Data Preparation

[SynthImgCap Dataset](https://jianjieluo.github.io/SynthImgCap/#SynthImgCap) is available.

META ANNO DATA and will be released soon...


## Training
Please refer to `scripts/train.sh`.


## Inference
Please refer to `scripts/final_eval_for_paper.sh`.


## Citation
If you use the SynthImgCap dataset or code or models for your research, please cite:

```
@inproceedings{luo2024unleashing,
    title = {Unleashing Text-to-Image Diffusion Prior for Zero-Shot Image Captioning},
    author = {Luo, Jianjie and Chen, Jingwen and Li, Yehao and Pan, Yingwei and Feng, Jianlin and Chao, Hongyang and Yao, Ting},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2024}
}
```

## Acknowledgement
This code used resources from [X-Modaler Codebase](https://github.com/YehLi/xmodaler) and [DenseCLIP code](https://github.com/raoyongming/DenseCLIP). We thank the authors for open-sourcing their awesome projects.


## License

MIT