# Sparsifiner: Learning Sparse Instance-Dependent Attention for Efficient Vision Transformers

This repository contains PyTorch implementation for Sparsifiner (CVPR 2023).

[[Project Page]](https://lim142857.github.io/lim142857.github.io-sparsifiner/) [[arXiv (CVPR 2023)]](https://arxiv.org/abs/2303.13755)

## Usage

### Requirements

- torch>=1.8.1
- torchvision>=0.9.1
- timm==0.3.2
- tensorboardX
- six
- fvcore

**Data preparation**: download and extract ImageNet images from http://image-net.org/. The directory structure should be

```
│ILSVRC2012/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

**Model preparation**: download pre-trained models if necessary:

| model | url | model | url |
| --- | --- | --- | --- |
| DeiT-Small | [link](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) | LVViT-S | [link](https://github.com/zihangJiang/TokenLabeling/releases/download/1.0/lvvit_s-26M-224-83.3.pth.tar) |
| DeiT-Base | [link](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth) | LVViT-M | [link](https://github.com/zihangJiang/TokenLabeling/releases/download/1.0/lvvit_m-56M-224-84.0.pth.tar) |


[//]: # (### Evaluation)

[//]: # ()
[//]: # (To evaluate a pre-trained Sparsifiner model on the ImageNet validation set with a single GPU, run:)

[//]: # ()
[//]: # (```)

[//]: # (# )

[//]: # (```)


### Training

To train a Sparsifiner model with default configuration on ImageNet, run:

**DeiT-S**

Train on 8 GPUs
```
bash run_model.sh --IMNET sparsifiner_default 8
```

## License

MIT License

## Acknowledgements

Our code is based on [DynamicVit](https://github.com/raoyongming/DynamicViT), [pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [DeiT](https://github.com/facebookresearch/deit), [LV-ViT](https://github.com/zihangJiang/TokenLabeling)

## Citation
If you find our work useful in your research, please consider citing:
```
@InProceedings{Wei_2023_CVPR,
    author    = {Wei, Cong and Duke, Brendan and Jiang, Ruowei and Aarabi, Parham and Taylor, Graham W. and Shkurti, Florian},
    title     = {Sparsifiner: Learning Sparse Instance-Dependent Attention for Efficient Vision Transformers},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {22680-22689}
}
```