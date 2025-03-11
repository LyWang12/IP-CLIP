# Vision-Language Model IP Protection via Prompt-based Learning

Code release for "Vision-Language Model IP Protection via Prompt-based Learning" (CVPR 2025)

## Paper

<div align=center><img src="https://github.com/LyWang12/IP-CLIP/blob/main/Figure/1.png" width="100%"></div>


[Vision-Language Model IP Protection via Prompt-based Learning](https://arxiv.org/abs/2503.02393) 
(CVPR2025 22024)

We propose IP-CLIP,  a lightweight ntellectual property (IP) protection strategy tailored to CLIP to protect Vision-Language Models.

## Prerequisites
The code is implemented with **CUDA 11.4**, **Python 3.8.5** and **Pytorch 1.8.0**.

## Datasets

### Office-31
Office-31 dataset can be found [here](https://opendatalab.com/OpenDataLab/Office-31).

### Office-Home
Office-Home dataset can be found [here](http://hemanthdv.org/OfficeHome-Dataset).

### Mini-DomainNet
Mini-DomainNet dataset can be found [here](https://github.com/KaiyangZhou/Dassl.pytorch).


## Running the code

Target-Specified IP-CLIP
```
python Target_Specified/train.py
```

Ownership Verification by IP-CLIP
```
python Ownership/train.py
```

Target-free IP-CLIP
```
python Target_Free/train.py
```

Applicability Authorization by IP-CLIP
```
python Authorization/train.py
```

## Citation
If you find this code useful for your research, please cite our [paper](https://arxiv.org/abs/2503.02393):
```
@article{wang2025vision,
  title={Vision-Language Model IP Protection via Prompt-based Learning},
  author={Wang, Lianyu and Wang, Meng and Fu, Huazhu and Zhang, Daoqiang},
  journal={arXiv preprint arXiv:2503.02393},
  year={2025}
}
```

## Contact
If you have any problem about our code, feel free to contact
- lywang12@126.com
- wangmeng9218@126.com



