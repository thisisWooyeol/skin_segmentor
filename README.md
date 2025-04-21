# Skin Segmentor with Mask2Former

This repository provides a skin segmentor based on the Mask2Former framework. As an extension of the original Mask2Former, this implementation is specifically designed for three classes: acne, hemoglobin and melanin. 

# Get Started
## 0. Installation

```bash
git clone https://github.com/thisisWooyeol/skin_segmentor.git
cd skin_segmentor
uv sync
```

## 1. Preprocess mask

Recommended threshold value for each class is as follows:
- acne: 20
- hemo: 7
- mela: 20

```bash
python src/utils/generate_mask.py --dataset_dir <dataset_dir> --threshold <threshold>
```


# References

[1] [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527)

# Acknowledgements

2025 Spring, Creative Integrated Design 2 class, Seoul National University, South Korea.
This project is supported by the SNU Creative Integrated Design 2 class and [Aram Huvis Co., Ltd](https://www.aramhuvis.com/). Collaborative effort were made with the following students:

- Minseo Kim
- Byeongho Park