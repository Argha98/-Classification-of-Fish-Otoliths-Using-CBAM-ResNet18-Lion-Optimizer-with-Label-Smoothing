# Otolith identification

# Note
Proper data cleaning and image preprocessing is required to get best results. 
## Install

Install [conda](https://docs.conda.io/en/latest/miniconda.html) and run the following script for local installation

```bash
 git clone https://github.com/Argha98/-Classification-of-Fish-Otoliths-Using-CBAM-ResNet18-Lion-Optimizer-with-Label-Smoothing/tree/main/otoliths-identification
 cd otoliths-identification
 conda env create -f environment.yml
 conda activate otoliths-identification-env
```

## Data preprocessing

All references images should be located in `./data/Reference pictures/`. Run the following script to split data into training and valid sets:

```bash
python preprocess.py
```

There will be two folders `./data/train` and `./data/valid/` containing training and valid data, respectively.

Note:  Some images might not be well segmented due to the quality of images, please remove them to clean the data.

## Test segment
After completing preprocess.py (which prepares train/valid from “Reference pictures”) and before making predictions, if you want a separate, pre-segmented test set for QC or batch reuse.

The script scans SAMPLE_DIR (default ./data/Scheelhoek samples 2017), finds .jpg and .tif images, segments each photo into regions, and writes cropped otoliths into TEST_DIR (default ./data/test) after recreating that folder.

Note - Ensure your raw sample photos are placed under ./data/Scheelhoek samples 2017, keeping any subfolder structure; confirm SAMPLE_DIR and TEST_DIR values in util/useful_imports.py match your layout.

```bash
python test_segment.py
```

## Train the model

```bash
python main.py --arch resnet18 --use_attention --pretrained --train -img_size 224 -b 16 -j 4 -epochs 150 -lr 0.00001 -lr_patience 10 -early_stop 20 -freeze 0 --weight_decay 0.01 -message "Attention_ResNet18_Lion_Optimizer"
```

## Predict the labels

```bash
python main.py --arch resnet18 --use_attention --pretrained --test -img_size 224 -b 16 -j 4
```

## TSNE

```bash
python make_tsne.py
```

## Highlight

```bash
python visualization.py --use_attention
```


