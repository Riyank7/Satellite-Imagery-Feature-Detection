# Satellite-Imagery-Feature-Detection

## Introduction

* In this project, we aim to achieve segmentation of satellite image to detect various features in the image, as part of the [DSTL challenge on Kaggle.](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection#description)

## Compatibility

* This code has been tested on Ubuntu 16.04 LTS and mac-OSX.
* **Dependencies** - Python 3.5+, OpenCV 3.0+, Tensorflow.

## Trained Checkpoint

* We have trained the model for a specific class of buildings,
* You can download the checkpoint [here.](https://drive.google.com/drive/folders/1T02s5ABQDATvnqdJUBOpO5PU8qsY5dxX?usp=sharing)

## Usage

* Clone this repository by typing `git clone https://github.com/atulapra/Satellite-Imagery-Feature-Detection.git` in the terminal.

* Enter the cloned folder.

* Download the three band data from [here](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data) and unzip it.

* Download `grid_sizes.csv` from [here](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data) and unzip it.

* Download `train_wkt_v4.csv` from [here](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data) and unzip it.

* Place `grid_sizes.csv` and `train_wkt_v4.csv` in the folder `data`.

* Execute `tiff_read.py` as `python tiff_read.py`. The masks are saved in the folder `Masks`.

* Then, run `model.py` as `python model.py`.

* When you run it in **train mode**, training process takes place and checkpoint is saved in outputs folder.

* When you run it in **test mode**, the output images are saved in outputs folder.