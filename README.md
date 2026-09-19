# BW to Color Converter

This repo contains two BW-to-color experiments:

- `BWtoColorKNN.py`: the original pixel-wise KNN approach.
- `linear_color_model.py`: a deploy-friendly pixel-wise linear model.

## Setup

```sh
python3 -m pip install -r requirements.txt
```

## Train the Linear Model

Put training images in a `data/` folder, then create the shared transformed
training data:

```sh
python3 transformtrainingdata.py
```

Then learn the compact linear model from `transformed_data.npz`:

```sh
python3 linear_color_model.py train \
  --transformed-data transformed_data.npz \
  --output linear_color_model.npz
```

This learns four parameters per pixel:

- slope/intercept for the LAB `A` channel
- slope/intercept for the LAB `B` channel

The saved `.npz` file is the deployable model artifact.

You can also train directly from images if you want to skip the intermediate
file:

```sh
python3 linear_color_model.py train --data data --output linear_color_model.npz
```

## Colorize an Image

```sh
python3 linear_color_model.py colorize \
  --input inputphoto2.jpg \
  --model linear_color_model.npz \
  --output linear_output.png
```
