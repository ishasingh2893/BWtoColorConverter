# BW to Color Converter

This repo contains two BW-to-color experiments:

- `BWtoColorKNN.py`: the original pixel-wise KNN approach.
- `linear_color_model.py`: a deploy-friendly pixel-wise linear model.

## Setup

```sh
python3 -m pip install -r requirements.txt
```

## Train the Linear Model

This project expects a local `data/` folder of color face portraits. If you are
using the Kaggle football-player face dataset, first install dependencies, then
run:

```sh
python3 download_training_data.py
```

The script downloads `saiharim/fifa-player-faces` with KaggleHub, recursively
finds image files, and copies valid images into `data/`. By default it copies
200 images, which keeps the transformed matrix and regression training small
enough for local iteration.

Then create the shared transformed training data:

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

## Run the Website

The Flask website lets a user upload a black-and-white portrait or paste a
photo link, then returns a colorized PNG from `linear_color_model.npz`.

```sh
python3 app.py
```

Open `http://127.0.0.1:5000`.

Upload a portrait or load a photo link, then drag the image inside the square
and adjust the zoom slider to align the face with the guide. Arrow keys move
the image; Shift + arrow keys move it farther. Reset restores the initial fit.
The guide uses the target face size and center in `face_crop_calibration.json`.
White space outside the photo becomes white padding.

Colorization uses exactly the chosen 240 × 240 crop for linear, KNN, and hybrid
outputs. The website does not automatically detect, recrop, or segment the face.
The guide is an alignment aid and is never included in the submitted image.

Run the manual crop integration checks (with the model artifacts present):

```sh
.venv/bin/python -m unittest discover -s tests
```

The web app requires `linear_color_model.npz`. If that file is missing, the UI
will load but colorization requests will show a model-artifact error.
