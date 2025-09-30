import numpy as np
import os
from PIL import Image
from sklearn.neighbors import KNeighborsRegressor

def add_white_background(img: Image.Image) -> Image.Image:
    # Ensure we have alpha for compositing
    img = img.convert("RGBA")
    # Create a white background in the same size
    white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    # Composite the original image over the white background
    combined = Image.alpha_composite(white_bg, img)
    # Drop the alpha channel so the result stays opaque
    return combined.convert("RGB")

def preprocess_image(inputdata):
    img_pil = Image.open(inputdata)
    img_pil = add_white_background(img_pil)
    img_pil = img_pil.resize((240, 240))
    img_lab = img_pil.convert("LAB")
    return img_lab.split()[0]


# Load L/A/B channels from disk once
with np.load(os.path.join(os.getcwd(), 'transformed_data.npz'), allow_pickle=True) as data:
    TRAIN_L_CHANNEL = data['L'].astype(np.float32)
    TRAIN_A_CHANNEL = data['A'].astype(np.float32)
    TRAIN_B_CHANNEL = data['B'].astype(np.float32)

def predict_channel(l_values, target_channel, n_neighbors=10):
    num_pixels = l_values.shape[0]
    predicted_channel = np.empty((num_pixels, 1), dtype=np.float32)

    for pixel_idx in range(num_pixels):
        X_pixel = TRAIN_L_CHANNEL[:, pixel_idx].reshape(-1, 1)
        Y_pixel = target_channel[:, pixel_idx]

        knn_model = KNeighborsRegressor(n_neighbors=n_neighbors)
        knn_model.fit(X_pixel, Y_pixel)

        predicted_value = knn_model.predict(l_values[pixel_idx].reshape(1, -1))[0]
        predicted_channel[pixel_idx, 0] = float(predicted_value)

        if (pixel_idx + 1) % 10000 == 0:
            print(f"Finished predicting {pixel_idx + 1}/{num_pixels} pixel models")

    return predicted_channel

if __name__ == '__main__':
    inputdata = 'inputphoto2.jpg'
    L_img = preprocess_image(inputdata)
    l_channel_uint8 = np.array(L_img, dtype=np.uint8).reshape(-1, 1)
    l_channel_float = l_channel_uint8.astype(np.float32)

    predicted_a = predict_channel(l_channel_float, TRAIN_A_CHANNEL)
    predicted_b = predict_channel(l_channel_float, TRAIN_B_CHANNEL)

    L_plane = l_channel_uint8.reshape(240, 240)
    A_plane = np.clip(predicted_a, 0, 255).astype(np.uint8).reshape(240, 240)
    B_plane = np.clip(predicted_b, 0, 255).astype(np.uint8).reshape(240, 240)

    lab_image = Image.merge(
        'LAB',
        (
            Image.fromarray(L_plane, mode='L'),
            Image.fromarray(A_plane, mode='L'),
            Image.fromarray(B_plane, mode='L'),
        ),
    )

    lab_image.convert('RGB').save('outputphoto2.png', 'PNG')
    print("Colorized image saved as 'outputphoto2.png'")
