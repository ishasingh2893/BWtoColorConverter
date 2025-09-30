import os
import numpy as np
from PIL import Image

def add_white_background(img: Image.Image) -> Image.Image:
    # Ensure we have alpha for compositing
    img = img.convert("RGBA")
    # Create a white background in the same size
    white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    # Composite the original image over the white background
    combined = Image.alpha_composite(white_bg, img)
    # Drop the alpha channel so the result stays opaque
    return combined.convert("RGB")

def read_images_to_matrix(data_folder):
    data = []
    for filename in os.listdir(data_folder):
        img = Image.open(os.path.join(data_folder, filename))
        img = add_white_background(img)
        img = img.resize((240, 240))
        img = img.convert('LAB')
        L = img.split()[0]  # L channel
        L = np.array(L).flatten()
        A = img.split()[1]  # A channel
        A = np.array(A).flatten()
        B = img.split()[2]  # B channel
        B = np.array(B).flatten()
        data.append((L, A, B))
    return data

def transform_training_data(data_folder):
    data = read_images_to_matrix(data_folder)
    L, A, B = zip(*data)
    return np.array(L), np.array(A), np.array(B)

def save_transformed_data(data, output_file):
    L, A, B = data
    np.savez(output_file, L=L, A=A, B=B)

if __name__ == "__main__":
    data_folder = os.path.join(os.getcwd(), "data")
    output_file = "transformed_data"

    L, A, B = transform_training_data(data_folder)
    save_transformed_data((L, A, B), output_file)
