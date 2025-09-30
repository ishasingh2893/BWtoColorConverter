from typing import Optional

import numpy as np
from PIL import Image
import plotly.graph_objects as go

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
    return img_pil.convert("RGB")


def plot_random_rgb_points(
    img_rgb: Image.Image,
    sample_size: int = 100,
    seed: Optional[int] = None,
    output_path: str = "rgb_sample_scatter.html",
) -> str:
    """Create an interactive 3D scatter plot of RGB coordinates sampled from the image."""

    rgb_array = np.array(img_rgb.convert("RGB"))
    pixels = rgb_array.reshape(-1, 3)

    rng = np.random.default_rng(seed)
    sample_size = min(sample_size, pixels.shape[0])
    sample_indices = rng.choice(pixels.shape[0], size=sample_size, replace=False)
    sampled_rgb = pixels[sample_indices]

    color_strings = [f"rgb({r},{g},{b})" for r, g, b in sampled_rgb]
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=sampled_rgb[:, 0],
                y=sampled_rgb[:, 1],
                z=sampled_rgb[:, 2],
                mode="markers",
                marker=dict(size=4, color=color_strings, line=dict(width=0)),
                hovertemplate="R=%{x}<br>G=%{y}<br>B=%{z}<extra></extra>",
            )
        ]
    )

    axis_range = np.linspace(0, 255, 60)
    fig.add_trace(
        go.Scatter3d(
            x=axis_range,
            y=np.zeros_like(axis_range),
            z=np.zeros_like(axis_range),
            mode="lines",
            line=dict(
                width=36,
                color=axis_range,
                colorscale=[[0, "rgb(255,0,0)"], [1, "rgb(255,255,255)"]],
                cmin=0,
                cmax=255,
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=np.zeros_like(axis_range),
            y=axis_range,
            z=np.zeros_like(axis_range),
            mode="lines",
            line=dict(
                width=36,
                color=axis_range,
                colorscale=[[0, "rgb(0,255,0)"], [1, "rgb(255,255,255)"]],
                cmin=0,
                cmax=255,
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=np.zeros_like(axis_range),
            y=np.zeros_like(axis_range),
            z=axis_range,
            mode="lines",
            line=dict(
                width=36,
                color=axis_range,
                colorscale=[[0, "rgb(0,0,255)"], [1, "rgb(255,255,255)"]],
                cmin=0,
                cmax=255,
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=dict(text="R", font=dict(color="red", size=18)),
                tickfont=dict(color="red", size=15),
                showticklabels=True,
                tickvals=[0, 64, 128, 192, 255],
            ),
            yaxis=dict(
                title=dict(text="G", font=dict(color="green", size=18)),
                tickfont=dict(color="green", size=15),
                showticklabels=True,
                tickvals=[0, 64, 128, 192, 255],
            ),
            zaxis=dict(
                title=dict(text="B", font=dict(color="blue", size=18)),
                tickfont=dict(color="blue", size=15),
                showticklabels=True,
                tickvals=[0, 64, 128, 192, 255],
            ),
        ),
        title=f"Random RGB sample ({sample_size} points)",
    )

    fig.write_html(output_path, auto_open=False, include_plotlyjs="cdn")
    return output_path


if __name__ == '__main__':
    inputdata = 'inputphoto2.jpg'
    rgb_img = preprocess_image(inputdata)
    plot_rgb_path = plot_random_rgb_points(rgb_img, sample_size=30000)
    print(f"Saved RGB scatter plot to {plot_rgb_path}")
