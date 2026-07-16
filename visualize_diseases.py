import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

CSV_PATH = "full_df.csv"
IMAGE_DIR = "preprocessed_images"

LABEL_COLUMNS = ["N", "D", "G", "C", "A", "H", "M", "O"]
LABEL_NAMES = {
    "N": "Normal",
    "D": "Diabetic Retinopathy",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "AMD",
    "H": "Hypertension",
    "M": "Myopia",
    "O": "Other",
}


def show_one_per_label(csv_path=CSV_PATH, image_dir=IMAGE_DIR):
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for ax, label in zip(axes, LABEL_COLUMNS):
        matches = df[(df[label] == 1) & (df[LABEL_COLUMNS].sum(axis=1) == 1)]

        ax.axis("off")
        if matches.empty:
            ax.set_title(f"{LABEL_NAMES[label]} ({label})\nno examples found")
            continue

        filename = matches.iloc[0]["filename"]
        img_path = os.path.join(image_dir, filename)

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            ax.set_title(f"{LABEL_NAMES[label]} ({label})\nmissing: {filename}")
            continue

        ax.imshow(image)
        ax.set_title(f"{LABEL_NAMES[label]} ({label})")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_one_per_label()
