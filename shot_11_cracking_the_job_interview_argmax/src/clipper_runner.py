import pandas as pd
from smart_open import open as sopen
import torch
import clip
from PIL import Image
import numpy as np
import json

from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = preprocess(Image.open(sopen(image_path, "rb"))).to(device)
        return image

    def __len__(self):
        return len(self.image_paths)


def encode_images(image_paths, batch_size=8):

    img_emb_ls = []

    image_dataset = ImageDataset(image_paths)
    data_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for img_batch in tqdm(data_loader):
            img_emb_ls.append(model.encode_image(img_batch))

    return img_emb_ls

if __name__  == '__main__':
    df = pd.read_parquet("../data/product_images.parquet")

    img_emb_ls = encode_images(df.iloc[:20]['primary_image'])

    np.save("../data/sample_emb.npy", np.vstack(img_emb_ls))
    with open("../data/sample_ids.json", "w") as f:
        json.dump(df.iloc[:20]['asin'].tolist(), f)

