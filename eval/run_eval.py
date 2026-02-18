import os
import sys

# Allow running as a script: `python eval/run_eval.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils.dataset import Flickr8kDataset
from model.CLIP_model import CLIPModel
from eval.retrieval import evaluate


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_dir = os.path.join(base_dir, "dataset", "Flickr8k image", "Flickr8k_images")
    if not os.path.isdir(image_dir):
        image_dir = os.path.join(base_dir, "dataset", "Flickr8k image")
    caption_file = os.path.join(base_dir, "dataset", "Flickr8k image", "Flickr8k_token.txt")

    dataset = Flickr8kDataset(image_dir=image_dir, caption_file=caption_file)

    model = CLIPModel(embed_dim=256)
    model_path = os.path.join(base_dir, "clip_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    evaluate(model, dataset, device)


if __name__ == "__main__":
    main()
