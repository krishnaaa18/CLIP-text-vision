import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from utils.dataset import Flickr8kDataset
from model.CLIP_model import CLIPModel
from loss import CLIPLoss
from torch.optim import AdamW
from tqdm import tqdm


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_dir = os.path.join(base_dir, "dataset", "Flickr8k image", "Flickr8k_images")
    if not os.path.isdir(image_dir):
        image_dir = os.path.join(base_dir, "dataset", "Flickr8k image")
    caption_file = os.path.join(base_dir, "dataset", "Flickr8k image", "Flickr8k_token.txt")

    dataset = Flickr8kDataset(
        image_dir=image_dir,
        caption_file=caption_file
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = CLIPModel(embed_dim=256).to(device)
    loss_fn = CLIPLoss()

    optimizer = AdamW(model.parameters(), lr=1e-4)

    epochs = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(loader):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(images, input_ids, attention_mask)

            loss = loss_fn(logits)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "clip_model.pth")
    print("Model saved as clip_model.pth")


if __name__ == "__main__":
    train()
