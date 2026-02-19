import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import DistilBertTokenizer
import torchvision.transforms as transforms


class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, caption_file, max_length=32):
        self.image_dir = image_dir
        self.max_length = max_length
        
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        
        self.image_caption_pairs = []
        self.image_to_indices = {}
        self.unique_images = []

        with open(caption_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if len(line) == 0:
                    continue

                img_name, caption = line.split(",", 1)
                img_path = os.path.join(self.image_dir, img_name.strip())
                if not os.path.isfile(img_path):
                    continue  # Skip if image file doesn't exist
                self.image_caption_pairs.append((img_name.strip(), caption.strip()))

                if img_name not in self.image_to_indices:
                    self.image_to_indices[img_name] = []
                    self.unique_images.append(img_name)

                self.image_to_indices[img_name].append(idx)


        if len(self.image_caption_pairs) == 0:
            raise FileNotFoundError(
                f"No image files found in {image_dir}. "
                "Download Flickr8k images and place .jpg files in the image_dir folder. "
                "Expected structure: image_dir/1000268201_693b08cb0e.jpg, etc."
            )

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx):
        img_name, caption = self.image_caption_pairs[idx]

        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        encoding = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "image": image,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }
