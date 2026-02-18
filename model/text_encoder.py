import torch
import torch.nn as nn
from transformers import DistilBertModel


class TextEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super(TextEncoder, self).__init__()

        self.backbone = DistilBertModel.from_pretrained("distilbert-base-uncased")

        self.projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
        embeddings = self.projection(cls_embedding)

        return embeddings
