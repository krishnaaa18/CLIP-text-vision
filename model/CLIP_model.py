import torch
import torch.nn as nn
import torch.nn.functional as F
from model.image_encoder import ImageEncoder
from model.text_encoder import TextEncoder


class CLIPModel(nn.Module):
    def __init__(self, embed_dim=256):
        super(CLIPModel, self).__init__()

        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim)

        # Learnable temperature parameter
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(0.07)))

    def forward(self, images, input_ids, attention_mask):
        image_embeddings = self.image_encoder(images)
        text_embeddings = self.text_encoder(input_ids, attention_mask)

        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=1)

        # Similarity matrix
        temperature = torch.exp(self.log_temperature)
        logits = torch.matmul(image_embeddings, text_embeddings.T) / temperature

        return logits
