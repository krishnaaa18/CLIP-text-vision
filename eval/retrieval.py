import os
import sys

# Allow running imports when launched as a script from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


def evaluate(model, dataset, device):

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model.eval()

    all_image_embeddings = []
    all_text_embeddings = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            image_emb = model.image_encoder(images)
            text_emb = model.text_encoder(input_ids, attention_mask)

            image_emb = F.normalize(image_emb, dim=1)
            text_emb = F.normalize(text_emb, dim=1)

            all_image_embeddings.append(image_emb)
            all_text_embeddings.append(text_emb)

    image_embeddings = torch.cat(all_image_embeddings)
    text_embeddings = torch.cat(all_text_embeddings)

    similarity_matrix = image_embeddings @ text_embeddings.T

    recall_at_1 = compute_recall(similarity_matrix, k=1)
    recall_at_5 = compute_recall(similarity_matrix, k=5)
    recall_at_10 = compute_recall(similarity_matrix, k=10)

    print(f"Recall@1: {recall_at_1:.4f}")
    print(f"Recall@5: {recall_at_5:.4f}")
    print(f"Recall@10: {recall_at_10:.4f}")


def compute_recall(similarity_matrix, k=1):
    correct = 0
    total = similarity_matrix.size(0)

    for i in range(total):
        similarities = similarity_matrix[i]
        top_k = torch.topk(similarities, k=k).indices

        if i in top_k:
            correct += 1

    return correct / total
