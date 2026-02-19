import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


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
    unique_image_embeddings = []
    seen = set()

    for idx, (img_name, _) in enumerate(dataset.image_caption_pairs):
        if img_name not in seen:
            unique_image_embeddings.append(image_embeddings[idx])
            seen.add(img_name)

    image_embeddings = torch.stack(unique_image_embeddings)

    similarity_matrix = image_embeddings @ text_embeddings.T

    recall_at_1 = compute_recall(similarity_matrix, dataset, k=1)
    recall_at_5 = compute_recall(similarity_matrix, dataset, k=5)
    recall_at_10 = compute_recall(similarity_matrix, dataset, k=10)

    print(f"Recall@1: {recall_at_1:.4f}")
    print(f"Recall@5: {recall_at_5:.4f}")
    print(f"Recall@10: {recall_at_10:.4f}")


def compute_recall(similarity_matrix, dataset, k=1):

    correct = 0
    total = len(dataset.unique_images)

    image_names = dataset.unique_images

    for i, img_name in enumerate(image_names):

        similarities = similarity_matrix[i]
        top_k = torch.topk(similarities, k=k).indices.tolist()

        correct_caption_indices = dataset.image_to_indices[img_name]

        if any(idx in top_k for idx in correct_caption_indices):
            correct += 1

    return correct / total
