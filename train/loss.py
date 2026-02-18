import torch
import torch.nn as nn


class CLIPLoss(nn.Module):
    def __init__(self):
        super(CLIPLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits):
        """
        logits: (B, B) similarity matrix
        """

        batch_size = logits.size(0)
        labels = torch.arange(batch_size).to(logits.device)

        # Image-to-text loss
        loss_i2t = self.cross_entropy(logits, labels)

        # Text-to-image loss
        loss_t2i = self.cross_entropy(logits.T, labels)

        loss = (loss_i2t + loss_t2i) / 2

        return loss
