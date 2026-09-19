import torch.nn as nn
from transformers import AutoModel


class TransformerWithClassifier(nn.Module):
    """Attach a classification head to a Hugging Face vision backbone."""

    def __init__(self, pretrained_name: str, num_classes: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained_name)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, x):
        outputs = self.backbone(x)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        return self.classifier(pooled)
