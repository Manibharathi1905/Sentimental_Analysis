import torch
import torch.nn as nn
from transformers import AutoModel

class EmotionClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", n_labels=50, hidden_dim=256, dropout=0.2, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = AutoModel.from_pretrained(model_name)
        in_dim = self.backbone.config.hidden_size
        self.pooler = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(hidden_dim, n_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # use [0][:,0,:] style pooling for DistilBERT
        pooled = out.last_hidden_state[:,0,:]
        h = self.pooler(pooled)
        logits = self.classifier(h)
        return logits
