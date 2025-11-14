# src/hybrid_model.py
import torch
import torch.nn as nn
from transformers import RobertaModel
from src.config import cfg

class HybridEmotionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # RoBERTa backbone
        self.roberta = RobertaModel.from_pretrained(cfg.model_name_or_path)

        # small dropout and single linear classifier (OLD architecture — compatible with checkpoints)
        self.dropout = nn.Dropout(0.1)

        # GTE-large embedding dim is 1024 (precomputed). If you used another embedding model, adjust here.
        gte_dim = 1024
        roberta_dim = self.roberta.config.hidden_size

        self.classifier = nn.Linear(roberta_dim + gte_dim, len(cfg.label_list))

    def forward(self, input_ids, attention_mask, texts, gte):
        """
        Forward expects:
         - input_ids, attention_mask : tokenized inputs for RoBERTa
         - texts : (unused here) kept for API-compatibility if required elsewhere
         - gte : precomputed GTE embeddings tensor (batch_size, 1024)
        """
        roberta_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = roberta_out.last_hidden_state[:, 0]  # CLS token

        # ensure gte on same device
        if gte.device != cls_vec.device:
            gte = gte.to(cls_vec.device)

        fused = torch.cat([cls_vec, gte], dim=1)
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return logits
