"""A genuinely pretrained mALBERT backbone and a new binary turn head."""
from pathlib import Path
import torch
from torch import nn

PARAMETER_LIMIT = 20_000_000


class MalbertTurnStudent(nn.Module):
    def __init__(self, config, sep_token_id=3):
        super().__init__()
        from transformers import AlbertModel
        config._attn_implementation = 'eager'
        self.backbone = AlbertModel(config, add_pooling_layer=True)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)
        nn.init.normal_(self.classifier.weight, std=config.initializer_range)
        nn.init.zeros_(self.classifier.bias)
        self.sep_token_id = sep_token_id
        self.config = config
        assert self.parameter_count() < PARAMETER_LIMIT

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids, attention_mask):
        # The context SEP belongs to segment0; future words and final SEP to1.
        separators = input_ids.eq(self.sep_token_id).long()
        seen_before = separators.cumsum(1)-separators
        types = ((seen_before > 0) & attention_mask.bool()).long()
        result = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                               token_type_ids=types)
        return self.classifier(self.dropout(result.pooler_output))

    @classmethod
    def from_verified_pretrained(cls, path, sep_token_id=3):
        from transformers import AlbertConfig
        from safetensors.torch import load_file
        path = Path(path)
        config = AlbertConfig.from_pretrained(path, local_files_only=True, token=False)
        model = cls(config, sep_token_id=sep_token_id)
        original = load_file(str(path/'model.safetensors'))
        backbone = {k[len('albert.'):]: v.float() for k, v in original.items()
                    if k.startswith('albert.')}
        assert set(backbone) == set(model.backbone.state_dict()), 'No missing pretrained backbone tensor permitted'
        model.backbone.load_state_dict(backbone, strict=True)
        assert all(torch.equal(v.cpu(), backbone[k]) for k, v in model.backbone.state_dict().items())
        return model, {'all_backbone_tensors_loaded_exactly': True,
            'pretrained_backbone_tensors': len(backbone),
            'newly_initialized_tensors': ['classifier.weight', 'classifier.bias'],
            'discarded_pretraining_heads': sorted(k for k in original if not k.startswith('albert.')),
            'deployment_parameters': model.parameter_count(), 'parameter_limit_exclusive': PARAMETER_LIMIT,
            'pooler_pretrained': True, 'sliced_or_factorized_backbone': False}
