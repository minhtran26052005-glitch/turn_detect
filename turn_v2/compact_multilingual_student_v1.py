"""Six-layer multilingual student with an explicit <20M parameter budget."""
import torch
from torch import nn

LIMIT = 20_000_000
CONFIG = dict(vocab_size=119547, dim=384, hidden_dim=1536, n_layers=6,
              n_heads=6, max_position_embeddings=512, num_labels=2,
              embedding_rank=64, pad_token_id=0)


class FactorizedEmbedding(nn.Module):
    def __init__(self, vocabulary, rank, width, padding):
        super().__init__()
        self.table = nn.Embedding(vocabulary, rank, padding_idx=padding)
        self.projection = nn.Linear(rank, width)

    def forward(self, ids):
        return self.projection(self.table(ids))


class CompactTurnTransformer(nn.Module):
    def __init__(self, **config):
        super().__init__()
        from transformers import DistilBertConfig, AutoModelForSequenceClassification
        values = {**CONFIG, **config}; rank = values.pop('embedding_rank')
        cfg = DistilBertConfig(**values)
        cfg._attn_implementation = 'eager'
        self.base = AutoModelForSequenceClassification.from_config(cfg, dtype=torch.float32)
        self.base.distilbert.embeddings.word_embeddings = FactorizedEmbedding(
            cfg.vocab_size, rank, cfg.dim, cfg.pad_token_id)
        self.teacher_projection = nn.Linear(cfg.dim, 768)
        self.config = {**values, 'embedding_rank': rank}
        assert self.parameter_counts()['training_total'] < LIMIT

    def parameter_counts(self):
        return {'deployment': sum(p.numel() for p in self.base.parameters()),
                'training_total': sum(p.numel() for p in self.parameters()),
                'limit_exclusive': LIMIT,
                'embedding_included': True, 'classifier_included': True}

    def forward(self, ids, mask, return_features=False):
        result = self.base(input_ids=ids, attention_mask=mask,
                           output_hidden_states=return_features)
        if return_features:
            return result.logits, self.teacher_projection(result.hidden_states[-1][:, 0])
        return result.logits

    def initialize_from_teacher(self, state):
        """Structured width slicing plus rank64 SVD; not a parity claim."""
        ix = torch.cat([torch.arange(h*64, (h+1)*64) for h in (0, 2, 4, 6, 8, 10)])
        ff = torch.arange(0, 3072, 2)
        target = self.base.state_dict()
        for key in list(target):
            if 'word_embeddings.' in key:
                continue
            source = state[key]
            if '.ffn.lin1.weight' in key:
                value = source[ff][:, ix]
            elif '.ffn.lin1.bias' in key:
                value = source[ff]
            elif '.ffn.lin2.weight' in key:
                value = source[ix][:, ff]
            elif key == 'classifier.weight':
                value = source[:, ix]
            elif key == 'classifier.bias':
                value = source
            elif key.endswith('position_embeddings.weight'):
                value = source[:, ix]
            elif source.ndim == 2:
                value = source[ix][:, ix]
            else:
                value = source[ix]
            assert value.shape == target[key].shape, (key, value.shape, target[key].shape)
            target[key] = value.clone()
        embedding = state['distilbert.embeddings.word_embeddings.weight'][:, ix].float()
        mean = embedding.mean(0); centered = embedding-mean
        torch.manual_seed(43)
        u, singular, v = torch.svd_lowrank(centered, q=72, niter=2)
        rank = self.config['embedding_rank']
        target['distilbert.embeddings.word_embeddings.table.weight'] = u[:, :rank]*singular[:rank]
        target['distilbert.embeddings.word_embeddings.projection.weight'] = v[:, :rank]
        target['distilbert.embeddings.word_embeddings.projection.bias'] = mean
        self.base.load_state_dict(target, strict=True)
        return {'method': 'alternate attention-head/FFN slicing and centered rank64 embedding SVD',
                'teacher_tokenizer_preserved': True, 'vocabulary_pruned': False,
                'exact_teacher_function_preserved': False}


class Exportable(nn.Module):
    """Eager attention with the same trained parameters, without KD-only head."""
    def __init__(self, source):
        super().__init__()
        self.embeddings = source.base.distilbert.embeddings
        self.layers = source.base.distilbert.transformer.layer
        self.pre_classifier = source.base.pre_classifier
        self.classifier = source.base.classifier
        self.heads = source.config['n_heads']

    def forward(self, input_ids, attention_mask):
        hidden = self.embeddings(input_ids)
        batch, length, width = hidden.shape
        for layer in self.layers:
            def heads(value):
                return value.reshape(batch, length, self.heads, width//self.heads).transpose(1, 2)
            q = heads(layer.attention.q_lin(hidden))/((width//self.heads)**.5)
            k = heads(layer.attention.k_lin(hidden)); v = heads(layer.attention.v_lin(hidden))
            scores = torch.matmul(q, k.transpose(-2, -1))
            scores = scores.masked_fill(attention_mask[:, None, None, :] == 0,
                                        torch.finfo(scores.dtype).min)
            a = torch.matmul(scores.softmax(-1), v).transpose(1, 2).reshape(batch, length, width)
            hidden = layer.sa_layer_norm(hidden+layer.attention.out_lin(a))
            hidden = layer.output_layer_norm(hidden+layer.ffn.lin2(
                torch.nn.functional.gelu(layer.ffn.lin1(hidden))))
        return self.classifier(torch.relu(self.pre_classifier(hidden[:, 0])))


def export_key_to_train(key):
    if key.startswith('embeddings.'):
        return 'base.distilbert.'+key
    if key.startswith('layers.'):
        return 'base.distilbert.transformer.layer.'+key[len('layers.'):]
    return 'base.'+key
