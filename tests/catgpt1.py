import torch
import torch.nn as nn
import torch.nn.functional as F
from cattorch.transpiler import transpile

CHARS    = ' abcdefghijklmnopqrstuvwxyz'
HUMAN_TOKEN = len(CHARS)      # 27
BOT_TOKEN   = len(CHARS) + 1  # 28
VOCAB_SIZE  = len(CHARS) + 2  # 29
TRAIN = False

EMBED_DIM   = 32
NUM_HEADS   = 2       # head_dim = 16
NUM_LAYERS  = 1
CONTEXT_LEN = 32

MLP_MULT    = 2
DROPOUT     = 0.0     # no dropout for tiny model — every param matters

BATCH_SIZE    = 128
LEARNING_RATE = 3e-4
MAX_ITERS     = 20000
EVAL_INTERVAL = 2000
EVAL_ITERS    = 50
GRAD_CLIP     = 1.0

CHECKPOINT_PATH = 'model_tiny.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─── Model ───────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert EMBED_DIM % NUM_HEADS == 0
        self.head_dim = EMBED_DIM // NUM_HEADS

        self.qkv_proj = nn.Linear(EMBED_DIM, 3 * EMBED_DIM, bias=False)
        self.out_proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.dropout  = nn.Dropout(DROPOUT)

        mask = torch.triu(torch.ones(CONTEXT_LEN, CONTEXT_LEN), diagonal=1)
        self.register_buffer('mask', mask.bool())

    def forward(self, x):
        B, T, E = x.shape
        q, k, v = self.qkv_proj(x).split(EMBED_DIM, dim=2)

        def reshape(t):
            return t.view(B, T, NUM_HEADS, self.head_dim).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)

        scale  = self.head_dim ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(self.mask[:T, :T], float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, E)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, MLP_MULT * EMBED_DIM, bias=False),
            nn.ReLU(),
            nn.Linear(MLP_MULT * EMBED_DIM, EMBED_DIM, bias=False),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        self.attn  = CausalSelfAttention()
        self.norm2 = nn.LayerNorm(EMBED_DIM)
        self.mlp   = MLP()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb  = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_emb  = nn.Embedding(CONTEXT_LEN, EMBED_DIM)
        self.drop     = nn.Dropout(DROPOUT)
        self.blocks   = nn.Sequential(*[TransformerBlock() for _ in range(NUM_LAYERS)])
        self.norm     = nn.LayerNorm(EMBED_DIM)
        self.out_head = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False)
        self.out_head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        positions = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions))
        x = self.blocks(x)
        x = self.norm(x)
        logits = self.out_head(x)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss

    def count_params(self):
        params = sum(p.numel() for p in self.parameters())
        params -= self.out_head.weight.numel()
        return params

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -CONTEXT_LEN:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)

        return idx

model = NanoGPT()
# CATGPT cannot be transpiled (yet)
transpile(model, torch.Size([1, 32]))