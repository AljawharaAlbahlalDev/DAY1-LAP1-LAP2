"""Lab 2 starter: scaled dot-product attention and multi-head attention."""

import math

import torch
import torch.nn as nn


def attention(q, k, v, mask=None):
    # 1) Compare Query with every Key Q × Kᵀ
    scores = q @ k.transpose(-2, -1)

    # 2) Scale scores
    scores = scores / math.sqrt(q.size(-1))

    # 3) Apply mask BEFORE softmax
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # 4) Convert scores to probabilities
    weights = torch.softmax(scores, dim=-1)

    # 5) Weighted combination of Values
    output = weights @ v

    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 768, n_heads: int = 12):
        super().__init__()
        
        # d_model = حجم vector كل Token
        # كل توكن داخل  بيدت بيس يكون ممثل فيكتور طوله 768 رقم.
        #
        # n_heads = عدد الـAttention Heads
        # في BERT Base عندنا 12 Heads

        # لازم 768 تنقسم على 12 بدون باقي
        # لأننا بنقسم الـvector بالتساوي على الـHeads        assert d_model % n_heads == 0
        
        # نخزن عدد الـHeads علاقة وحدة
        self.h = n_heads
        
        # حجم فيكتور داخل كل هيد
        #حجم ربرزنتيشن لكل توكن داخل هيد الواحد.
        # 768 / 12 = 64
        self.d_k = d_model // n_heads

        # Learned Linear Layers
        #
        # نأخذ نفس input vector x
        # ونحوله إلى 3 representations مختلفة:
        #
        # x -> Wq -> Q
        # x -> Wk -> K
        # x -> Wv -> V
        #
        # هذه الـweights يتعلمها الموديل أثناء التدريب

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        # بعد ما تخلص كل الـHeads
        # ونجمع نتائجها مرة ثانية
        # نستخدم Linear Layer أخيرة
        # عشان تمزج معلومات الـHeads مع بعض
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        b, n, _ = x.shape

        def split(t):            
            return (
                t.view(b, n, self.h, self.d_k)
                .transpose(1, 2)
            )
        # أول شيء نصنع كي وكيو فاليو من الفكتور
        # ثم نقسمه على الـ12 Heads
        q = split(self.wq(x))
        k = split(self.wk(x))
        v = split(self.wv(x))

        out = attention(q, k, v, mask)

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(b, n, -1)
        )

        out = self.wo(out)

        return out
    
"""

    pytest tests/test_attention.py -q
    2 passed in 2.84s



then check pyTorch in 
pytest tests/test_attention.py -q

Lab 2
│
├─ هل Attention حقي صح؟
│   → Equivalence test
│
├─ وين حجم الموديل؟
│   → Parameter audit
│
├─ الموديل وين يركز؟
│   → Attention map
│
└─ هل يركز على PAD بالغلط؟
    → Pad leakage test
    

Q = what am I looking for?
K = what do I contain / how can others match me?
V = what information do I provide?

Attention:
1. scores = Q @ K^T
2. scale by sqrt(d_k)
3. apply mask BEFORE softmax
4. softmax -> attention weights
5. weights @ V -> output

Shapes:
Q,K,V = [batch, heads, seq, d_k]
scores = [batch, heads, seq, seq]

BERT base:
d_model = 768
heads = 12
d_k = 64

Multi-head:
split 768 into 12 heads * 64
run attention independently
concat heads back to 768
apply output projection
    """
    