"""Lab 2: inspect scaled dot-product attention and multi-head attention."""

import math

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from bayan.attention import attention, MultiHeadAttention


def main():
    # ---------------------------------------------------------
    # 1) Create a tiny toy example: 4 tokens, 1 attention head
    # ---------------------------------------------------------
    torch.manual_seed(0)

    batch = 1
    heads = 1
    seq_len = 4
    d_k = 8
    #" أعطي Attention vectors عشوائية وأختبر هل الحساب الرياضي اللي كتبته مطابق لـPyTorch."
    q = torch.randn(batch, heads, seq_len, d_k)
    k = torch.randn(batch, heads, seq_len, d_k)
    v = torch.randn(batch, heads, seq_len, d_k)

    # ---------------------------------------------------------
    # 2) Our attention vs PyTorch attention
    # ---------------------------------------------------------
    actual = attention(q, k, v)

    expected = F.scaled_dot_product_attention(
        q,
        k,
        v,
    )

    assert torch.allclose(actual, expected, atol=1e-6)

    print("Attention equivalence: PASSED (atol=1e-6)")

    # ---------------------------------------------------------
    # 3) Calculate the attention weights for inspection
    # ---------------------------------------------------------
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)

    weights = torch.softmax(scores, dim=-1)

    print("\nAttention weights:")
    print(weights[0, 0].round(decimals=2))

    # Before training these weights are based on random vectors,
    # so we should NOT interpret them as meaningful language patterns.

    # ---------------------------------------------------------
    # 4) Multi-Head Attention
    # ---------------------------------------------------------
    mha = MultiHeadAttention(
        d_model=768,
        n_heads=12,
    )

    n_params = sum(
        p.numel()
        for p in mha.parameters()
    )

    print(f"\nMHA parameters: {n_params:,}")

    # Small input just to verify the output shape
    x = torch.randn(1, 4, 768)

    output = mha(x)

    print("MHA input shape: ", x.shape)
    print("MHA output shape:", output.shape)


    # ---------------------------------------------------------
    # 5) Causal mask
    # ---------------------------------------------------------

    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len)
    ).bool()

    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    print("\nCausal mask:")
    print(causal_mask[0, 0].int())

    # Recalculate scores
    causal_scores = q @ k.transpose(-2, -1)
    causal_scores = causal_scores / math.sqrt(d_k)

    # Block future tokens BEFORE softmax
    causal_scores = causal_scores.masked_fill(
        causal_mask == 0,
        float("-inf")
    )

    causal_weights = torch.softmax(
        causal_scores,
        dim=-1
    )

    print("\nCausal attention weights:")
    print(causal_weights[0, 0].round(decimals=2))

    # Verify that no attention goes to future positions
    assert torch.all(
        causal_weights[0, 0].triu(diagonal=1) == 0
    )

    print("\nCausal mask check: PASSED")
    print("Architecture family: Decoder")


    # ---------------------------------------------------------
    # 6) Attention on real Arabic text
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("STEP 6 — REAL ARABIC ATTENTION")
    print("=" * 60)

    checkpoint = "CAMeL-Lab/bert-base-arabic-camelbert-mix"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    model = AutoModel.from_pretrained(
        checkpoint,
        output_attentions=True,
    ).eval()

    sentences = [
        "انقطعت الكهرباء في حي النرجس منذ ثلاث ساعات",
        "الخدمة لا تعمل منذ الصباح",
        "تم حل المشكلة ولكن التطبيق ما زال بطيئًا",
    ]

    for text in sentences:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=24,
            truncation=True,
        )

        with torch.inference_mode():
            outputs = model(**encoded)

        attentions = outputs.attentions

        tokens = tokenizer.convert_ids_to_tokens(
            encoded["input_ids"][0]
        )

        real_length = int(
            encoded["attention_mask"][0].sum().item()
        )

        print("\nTEXT:")
        print(text)

        print("\nTOKENS:")
        print(tokens[:real_length])

        print(
            "Attention tensor shape:",
            attentions[0].shape
        )

        # -----------------------------------------------------
        # Find a candidate head that attends to nearby tokens
        # -----------------------------------------------------

        best_adj_score = -1.0
        best_adj_layer = None
        best_adj_head = None

        for layer_index, layer_attention in enumerate(attentions):

            for head_index in range(layer_attention.shape[1]):

                weights = layer_attention[0, head_index]

                adjacency_scores = []

                # Ignore [CLS] and [SEP]
                for i in range(1, real_length - 1):

                    neighbours = []

                    if i - 1 >= 1:
                        neighbours.append(weights[i, i - 1])

                    if i + 1 < real_length - 1:
                        neighbours.append(weights[i, i + 1])

                    if neighbours:
                        adjacency_scores.append(
                            torch.stack(neighbours).sum()
                        )

                if adjacency_scores:
                    score = (
                        torch.stack(adjacency_scores)
                        .mean()
                        .item()
                    )

                    if score > best_adj_score:
                        best_adj_score = score
                        best_adj_layer = layer_index
                        best_adj_head = head_index

        print(
            "\nCandidate adjacency head:"
            f" layer={best_adj_layer},"
            f" head={best_adj_head},"
            f" score={best_adj_score:.4f}"
        )

        # -----------------------------------------------------
        # Find a candidate head that attends strongly to [SEP]
        # -----------------------------------------------------

        sep_index = tokens.index("[SEP]")

        best_sep_score = -1.0
        best_sep_layer = None
        best_sep_head = None

        for layer_index, layer_attention in enumerate(attentions):

            for head_index in range(layer_attention.shape[1]):

                weights = layer_attention[0, head_index]

                sep_score = (
                    weights[:real_length, sep_index]
                    .mean()
                    .item()
                )

                if sep_score > best_sep_score:
                    best_sep_score = sep_score
                    best_sep_layer = layer_index
                    best_sep_head = head_index

        print(
            "Candidate [SEP]-sink head:"
            f" layer={best_sep_layer},"
            f" head={best_sep_head},"
            f" score={best_sep_score:.4f}"
        )


    # ---------------------------------------------------------
    # 7) PAD attention leakage
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("STEP 7 — PAD LEAK")
    print("=" * 60)

    text = "انقطعت الكهرباء في حي النرجس منذ ثلاث ساعات"

    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        max_length=24,
        truncation=True,
    )

    tokens = tokenizer.convert_ids_to_tokens(
        encoded["input_ids"][0]
    )

    pad_positions = [
        i
        for i, token in enumerate(tokens)
        if token == "[PAD]"
    ]

    real_query_positions = (
        encoded["attention_mask"][0].bool()
    )

    # ---------------------------------------------------------
    # WITH attention mask
    # ---------------------------------------------------------

    with torch.inference_mode():
        outputs_with_mask = model(**encoded)

    # Use a deeper layer/head for the diagnostic
    layer = 8
    head = 3

    weights_with_mask = (
        outputs_with_mask.attentions[layer][0, head]
    )

    pad_mass_with_mask = (
        weights_with_mask[real_query_positions]
        [:, pad_positions]
        .sum(dim=-1)
        .mean()
        .item()
    )

    # ---------------------------------------------------------
    # WITHOUT attention mask
    # ---------------------------------------------------------

    encoded_without_mask = {
        key: value
        for key, value in encoded.items()
        if key != "attention_mask"
    }

    with torch.inference_mode():
        outputs_without_mask = model(
            **encoded_without_mask
        )

    weights_without_mask = (
        outputs_without_mask.attentions[layer][0, head]
    )

    pad_mass_without_mask = (
        weights_without_mask[real_query_positions]
        [:, pad_positions]
        .sum(dim=-1)
        .mean()
        .item()
    )

    print(
        f"PAD attention mass WITH mask:    "
        f"{pad_mass_with_mask:.6f}"
    )

    print(
        f"PAD attention mass WITHOUT mask: "
        f"{pad_mass_without_mask:.6f}"
    )

    assert pad_mass_with_mask < 0.01

    print("PAD leak regression check: PASSED")
if __name__ == "__main__":
    main()
    """
    1) Attention vs PyTorch
    2) Attention weights 4×4
    3) Multi-Head Attention
    4) MHA parameter count
    5) Causal mask
    6) Real Arabic attention
    7) PAD leak
    """
    
    
    """
python notebooks/02_transformer_anatomy.py
Attention equivalence: PASSED (atol=1e-6)

Attention weights:
tensor([[0.1200, 0.7400, 0.1100, 0.0300],
        [0.2600, 0.2700, 0.1700, 0.3100],
        [0.3400, 0.0200, 0.1900, 0.4600],
        [0.2000, 0.6200, 0.1000, 0.0700]])

MHA parameters: 2,362,368
MHA input shape:  torch.Size([1, 4, 768])
MHA output shape: torch.Size([1, 4, 768])


Our attention
      ≈
PyTorch attention
atol = 1e-6

4 tokens
→ attention matrix = 4 × 4

Each row:
one token's attention distribution
over all tokens

Rows sum to 1.

Random Q/K/V:
weights are NOT linguistically meaningful yet.

MHA:
768 = 12 heads × 64
input  [B, seq, 768]
output [B, seq, 768]

Expected MHA params:
2,362,368




python notebooks/02_transformer_anatomy.py

Attention equivalence: PASSED (atol=1e-6)

Attention weights:
tensor([[0.1200, 0.7400, 0.1100, 0.0300],
        [0.2600, 0.2700, 0.1700, 0.3100],
        [0.3400, 0.0200, 0.1900, 0.4600],
        [0.2000, 0.6200, 0.1000, 0.0700]])

MHA parameters: 2,362,368
MHA input shape:  torch.Size([1, 4, 768])
MHA output shape: torch.Size([1, 4, 768])

Causal mask:
tensor([[1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]], dtype=torch.int32)

Causal attention weights:
tensor([[1.0000, 0.0000, 0.0000, 0.0000],
        [0.4900, 0.5100, 0.0000, 0.0000],
        [0.6200, 0.0300, 0.3500, 0.0000],
        [0.2000, 0.6200, 0.1000, 0.0700]])

Causal mask check: PASSED
Architecture family: Decoder



Lab 2 — Attention

✅ Our Attention matches PyTorch
✅ 4 tokens → 4×4 attention matrix
✅ Each row = where one token distributes its attention
✅ Row weights sum ≈ 1
⚠ Random Q/K/V → don't interpret linguistically

MHA:
768 = 12 heads × 64

Input:  [1, 4, 768]
Output: [1, 4, 768]

MHA params = 2,362,368
    """
    