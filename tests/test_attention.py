"""Lab 2 contract checks."""
import torch
import torch.nn.functional as F
from bayan.attention import attention


def test_attention_matches_pytorch():
    torch.manual_seed(42)
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    actual = attention(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(actual, expected, atol=1e-6)

def test_causal_mask_prevents_future_attention():

    # عندنا 4 توكن فقط
    # Head واحد
    # وكل توكن لها دايمنشن واحد فقط عشان المثال يكون بسيط
    q = torch.zeros(1, 1, 4, 1)
    k = torch.zeros(1, 1, 4, 1)

    # كل توكن نعطيها فاليو واضحة
    v = torch.tensor([[[[1.0],
                        [2.0],
                        [3.0],
                        [4.0]]]])

    # Causal Mask
    # كل توكن تشوف نفسها واللي قبلها فقط
    mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))

    # نضيف batch dimension و head dimension
    # [4,4] -> [1,1,4,4]
    mask = mask.unsqueeze(0).unsqueeze(0)

    # نشغل الاتنشن حقنا
    actual = attention(q, k, v, mask)

    # النتيجة اللي نتوقعها
    expected = torch.tensor([[[[1.0],
                              [1.5],
                              [2.0],
                              [2.5]]]])

    # نقارن actual مع expected
    assert torch.allclose(actual, expected, atol=1e-6)
