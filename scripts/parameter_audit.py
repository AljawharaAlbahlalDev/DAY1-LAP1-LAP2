"""Lab 2: parameter accounting for mBERT and CAMeLBERT."""

from collections import defaultdict
from transformers import AutoModel

def audit(checkpoint: str) -> dict:
    
    #    هل الموديل الأكبر حجمه بسبب الاتنشن ولا بسبب الفوكابلري  والامبدنق ؟
#البرامتر هي الأرقام اللي الموديل يتعلمها أثناء التدريب
    # Load the pretrained model
    model = AutoModel.from_pretrained(checkpoint)

    # Store parameter counts by subsystem
    buckets = defaultdict(int)

    for name, parameter in model.named_parameters():
        count = parameter.numel()

        # LayerNorm parameters
        if "LayerNorm" in name or "layer_norm" in name:
            buckets["norms"] += count

        # Token + position + token-type embeddings
        elif name.startswith("embeddings."):
            buckets["embeddings"] += count

        # Query / Key / Value + attention output projection
        elif ".attention." in name:
            buckets["attention"] += count

        # Feed-forward network
        elif ".intermediate." in name or (
            ".output.dense." in name and ".attention." not in name
        ):
            buckets["ffn"] += count

        # Pooler
        elif name.startswith("pooler."):
            buckets["pooler"] += count

        # Anything not captured above
        else:
            buckets["other"] += count

    total = sum(buckets.values())

    result = {
        "total": total,
        "embeddings": buckets["embeddings"],
        "attention": buckets["attention"],
        "ffn": buckets["ffn"],
        "norms": buckets["norms"],
        "pooler": buckets["pooler"],
        "other": buckets["other"],
    }

    return result


if __name__ == "__main__":
    for ckpt in [
        "bert-base-multilingual-cased",
        "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    ]:
        result = audit(ckpt)
        total = result["total"]

        print("\n" + "=" * 60)
        print(ckpt)
        print("=" * 60)

        for name, count in result.items():
            if name == "total":
                continue

            percentage = count / total * 100

            print(
                f"{name:12s}: "
                f"{count / 1_000_000:7.2f}M "
                f"({percentage:5.1f}%)"
            )

        print(f"{'TOTAL':12s}: {total / 1_000_000:7.2f}M")
        
        
""" 


        python scripts/parameter_audit.py
        
        
        
        
============================================================
bert-base-multilingual-cased
============================================================
embeddings  :   92.21M ( 51.8%)
attention   :   28.35M ( 15.9%)
ffn         :   56.67M ( 31.9%)
norms       :    0.04M (  0.0%)
pooler      :    0.59M (  0.3%)
other       :    0.00M (  0.0%)
TOTAL       :  177.85M
pytorch_model.bin: 100%

============================================================
CAMeL-Lab/bert-base-arabic-camelbert-mix
============================================================
embeddings  :   23.43M ( 21.5%)
attention   :   28.35M ( 26.0%)
ffn         :   56.67M ( 52.0%)
norms       :    0.04M (  0.0%)
pooler      :    0.59M (  0.5%)
other       :    0.00M (  0.0%)
TOTAL       :  109.08M



mBERT:
177.85M total
92.21M embeddings = 51.8%

CAMeLBERT:
109.08M total
23.43M embeddings = 21.5%

Attention = same ~28.35M
FFN       = same ~56.67M

Key insight:
mBERT is much larger mainly because of its multilingual vocabulary.

Big vocabulary
→ big embedding table
→ multilingual tax





python notebooks/02_transformer_anatomy.py


        """
        