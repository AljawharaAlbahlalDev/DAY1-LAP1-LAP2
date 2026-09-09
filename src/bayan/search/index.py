"""Lab 5: versioned FAISS index build."""

import json
from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# الـBi-Encoder المستخدم لتحويل النصوص إلى Embeddings
#
# Multilingual:
# يعني يدعم العربي والإنجليزي في نفس embedding space.
# ---------------------------------------------------------
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ---------------------------------------------------------
# نسخة الـpreprocessing.
#
# نحفظها في الـmanifest حتى نعرف بأي preprocessing
# انبنى هذا الـindex.
# ---------------------------------------------------------
PREPROC_VERSION = "1.0.0"


def build_index(
    prefix: str,
    limit: int | None = None,
    cases_csv: str = "data/search/bayan_cases.csv",
) -> None:

    # -----------------------------------------------------
    # 1) قراءة الحالات التاريخية
    # -----------------------------------------------------
    df = pd.read_csv(cases_csv)

    # يستخدمها الـtest عشان ما نضطر نبني 20 ألف vector
    # أثناء الاختبار.
    #
    # مثال:
    # limit=20
    # → نستخدم أول 20 case فقط.
    if limit is not None:
        df = df.head(limit).copy()


    # -----------------------------------------------------
    # 2) النص اللي بنسوي له Embedding
    #
    # في بيانات Bayan الـsummary هي وصف الحالة.
    # -----------------------------------------------------
    texts = df["summary"].fillna("").astype(str).tolist()


    # -----------------------------------------------------
    # 3) تحميل الـBi-Encoder
    # -----------------------------------------------------
    model = SentenceTransformer(MODEL_NAME)


    # -----------------------------------------------------
    # 4) تحويل النصوص إلى Embedding vectors
    #
    # كل case:
    #
    # "عمود الإنارة معطل"
    #
    # ↓
    #
    # [0.12, -0.03, 0.87, ...]
    # -----------------------------------------------------
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


    # -----------------------------------------------------
    # 5) L2 Normalization
    #
    # نخلي طول كل vector = 1
    #
    # وهذا يسمح لنا نستخدم Inner Product
    # كـcosine similarity.
    # -----------------------------------------------------
    faiss.normalize_L2(embeddings)


    # -----------------------------------------------------
    # 6) إنشاء FAISS index
    #
    # embeddings.shape[1]
    # = عدد الأبعاد لكل vector
    #
    # مثال المرجع المتوقع تقريباً dim=384.
    # -----------------------------------------------------
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)


    # -----------------------------------------------------
    # 7) إضافة كل الـvectors إلى FAISS
    # -----------------------------------------------------
    index.add(embeddings)


    # -----------------------------------------------------
    # تأكد إن الفولدر موجود قبل الحفظ
    # -----------------------------------------------------
    prefix_path = Path(prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)


    # -----------------------------------------------------
    # 8) حفظ FAISS index
    #
    # مثال:
    # artifacts/case_index_v1.faiss
    # -----------------------------------------------------
    faiss.write_index(
        index,
        f"{prefix}.faiss",
    )


    # -----------------------------------------------------
    # 9) حفظ Metadata
    #
    # لما FAISS يرجع لنا vector رقم 10،
    # نحتاج نعرف هذا vector يرجع لأي case.
    # -----------------------------------------------------
    metadata_columns = [
        column
        for column in ["case_id", "summary", "resolution", "lang"]
        if column in df.columns
    ]

    df[metadata_columns].to_parquet(
        f"{prefix}_meta.parquet",
        index=False,
    )


    # -----------------------------------------------------
    # 10) Manifest
    #
    # هذا مثل بطاقة تعريف للـindex:
    #
    # بأي model انبنى؟
    # بأي preprocessing؟
    # كم vector؟
    # كم dimension؟
    # -----------------------------------------------------
    manifest = {
        "model": MODEL_NAME,
        "preproc_version": PREPROC_VERSION,
        "n_vectors": int(index.ntotal),
        "dim": int(dim),
    }


    # -----------------------------------------------------
    # 11) حفظ الـmanifest
    # -----------------------------------------------------
    with open(
        f"{prefix}_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )
        
        
        
    """
    Case 1:
"عمود الإنارة معطل"

Case 2:
"فاتورة المياه مرتفعة"

Case 3:
"مشكلة في إصدار الرخصة"
    
    bi encoder will do 
    Case 1 → vector
    Case 2 → vector
    Case 3 → vector
    
    ونحطهم جوا FAISS
    
    عشان لما احد يدخل حالة 
    
    FAISS يقدر بسرعة يقول:

أقرب vector هو Case 1.
    
    """
    
    