"""Lab 5: versioned FAISS index build."""

import json
from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Multilingual bi-encoder
# يحول النصوص العربية والإنجليزية إلى embeddings
# في نفس vector space
# ---------------------------------------------------------
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ---------------------------------------------------------
# نسخة الـpreprocessing المستخدمة عند بناء الـindex
# نحفظها داخل الـmanifest
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

    # أثناء الـtest نستخدم عدد قليل فقط
    # مثال: limit=20
    if limit is not None:
        df = df.head(limit).copy()


    # -----------------------------------------------------
    # 2) النص الفعلي الموجود في dataset عندنا
    #
    # الـdataset ما فيه summary
    # فيه case_text
    # -----------------------------------------------------
    texts = (
        df["case_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )


    # -----------------------------------------------------
    # 3) تحميل الـbi-encoder
    # -----------------------------------------------------
    model = SentenceTransformer(MODEL_NAME)


    # -----------------------------------------------------
    # 4) تحويل كل case إلى embedding vector
    #
    # مثال:
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
    # 5) L2 normalization
    #
    # نخلي طول كل vector = 1
    #
    # وبعدها نقدر نستخدم Inner Product
    # وكأنه cosine similarity
    # -----------------------------------------------------
    faiss.normalize_L2(embeddings)


    # -----------------------------------------------------
    # 6) إنشاء FAISS index
    # -----------------------------------------------------
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)


    # -----------------------------------------------------
    # 7) إضافة الـembeddings إلى الـindex
    # -----------------------------------------------------
    index.add(embeddings)


    # -----------------------------------------------------
    # 8) إنشاء الفولدر إذا ما كان موجود
    # -----------------------------------------------------
    prefix_path = Path(prefix)

    prefix_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    # -----------------------------------------------------
    # 9) حفظ FAISS index
    #
    # مثال:
    # case_index_v1.faiss
    # -----------------------------------------------------
    faiss.write_index(
        index,
        f"{prefix}.faiss",
    )


    # -----------------------------------------------------
    # 10) حفظ metadata
    #
    # FAISS يرجع لنا index position فقط
    # فالـmetadata تربطه بالـcase الحقيقي
    # -----------------------------------------------------
    metadata_columns = [
        column
        for column in [
            "case_id",
            "case_text",
            "resolution",
            "lang",
        ]
        if column in df.columns
    ]

    df[metadata_columns].to_parquet(
        f"{prefix}_meta.parquet",
        index=False,
    )


    # -----------------------------------------------------
    # 11) Manifest
    #
    # الـtest يتأكد تحديدًا من وجود:
    #
    # model
    # preproc_version
    # n_vectors
    # dim
    # -----------------------------------------------------
    manifest = {
        "model": MODEL_NAME,
        "preproc_version": PREPROC_VERSION,
        "n_vectors": int(index.ntotal),
        "dim": int(dim),
    }


    # -----------------------------------------------------
    # 12) حفظ الـmanifest
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
    
    