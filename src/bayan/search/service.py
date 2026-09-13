"""
Lab 5.2 — Two-stage Search Service

الفكرة:
1) نحمل الـ FAISS index اللي بنيناه في Lab 5.1.
2) نحول query إلى embedding باستخدام نفس الـ encoder.
3) نعمل L2 normalization للـ query.
4) FAISS يرجع أفضل candidates بسرعة.
5) Cross-Encoder يعيد ترتيب النتائج بشكل أدق.
6) إذا أفضل نتيجة أقل من min_score نرجع [] بدل نتيجة غير موثوقة.

تشغيل الاختبار:
    pytest tests/test_search_contract.py -q

بعد بناء الـ index كامل:
    python -m bayan.search.service
"""

import json

import faiss
import numpy as np
import pandas as pd

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)


# الـ Cross-Encoder المستخدم لإعادة ترتيب النتائج.
RERANKER = (
    "cross-encoder/"
    "mmarco-mMiniLMv2-L12-H384-v1"
)


class CaseSearch:

    def __init__(
        self,
        prefix: str,
    ):
        """
        prefix مثال:

            artifacts/case_index_v1

        ومنه نقرأ:

            artifacts/case_index_v1.faiss
            artifacts/case_index_v1_meta.parquet
            artifacts/case_index_v1_manifest.json
        """

        # --------------------------------------------------------
        # 1) Load manifest
        # --------------------------------------------------------

        with open(
            f"{prefix}_manifest.json",
            encoding="utf-8",
        ) as f:

            self.manifest = json.load(
                f
            )

        # --------------------------------------------------------
        # 2) Load FAISS index
        # --------------------------------------------------------

        self.index = faiss.read_index(
            f"{prefix}.faiss"
        )

        # --------------------------------------------------------
        # 3) Load metadata
        # --------------------------------------------------------

        self.meta = pd.read_parquet(
            f"{prefix}_meta.parquet"
        )

        # --------------------------------------------------------
        # 4) Load SAME encoder used in Lab 5.1
        # --------------------------------------------------------
        #
        # ما نكتب اسم موديل جديد هنا.
        # نقرأه من manifest حتى نضمن:
        #
        # index encoder == query encoder
        #

        self.encoder = (
            SentenceTransformer(
                self.manifest[
                    "model"
                ]
            )
        )

        # --------------------------------------------------------
        # 5) Load Cross-Encoder
        # --------------------------------------------------------

        self.reranker = (
            CrossEncoder(
                RERANKER
            )
        )

        # --------------------------------------------------------
        # 6) Load canary
        # --------------------------------------------------------
        #
        # إذا manifest يقول عندنا 20,000 vector
        # والـ index فعليًا يحتوي رقم ثاني،
        # معناها الملفات ليست من نفس build/version.
        #

        assert (
            self.index.ntotal
            == self.manifest[
                "n_vectors"
            ]
        ), (
            "Index and manifest "
            "do not match."
        )

        # metadata أيضًا لازم تطابق عدد vectors.

        assert (
            len(self.meta)
            == self.index.ntotal
        ), (
            "Metadata and FAISS index "
            "do not match."
        )

        # --------------------------------------------------------
        # مهم:
        # Instructor example يستخدم summary.
        #
        # لكن dataset حقنا الحالي يستخدم:
        #
        #     case_text
        #
        # لذلك نستخدم case_text.
        # --------------------------------------------------------

        if (
            "case_text"
            in self.meta.columns
        ):

            self.text_column = (
                "case_text"
            )

        elif (
            "summary"
            in self.meta.columns
        ):

            # fallback لو تغيرت نسخة الداتا.
            self.text_column = (
                "summary"
            )

        else:

            raise KeyError(
                "Expected case_text "
                "or summary in metadata."
            )

    def search(
        self,
        query: str,
        k: int = 5,
        candidates: int = 50,
        min_score: float = 0.25,
        rerank: bool = True,
    ) -> list[dict]:
        """
        query:
            النص اللي نبحث عنه.

        k:
            عدد النتائج النهائية.

        candidates:
            عدد النتائج الأولية من FAISS.

        min_score:
            أقل Cross-Encoder score نقبله.

        rerank:
            False -> Stage 1 فقط.
            True  -> Stage 1 + Stage 2.
        """

        # Query فاضية؟
        # نرجع [] مباشرة.

        if not query.strip():
            return []

        # --------------------------------------------------------
        # STAGE 1
        # Bi-Encoder + FAISS
        # --------------------------------------------------------

        # نحول query إلى embedding.

        q = self.encoder.encode(
            [query],
            convert_to_numpy=True,
        ).astype(
            "float32"
        )

        # --------------------------------------------------------
        # L2 Normalization
        # --------------------------------------------------------
        #
        # لازم query تكون normalized
        # مثل vectors اللي دخلت الـ index.
        #
        # بهذا الشكل Inner Product
        # يصير قريب من cosine similarity.
        #




# ================================================================
# LAB 5.4 — PLANTED BUG
# ================================================================
#
# جربي مؤقتًا تعطيل السطر التالي:
#
#     faiss.normalize_L2(q)
#
# ثم شغلي:
#
#     python notebooks/05_retrieval_eval.py
#
# المفروض تلاحظين أن Recall/MRR تتأثر.
#
# السبب:
#
# بدون L2 normalization،
# Inner Product ما يعود يقيس اتجاه الـ vectors فقط،
# بل يتأثر أيضًا بحجم الـ vector.
#
# ممكن بعض النتائج تبدو منطقية يدويًا،
# لكن الـ labelled metrics تكشف أن retrieval الحقيقي أسوأ.
#
# بعد التجربة:
# رجعي السطر ولا تخلينه محذوف.
# ================================================================

        faiss.normalize_L2(
            q
        )

        # ما نطلب candidates أكثر من عدد الـ vectors.

        candidates = min(
            candidates,
            self.index.ntotal,
        )

        # Search inside FAISS.

        scores, idx = (
            self.index.search(
                q,
                candidates,
            )
        )

        # idx shape:
        #
        # [[12, 500, 90, ...]]
        #
        # يعني row numbers داخل metadata.

        row_ids = idx[0]

        # FAISS قد يرجع -1 إذا ما وجد position صالح.
        row_ids = row_ids[
            row_ids >= 0
        ]

        if (
            len(row_ids)
            == 0
        ):
            return []

        # نجيب بيانات الـ candidates.

        cand = (
            self.meta
            .iloc[
                row_ids
            ]
            .copy()
        )

        # نخزن FAISS score
        # عشان نقدر نقارنه لاحقًا.

        cand[
            "bi_score"
        ] = scores[
            0
        ][
            :len(cand)
        ]

        # --------------------------------------------------------
        # إذا rerank=False
        # نرجع نتائج Stage 1 فقط.
        # --------------------------------------------------------

        if not rerank:

            return (
                cand
                .head(k)
                .to_dict(
                    "records"
                )
            )

        # --------------------------------------------------------
        # STAGE 2
        # Cross-Encoder Re-ranking
        # --------------------------------------------------------
        #
        # Cross-Encoder ما يقارن embeddings فقط.
        #
        # يدخل:
        #
        #     (query, candidate_text)
        #
        # مع بعض داخل model.
        #
        # أبطأ، لكن يعطي ترتيب أدق.
        #

        pairs = [
            (
                query,
                str(text),
            )
            for text
            in cand[
                self.text_column
            ].tolist()
        ]

        ce_scores = (
            self.reranker.predict(
                pairs
            )
        )

        cand[
            "ce_score"
        ] = np.asarray(
            ce_scores,
            dtype=float,
        )

        # نرتب حسب Cross-Encoder score.

        cand = (
            cand
            .sort_values(
                "ce_score",
                ascending=False,
            )
            .head(k)
        )

        if (
            len(cand)
            == 0
        ):
            return []

        # --------------------------------------------------------
        # Honest empty result
        # --------------------------------------------------------
        #
        # Semantic search دائمًا تقريبًا يقدر يجيب "شيء".
        #
        # لكن إذا أفضل score ضعيف جدًا،
        # الأفضل نقول:
        #
        #     ما عندي نتيجة موثوقة
        #
        # بدل نرجع نتيجة بالقوة.
        #

        if (
            cand[
                "ce_score"
            ].max()
            < min_score
        ):
            return []

        return cand.to_dict(
            "records"
        )


if __name__ == "__main__":

    # ------------------------------------------------------------
    # هذا الـ smoke test يحتاج Lab 5.1 index يكون مبني.
    # ------------------------------------------------------------

    search = CaseSearch(
        "artifacts/case_index_v1"
    )

    results = search.search(
        "street light broken for two weeks",
        k=3,
        candidates=50,
        min_score=0.25,
    )

    print(
        f"Returned results: "
        f"{len(results)}"
    )

    for result in results:

        print(
            result[
                "case_id"
            ],
            result[
                "ce_score"
            ],
            result[
                "case_text"
            ][:80],
        )


# ================================================================
# EXPECTED OUTPUT SHAPE
# ================================================================
#
# Returned results: 3
# <case-id> <score> <case text ...>
# <case-id> <score> <case text ...>
# <case-id> <score> <case text ...>
#
# Exact IDs / scores will depend on your actual index.
#
# Course idea:
# An English query may retrieve an Arabic historical case
# because the multilingual bi-encoder places both languages
# in the same embedding space.