"""Lab 5: two-stage bilingual case search."""

import json

import faiss
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer


# ---------------------------------------------------------
# Cross-Encoder المستخدم في المرحلة الثانية
# يعيد ترتيب النتائج اللي جابها FAISS
# ---------------------------------------------------------
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class CaseSearch:

    def __init__(self, prefix: str):

        # -------------------------------------------------
        # 1) تحميل الـManifest
        #
        # فيه:
        # model
        # preproc_version
        # n_vectors
        # dim
        # -------------------------------------------------
        with open(
            f"{prefix}_manifest.json",
            "r",
            encoding="utf-8",
        ) as f:
            self.manifest = json.load(f)


        # -------------------------------------------------
        # 2) التأكد إن الـmanifest يحتوي الأشياء الأساسية
        # -------------------------------------------------
        required_keys = [
            "model",
            "preproc_version",
            "n_vectors",
            "dim",
        ]

        for key in required_keys:
            if key not in self.manifest:
                raise ValueError(
                    f"Manifest is missing required key: {key}"
                )


        # -------------------------------------------------
        # 3) تحميل FAISS index
        #
        # هذا هو الـindex اللي بنيناه في Part 1
        # -------------------------------------------------
        self.index = faiss.read_index(
            f"{prefix}.faiss"
        )


        # -------------------------------------------------
        # 4) تحميل Metadata
        #
        # تربط رقم الـvector بالحالة الأصلية
        # -------------------------------------------------
        self.meta = pd.read_parquet(
            f"{prefix}_meta.parquet"
        )


        # -------------------------------------------------
        # 5) تحميل نفس الـBi-Encoder
        # المستخدم في بناء الـindex
        #
        # مهم جدًا يكون نفس الموديل.
        # -------------------------------------------------
        self.encoder = SentenceTransformer(
            self.manifest["model"]
        )


        # -------------------------------------------------
        # 6) تحميل Cross-Encoder
        #
        # هذا يستخدم في Stage 2
        # عشان يرتب الـcandidates بدقة أعلى.
        # -------------------------------------------------
        self.reranker = CrossEncoder(
            RERANKER_MODEL
        )


        # -------------------------------------------------
        # 7) Integrity checks
        #
        # نتأكد إن الـindex والmanifest والmetadata
        # كلهم تابعين لنفس build.
        # -------------------------------------------------
        if self.index.ntotal != self.manifest["n_vectors"]:
            raise ValueError(
                "Index vector count does not match manifest."
            )

        if self.index.d != self.manifest["dim"]:
            raise ValueError(
                "Index dimension does not match manifest."
            )

        if len(self.meta) != self.manifest["n_vectors"]:
            raise ValueError(
                "Metadata row count does not match index."
            )


    def search(
        self,
        query: str,
        k: int = 5,
        candidates: int = 50,
        min_score: float = 0.25,
    ) -> list[dict]:

        # -------------------------------------------------
        # لو الـquery فاضي ما نبحث
        # -------------------------------------------------
        if query is None or not query.strip():
            return []


        # -------------------------------------------------
        # 1) تحويل Query إلى Embedding
        #
        # IMPORTANT:
        # في Part 1 الحالي خزنا case_text بدون
        # Arabic normalization إضافي،
        # لذلك هنا نستخدم query كما هي أيضًا
        # عشان preprocessing يكون consistent.
        # -------------------------------------------------
        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True,
        )


        # -------------------------------------------------
        # 2) L2 Normalization
        #
        # نفس الشيء اللي سويناه للـcase embeddings
        # في Part 1.
        # -------------------------------------------------
        faiss.normalize_L2(
            query_embedding
        )


        # -------------------------------------------------
        # 3) STAGE 1
        #
        # FAISS يبحث بسرعة عن أقرب candidates
        #
        # مثال:
        # 20,000 cases
        #
        # ↓
        #
        # أفضل 50 candidate
        # -------------------------------------------------

        # ما نطلب candidates أكثر من الموجود بالـindex
        candidate_count = min(
            candidates,
            self.index.ntotal,
        )

        scores, indices = self.index.search(
            query_embedding,
            candidate_count,
        )


        # -------------------------------------------------
        # FAISS ممكن يرجع -1 إذا ما وجد index صالح
        # -------------------------------------------------
        valid_positions = [
            int(i)
            for i in indices[0]
            if i >= 0
        ]

        if not valid_positions:
            return []


        # -------------------------------------------------
        # نجيب الحالات الأصلية من metadata
        # -------------------------------------------------
        candidate_df = (
            self.meta
            .iloc[valid_positions]
            .copy()
            .reset_index(drop=True)
        )


        # نحفظ FAISS score أيضًا
        candidate_df["bi_score"] = [
            float(score)
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]


        # -------------------------------------------------
        # 4) STAGE 2
        #
        # Cross-Encoder يشوف:
        #
        # (query, candidate text)
        #
        # ويعطي score أدق لكل candidate.
        # -------------------------------------------------

        pairs = [
            (query, case_text)
            for case_text in candidate_df["case_text"]
            .fillna("")
            .astype(str)
        ]


        ce_scores = self.reranker.predict(
            pairs
        )


        candidate_df["ce_score"] = ce_scores


        # -------------------------------------------------
        # 5) إعادة ترتيب النتائج
        #
        # أعلى Cross-Encoder score أول
        # -------------------------------------------------
        candidate_df = (
            candidate_df
            .sort_values(
                "ce_score",
                ascending=False,
            )
            .head(k)
        )


        # -------------------------------------------------
        # 6) Honest empty result
        #
        # إذا حتى أفضل نتيجة score حقها ضعيف
        # ما نقول للمستخدم إنها مناسبة.
        #
        # نرجع []
        # -------------------------------------------------
        if candidate_df.empty:
            return []

        if float(candidate_df["ce_score"].max()) < min_score:
            return []


        # -------------------------------------------------
        # 7) تحويل النتائج إلى list of dictionaries
        # -------------------------------------------------
        return candidate_df.to_dict(
            orient="records"
        )
        
        
        
        """
        User query
   ↓
Bi-Encoder
   ↓
Query Embedding
   ↓
FAISS
   ↓
أفضل 50 Candidate
   ↓
Cross-Encoder
   ↓
إعادة ترتيب
   ↓
أفضل 5 نتائج
        """