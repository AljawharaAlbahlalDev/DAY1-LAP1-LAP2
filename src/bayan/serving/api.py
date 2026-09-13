"""
Lab 7.5 — FastAPI Serving

RUN:

    uvicorn bayan.serving.api:app \
        --host 0.0.0.0 \
        --port 8000

CHECK:

    http://localhost:8000/health

CLASSIFY:

    POST /v1/classify
"""

import numpy as np
import torch

from fastapi import (
    FastAPI,
)

from pydantic import (
    BaseModel,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


MODEL_PATH = (
    "artifacts/"
    "topic_classifier"
)


app = FastAPI(
    title="Bayan NLP API"
)


tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_PATH
    )
)


model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_PATH
    )
)


model.eval()


class ClassifyRequest(
    BaseModel
):
    text: str


@app.get(
    "/health"
)
def health():

    return {
        "status": "ok"
    }


@app.post(
    "/v1/classify"
)
def classify(
    request: ClassifyRequest,
):

    batch = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    with torch.inference_mode():

        logits = (
            model(
                **batch
            )
            .logits[0]
            .numpy()
        )

    probabilities = np.exp(
        logits
        - logits.max()
    )

    probabilities = (
        probabilities
        / probabilities.sum()
    )

    prediction_id = int(
        probabilities.argmax()
    )

    label = (
        model.config.id2label[
            prediction_id
        ]
    )

    return {
        "label": label,
        "score": float(
            probabilities[
                prediction_id
            ]
        ),
    }


# EXPECTED:
#
# GET /health
#
# {
#   "status": "ok"
# }
#
# POST /v1/classify
#
# {
#   "label": "<topic>",
#   "score": 0.xxxx
# }