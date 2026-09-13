"""
Lab 7.2 + 7.3

Step 2:
    PyTorch fp32 -> ONNX fp32

Step 3:
    ONNX fp32 -> ONNX INT8

INSTALL:
    pip install "optimum[onnxruntime]" onnx onnxruntime

RUN:
    python scripts/export_onnx.py

OUTPUT:
    artifacts/onnx-fp32
    artifacts/onnx-int8

مهم:
نحتفظ fp32 كـ rollback.
"""

from pathlib import Path

from optimum.onnxruntime import (
    ORTModelForSequenceClassification,
    ORTQuantizer,
)

from optimum.onnxruntime.configuration import (
    AutoQuantizationConfig,
)

from transformers import (
    AutoTokenizer,
)


SOURCE = (
    "artifacts/"
    "topic_classifier"
)

FP32_OUTPUT = Path(
    "artifacts/"
    "onnx-fp32"
)

INT8_OUTPUT = Path(
    "artifacts/"
    "onnx-int8"
)


def export_fp32():

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            SOURCE
        )
    )

    model = (
        ORTModelForSequenceClassification
        .from_pretrained(
            SOURCE,
            export=True,
        )
    )

    FP32_OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        FP32_OUTPUT
    )

    tokenizer.save_pretrained(
        FP32_OUTPUT
    )

    print(
        f"ONNX fp32 saved to: "
        f"{FP32_OUTPUT}"
    )


def quantize_int8():

    onnx_files = list(
        FP32_OUTPUT.glob(
            "*.onnx"
        )
    )

    if not onnx_files:

        raise FileNotFoundError(
            "No ONNX model found."
        )

    model_file = (
        onnx_files[0]
    )

    quantizer = (
        ORTQuantizer
        .from_pretrained(
            FP32_OUTPUT,
            file_name=(
                model_file.name
            ),
        )
    )

    # Dynamic INT8 quantisation.
    #
    # AVX2 مثال مناسب لمعظم CPU labs.

    config = (
        AutoQuantizationConfig
        .avx2(
            is_static=False
        )
    )

    quantizer.quantize(
        save_dir=INT8_OUTPUT,
        quantization_config=config,
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            FP32_OUTPUT
        )
    )

    tokenizer.save_pretrained(
        INT8_OUTPUT
    )

    print(
        f"ONNX INT8 saved to: "
        f"{INT8_OUTPUT}"
    )


def main():

    export_fp32()

    quantize_int8()

    print(
        "\nDone."
    )

    print(
        f"Rollback fp32: "
        f"{FP32_OUTPUT}"
    )

    print(
        f"INT8 candidate: "
        f"{INT8_OUTPUT}"
    )


if __name__ == "__main__":
    main()


# EXPECTED:
#
# ONNX fp32 saved to:
# artifacts/onnx-fp32
#
# ONNX INT8 saved to:
# artifacts/onnx-int8
#
# بعدها:
# شغلي benchmark مرة ثانية
# وسجلي p50 / p99 / speed-up.