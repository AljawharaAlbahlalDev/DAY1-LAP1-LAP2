#!/usr/bin/env bash

# ================================================================
# Lab 7.6 — HTTP Load Test
# ================================================================
#
# الهدف:
# نختبر الـ API مثل production traffic،
# مو request واحد فقط.
#
# إعداد الكورس:
#
#   concurrent clients = 16
#   duration           = 60 seconds
#
# RUN:
#
#   bash scripts/load_test.sh
#
# قبل التشغيل:
# لازم FastAPI تكون شغالة:
#
#   uvicorn bayan.serving.api:app \
#       --host 0.0.0.0 \
#       --port 8000
#
# COURSE TARGET:
#
#   HTTP p99 <= 40 ms
#   0 request errors
#   startup canaries green
#
# ملاحظة:
# latency تعتمد على الجهاز.
# لا تنسخ reference number.
# سجلي measured result في BENCHMARKS.md.
# ================================================================


# إذا أي command فشل،
# نوقف script.
set -euo pipefail


# ------------------------------------------------------------
# API URLs
# ------------------------------------------------------------

HEALTH_URL="http://127.0.0.1:8000/health"

CLASSIFY_URL="http://127.0.0.1:8000/v1/classify"


# ------------------------------------------------------------
# Test request body
# ------------------------------------------------------------

BODY='{
  "text":
  "الإنارة متعطلة في الشارع منذ يومين"
}'


# ------------------------------------------------------------
# Step 1 — Health check
# ------------------------------------------------------------

echo "Checking Bayan API health..."

curl -fsS \
    "${HEALTH_URL}"

echo
echo


# ------------------------------------------------------------
# Step 2 — Start load test
# ------------------------------------------------------------
#
# hey:
#
#   -z 60s
#       run for 60 seconds
#
#   -c 16
#       16 concurrent clients
#
#   -m POST
#       HTTP POST
#
# ------------------------------------------------------------

echo "Starting Lab 7 load test..."
echo
echo "Concurrent clients: 16"
echo "Duration:           60 seconds"
echo "Endpoint:           ${CLASSIFY_URL}"
echo


hey \
    -z 60s \
    -c 16 \
    -m POST \
    -H "Content-Type: application/json" \
    -d "${BODY}" \
    "${CLASSIFY_URL}"


# ================================================================
# EXPECTED OUTPUT SHAPE
# ================================================================
#
# Summary:
#
#   Total:        ...
#   Slowest:      ...
#   Fastest:      ...
#   Average:      ...
#   Requests/sec: ...
#
#
# Response time histogram:
#
#   ...
#
#
# Latency distribution:
#
#   50% in ...
#   75% in ...
#   90% in ...
#   95% in ...
#   99% in ...
#
#
# Status code distribution:
#
#   [200] ...
#
#
# COURSE ACCEPTANCE:
#
#   99% latency <= 0.040 sec
#
#   no request errors
#
#   all/most responses:
#       HTTP 200
#
# IMPORTANT:
#
# النتائج تعتمد على:
#   CPU
#   OS
#   number of threads
#   model
#   ONNX / PyTorch
#
# سجلي measured result الحقيقي.
# ================================================================


"""
# ================================================================
# LAB 7 — FINAL CHECK
# ================================================================


# ------------------------------------------------------------
# 1) Pin CPU threads
# ------------------------------------------------------------

export OMP_NUM_THREADS=4


# ------------------------------------------------------------
# 2) Serving contract tests
# ------------------------------------------------------------

pytest tests/test_serving_contract.py -q


# ------------------------------------------------------------
# 3) Baseline CPU benchmark
# ------------------------------------------------------------

python scripts/benchmark_inference.py


# ------------------------------------------------------------
# 4) Export ONNX + INT8
# ------------------------------------------------------------

python scripts/export_onnx.py


# ------------------------------------------------------------
# 5) Benchmark again after optimisation
# ------------------------------------------------------------

python scripts/benchmark_inference.py


# ------------------------------------------------------------
# 6) Start API
# ------------------------------------------------------------

uvicorn bayan.serving.api:app \
    --host 0.0.0.0 \
    --port 8000


# ------------------------------------------------------------
# 7) In another terminal:
# ------------------------------------------------------------

bash scripts/load_test.sh
"""