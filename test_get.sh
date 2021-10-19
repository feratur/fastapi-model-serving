#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
cd "$SCRIPT_DIR"

curl 'http://localhost:80/predict_proba?age=49&ca=0&chol=266&cp=1&exang=0&fbs=0&oldpeak=0.6&restecg=1&sex=1&slope=2&thal=2&thalach=171&trestbps=130'
