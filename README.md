# fastapi-model-serving
Simplest example of serving sklearn model through FastAPI REST service

### How to use
Service can be built and launched simply by `docker-compose up`. The command does the following:
- Installs necessary requirements (mainly `sklearn`)
- Trains simple classifier on the following dataset: https://www.kaggle.com/ronitf/heart-disease-uci?select=heart.csv
- Saves the model inside the Docker image
- Launches `tiangolo/uvicorn-gunicorn-fastapi` web service (on port 80) for model inference (classification task)

### Endpoints
The service launches on port 80 by default and provides `/predict` and `/predict_proba` methods.
Both `/predict` and `/predict_proba` can be used via `HTTP GET` (for single vector prediction) or `HTTP POST` (for batch prediction).

### Examples
Script `./test_get.sh` is an example of calling `/predict_proba` on a single vector via GET request.
Script `./test_post.sh` is an example of calling `/predict` on a batch of values (from the file `./testRequest.json`) via POST request.
