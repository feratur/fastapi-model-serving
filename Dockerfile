FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY ./requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

COPY ./dataset.csv ./train_model.py /tmp/
RUN python3 /tmp/train_model.py /tmp/dataset.csv /app/

COPY ./main.py /app/main.py
