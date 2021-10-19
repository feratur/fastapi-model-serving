#!/usr/bin/env python3
import pandas as pd
from fastapi import FastAPI, Request
from joblib import load
from pathlib import Path

app = FastAPI()
column_names = Path('columns.txt').read_text().split()
model = load('model.joblib')


def parse_as_dataframe(body):
    return pd.DataFrame.from_dict(body)[column_names]


@app.get("/predict")
async def get_predict(
    age: int,
    sex: int,
    cp: int,
    trestbps: int,
    chol: int,
    fbs: int,
    restecg: int,
    thalach: int,
    exang: int,
    oldpeak: float,
    slope: int,
    ca: int,
    thal: int,
):
    batch = parse_as_dataframe([dict(
        age=age,
        sex=sex,
        cp=cp,
        trestbps=trestbps,
        chol=chol,
        fbs=fbs,
        restecg=restecg,
        thalach=thalach,
        exang=exang,
        oldpeak=oldpeak,
        slope=slope,
        ca=ca,
        thal=thal,
    )])
    return model.predict(batch).tolist()


@app.get("/predict_proba")
async def get_predict_proba(
    age: int,
    sex: int,
    cp: int,
    trestbps: int,
    chol: int,
    fbs: int,
    restecg: int,
    thalach: int,
    exang: int,
    oldpeak: float,
    slope: int,
    ca: int,
    thal: int,
):
    batch = parse_as_dataframe([dict(
        age=age,
        sex=sex,
        cp=cp,
        trestbps=trestbps,
        chol=chol,
        fbs=fbs,
        restecg=restecg,
        thalach=thalach,
        exang=exang,
        oldpeak=oldpeak,
        slope=slope,
        ca=ca,
        thal=thal,
    )])
    return model.predict_proba(batch).tolist()


@app.post("/predict")
async def post_predict(request: Request):
    batch = parse_as_dataframe(await request.json())
    return model.predict(batch).tolist()


@app.post("/predict_proba")
async def post_predict_proba(request: Request):
    batch = parse_as_dataframe(await request.json())
    return model.predict_proba(batch).tolist()
