"""Tiny client that exercises the running API.   1) uvicorn api:app    2) python try_api.py FD001"""
import json
import sys

import requests

from src.data import load_fd
from src.service import REQUIRED

BASE = "http://127.0.0.1:8000"
fd = sys.argv[1] if len(sys.argv) > 1 else "FD001"

print("health :", requests.get(f"{BASE}/health").json())
print("demo   :", requests.get(f"{BASE}/demo/{fd}/1").json())

_, test, rul_true = load_fd("data", fd)
eng = test[test.unit == 1].tail(80)  # last 80 cycles of engine 1, sent as JSON
body = {"dataset": fd, "readings": json.loads(eng[REQUIRED].to_json(orient="records"))}
r = requests.post(f"{BASE}/predict", json=body)
print("predict:", r.status_code, r.json(), "| official true RUL:", int(rul_true[0]))
