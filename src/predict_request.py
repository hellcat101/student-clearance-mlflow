"""Send a sample inference request to the served model (Step 3: expose for inference).

Start the server first (see README):
    mlflow models serve -m "models:/student-clearance-classifier/Production" -p 5001 --no-conda

Then run:
    python src/predict_request.py
"""
from __future__ import annotations

import json

import requests

URL = "http://127.0.0.1:5001/invocations"

# MLflow's scoring server accepts the "dataframe_split" format.
payload = {
    "dataframe_split": {
        "columns": [
            "Department",
            "University",
            "Degree Level",
            "Country",
            "City",
            "Semester",
            "Section",
        ],
        "data": [
            ["Mathematics", "Massachusetts Institute of Technology (MIT)",
             "MS", "United States", "Cambridge", "3rd", "A"],
            ["Mechanical Engineering", "Stanford University",
             "PhD", "United States", "Stanford", "1st", "B"],
        ],
    }
}

if __name__ == "__main__":
    resp = requests.post(
        URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload)
    )
    print("status:", resp.status_code)
    print("predictions (1=cleared, 0=at-risk):", resp.json())
