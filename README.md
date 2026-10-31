# AryanSignAI Upgrade 02

Responsive local webcam hand-tracking prototype with estimated finger count and session history.

## Run on Windows / Python 3.11
```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## Limitation
This prototype uses a simple finger-extension heuristic. It is not a validated Indian Sign Language (ISL) alphabet or word translator. Real ISL recognition needs a labelled dataset, a trained model, and evaluation with ISL users.
