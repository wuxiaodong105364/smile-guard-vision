"""Smile Guard facial palsy vision service.

Receives photo URLs from the facial-analysis cloud function and returns an
AI-reference JSON compatible with the mini-program aiReference shape.
"""

import os
import sys
import tempfile
import urllib.request
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import analyze_image  # noqa: E402


app = FastAPI(title="Smile Guard Vision Service", version="1.0.0")
VISION_SERVICE_TOKEN = os.getenv("VISION_SERVICE_TOKEN", "")


def require_token(authorization: str = Header(default="")):
    if not VISION_SERVICE_TOKEN:
        return
    expected = "Bearer " + VISION_SERVICE_TOKEN
    if authorization != expected:
        raise HTTPException(status_code=401, detail="invalid vision service token")


class FileItem(BaseModel):
    url: str = ""
    fileID: str = ""
    type: str = "image"
    size: int = 0
    duration: int = 0


class PatientItem(BaseModel):
    patientId: str = ""
    diseaseKey: str = "facialPalsy"
    surgeryDate: str = ""


class AnalyzeRequest(BaseModel):
    patient: PatientItem = PatientItem()
    files: list[FileItem] = []


def download(url, out_path):
    req = urllib.request.Request(url, headers={"User-Agent": "smile-guard-vision"})
    with urllib.request.urlopen(req, timeout=30) as response, open(out_path, "wb") as f:
        f.write(response.read())


def insufficient_reference(message="照片不足或无法识别人脸，请重新拍摄正脸照片。"):
    return {
        "possibleFacialPalsy": None,
        "symmetryScore": None,
        "hbGrade": None,
        "confidence": 0.0,
        "informationInsufficient": True,
        "findings": message,
        "advice": "请按正脸放松、抬眉、闭眼、示齿微笑重新拍摄，或联系主管医生面诊。",
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "smile-guard-vision"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest, _auth: None = Depends(require_token)):
    results = []
    for item in request.files:
        if not item.url:
            continue
        suffix = ".jpg"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb"):
                download(item.url, tmp_path)
            reference = analyze_image(tmp_path)
            if reference and not reference.get("informationInsufficient"):
                results.append(reference)
        except Exception:
            pass
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    if not results:
        return {"aiReference": insufficient_reference(), "doctorResult": None}
    best = max(results, key=lambda r: r.get("riskProbability") or 0.0)
    best["findings"] = (
        "AI 参考：已分析 %d 张照片，取风险最高的一张。%s"
        % (len(results), best.get("findings", ""))
    )
    return {"aiReference": best, "doctorResult": None}
