"""Smile Guard facial palsy vision service.

Receives photo URLs from the facial-analysis cloud function and returns an
AI-reference JSON compatible with the mini-program aiReference shape.
"""

import os
import sys
import tempfile
import urllib.request
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import analyze_image, analyze_video  # noqa: E402


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


def analyze_local_path(path, media_type="image"):
    if media_type in ("video", "mp4", "mov", "avi"):
        return analyze_video(path)
    return analyze_image(path)


def analyze_downloaded(url, media_type="image"):
    if not url:
        return None
    fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    try:
        with os.fdopen(fd, "wb"):
            download(url, tmp_path)
        return analyze_local_path(tmp_path, media_type)
    except Exception:
        return None
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


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
        reference = analyze_downloaded(item.url, item.type)
        if reference and not reference.get("informationInsufficient"):
            results.append(reference)
    if not results:
        return {"aiReference": insufficient_reference(), "doctorResult": None}
    best = max(results, key=lambda r: r.get("riskProbability") or 0.0)
    best["findings"] = (
        "AI 参考：已分析 %d 张照片，取风险最高的一张。%s"
        % (len(results), best.get("findings", ""))
    )
    return {"aiReference": best, "doctorResult": None}


@app.api_route("/analyze_simple", methods=["GET", "POST"])
def analyze_simple(
    image_url: str = "",
    media_type: str = "image",
    patient_id: str = "",
    disease_key: str = "facialPalsy",
    surgery_date: str = "",
    _auth: None = Depends(require_token),
):
    reference = analyze_downloaded(image_url, media_type)
    if not reference:
        reference = insufficient_reference()
    reference["diseaseKey"] = disease_key
    reference["patientId"] = patient_id
    return reference


@app.post("/analyze_upload")
def analyze_upload(
    image_file: UploadFile = File(...),
    media_type: str = Form("image"),
    patient_id: str = Form(""),
    disease_key: str = Form("facialPalsy"),
    surgery_date: str = Form(""),
    _auth: None = Depends(require_token),
):
    suffix = ".mp4" if media_type in ("video", "mp4", "mov", "avi") else ".jpg"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    reference = None
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(image_file.file.read())
        reference = analyze_local_path(tmp_path, media_type)
    except Exception:
        reference = None
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    if not reference:
        reference = insufficient_reference()
    reference["diseaseKey"] = disease_key
    reference["patientId"] = patient_id
    return reference
