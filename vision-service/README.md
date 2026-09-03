# 微笑守护记录 - 面瘫视觉服务

基于 MediaPipe Face Landmarker 与现有面瘫风险模型，接收面部照片 URL，
返回与小程序 `aiReference` 一致的结构。

## 本地启动

```bash
cd smile-guard-vision
pip install -r vision-service/requirements.txt
uvicorn vision-service.app:app --host 0.0.0.0 --port 8000
```

如需启用鉴权，设置环境变量后启动：

```bash
VISION_SERVICE_TOKEN=你的Token uvicorn vision-service.app:app --host 0.0.0.0 --port 8000
```

## Docker 部署

```bash
cd smile-guard-vision
docker build -f vision-service/Dockerfile -t smile-guard-vision .
docker run -d -p 8000:8000 smile-guard-vision
```

## 部署确认

确认服务已启动：

```bash
curl http://127.0.0.1:8000/health
```

返回：

```json
{"status":"ok","service":"smile-guard-vision"}
```

本地测试一张正脸照片：

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 你的Token" \
  -d '{"files":[{"url":"https://example.com/face.jpg","type":"image"}]}'
```

能返回 `aiReference` 即说明视觉服务已可用。

## 接口

- `GET /health`：健康检查
- `POST /analyze`：

```json
{
  "patient": { "patientId": "p1", "diseaseKey": "facialPalsy" },
  "files": [
    { "url": "https://example.com/photo.jpg", "type": "image" }
  ]
}
```

返回：

```json
{
  "aiReference": {
    "possibleFacialPalsy": true,
    "symmetryScore": 62,
    "hbGrade": "III",
    "confidence": 0.81,
    "findings": "...",
    "advice": "..."
  },
  "doctorResult": null
}
```

## 云函数对接

在 `facial-analysis` 云函数环境变量中配置：

```text
VISION_SERVICE_URL=https://你的视觉服务域名/analyze
VISION_SERVICE_TOKEN=你的鉴权Token
MOCK_MODE=false
```

`facial-analysis` 云函数会把云存储文件转成临时 URL 后 POST 给本服务。
本服务仅输出 AI 参考结果，不做诊断。
