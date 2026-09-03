# Smile Guard Vision Service

面向微笑守护记录小程序的“面瘫视觉服务”，部署到微信云托管。

## 部署

1. 在微信云开发控制台进入“云托管”。
2. 新建服务，选择“从源码构建”。
3. 绑定本仓库，构建目录使用根目录，Dockerfile 路径：

```text
vision-service/Dockerfile
```

4. 设置环境变量：

```text
VISION_SERVICE_TOKEN=你的64位Token
```

5. 部署后获取 HTTPS 域名，例如：

```text
https://xxxx.service.tcloudbase.com
```

6. 在 `facial-analysis` 云函数环境变量配置：

```text
VISION_SERVICE_URL=https://xxxx.service.tcloudbase.com/analyze
VISION_SERVICE_TOKEN=你的64位Token
MOCK_MODE=false
```

## 本地测试

```bash
pip install -r vision-service/requirements.txt
VISION_SERVICE_TOKEN=你的64位Token uvicorn vision-service.app:app --host 0.0.0.0 --port 8000
```

```bash
curl http://127.0.0.1:8000/health
```
