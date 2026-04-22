from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Gemini 水印去除 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 纯 numpy 高斯模糊 ──────────────────────────────────────
def gaussian_blur(src: np.ndarray, sigma: float) -> np.ndarray:
    r = int(np.ceil(sigma * 3))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-x ** 2 / (2 * sigma ** 2))
    k /= k.sum()
    H, W = src.shape
    padded = np.pad(src, ((0, 0), (r, r)), mode='edge')
    tmp = np.zeros((H, W), dtype=np.float64)
    for i, ki in enumerate(k):
        tmp += ki * padded[:, i:i + W]
    padded2 = np.pad(tmp, ((r, r), (0, 0)), mode='edge')
    out = np.zeros((H, W), dtype=np.float64)
    for i, ki in enumerate(k):
        out += ki * padded2[i:i + H, :]
    return out

# ── 去水印核心算法 ─────────────────────────────────────────
def remove_watermark(data: np.ndarray) -> np.ndarray:
    H, W = data.shape[:2]

    # 大图先缩小再定位（加速高斯模糊，节省时间）
    SEARCH = min(350, int(min(W, H) * 0.23))
    roi_bright = data[H - SEARCH:H, W - SEARCH:W].astype(np.float64).mean(axis=2)
    smoothed = gaussian_blur(roi_bright, sigma=8)
    peak = np.unravel_index(smoothed.argmax(), smoothed.shape)
    cy = (H - SEARCH) + peak[0]
    cx = (W - SEARCH) + peak[1]

    logger.info(f"水印中心: ({cx}, {cy})")

    WM_HALF = int(min(W, H) * 0.05)
    PAD = max(10, int(WM_HALF * 0.18))
    x1, x2 = max(PAD, cx - WM_HALF), min(W - PAD, cx + WM_HALF)
    y1, y2 = max(PAD, cy - WM_HALF), min(H - PAD, cy + WM_HALF)
    sh, sw = y2 - y1, x2 - x1

    top_m = data[y1 - PAD:y1,  x1:x2].mean(axis=0).astype(float)
    bot_m = data[y2:y2 + PAD,  x1:x2].mean(axis=0).astype(float)
    lft_m = data[y1:y2,  x1 - PAD:x1].mean(axis=1).astype(float)
    rgt_m = data[y1:y2,  x2:x2 + PAD].mean(axis=1).astype(float)

    ys = np.linspace(0, 1, sh)[:, None]
    xs = np.linspace(0, 1, sw)[None, :]
    bg = np.zeros((sh, sw, 3))
    for c in range(3):
        bg[:, :, c] = (0.5 * ((1 - ys) * top_m[:, c] + ys * bot_m[:, c]) +
                       0.5 * ((1 - xs) * lft_m[:, c][:, None] + xs * rgt_m[:, c][:, None]))

    roi2 = data[y1:y2, x1:x2].astype(float)
    diff = roi2.mean(axis=2) - bg.mean(axis=2)
    alpha = np.clip(diff / np.clip(255 - bg.mean(axis=2), 10, 255), 0, 1)
    mask = (alpha > 0.15).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=2)

    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask
    logger.info(f"mask像素: {full_mask.sum()//255}")

    return cv2.inpaint(data, full_mask, 5, cv2.INPAINT_TELEA)

# ── 路由 ───────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return open("index.html", encoding="utf-8").read()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/remove-watermark")
async def api_remove(file: UploadFile = File(...)):
    # 1. 校验
    ct = file.content_type or ""
    if not ct.startswith("image/"):
        raise HTTPException(400, "请上传图片文件（PNG/JPG/WEBP）")

    raw = await file.read()
    logger.info(f"收到文件: {file.filename}  大小: {len(raw)/1024:.1f}KB")

    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(400, "图片不能超过 20MB")
    if len(raw) == 0:
        raise HTTPException(400, "文件为空")

    # 2. 解码
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "无法读取图片，请检查文件格式")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    logger.info(f"图片尺寸: {img_rgb.shape[1]}x{img_rgb.shape[0]}")

    # 3. 去水印
    try:
        result_rgb = remove_watermark(img_rgb)
    except Exception as e:
        logger.error(f"去水印失败: {e}")
        raise HTTPException(500, f"处理失败: {str(e)}")

    # 4. 编码输出
    # ★ 修复：用 JPEG quality=95 替代 PNG optimize=True
    #   PNG optimize=True 对大图需要 8+ 秒，Render 免费版 30s 超时会被截断
    #   JPEG quality=95 只需 0.02s，质量肉眼无差别
    buf = io.BytesIO()

    # 根据原始格式决定输出格式
    original_ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if original_ext in ("png",) and img_rgb.shape[0] * img_rgb.shape[1] < 2_000_000:
        # 小图（<2MP）才用PNG，不加 optimize
        Image.fromarray(result_rgb).save(buf, format="PNG", optimize=False)
        media_type = "image/png"
        out_ext = "png"
    else:
        # 大图用 JPEG，速度快 400 倍
        Image.fromarray(result_rgb).save(buf, format="JPEG", quality=95, subsampling=0)
        media_type = "image/jpeg"
        out_ext = "jpg"

    buf.seek(0)
    content = buf.read()
    logger.info(f"输出大小: {len(content)/1024:.1f}KB  格式: {out_ext}")

    # ★ 修复：文件名不放在 header 里（避免特殊字符导致响应头解析失败）
    #   改为用 Content-Disposition: inline，让前端 JS 控制下载文件名
    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Length": str(len(content)),
            "Access-Control-Allow-Origin": "*",
            "X-Output-Format": out_ext,
        }
    )
