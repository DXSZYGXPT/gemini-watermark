from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import io

app = FastAPI(title="Gemini 水印去除 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

def remove_watermark(img_array: np.ndarray) -> np.ndarray:
    data = img_array
    H, W = data.shape[:2]

    # Step 1: 高斯平滑定位水印中心
    SEARCH = min(350, int(min(W, H) * 0.23))
    roi = data[H-SEARCH:H, W-SEARCH:W].astype(float)
    smoothed = gaussian_filter(roi.mean(axis=2), sigma=8)
    peak = np.unravel_index(smoothed.argmax(), smoothed.shape)
    cy = (H - SEARCH) + peak[0]
    cx = (W - SEARCH) + peak[1]

    # Step 2: 四周精确取背景（不触碰图片边界）
    WM_HALF = int(min(W, H) * 0.05)
    PAD = max(10, int(WM_HALF * 0.18))
    x1, x2 = max(PAD, cx-WM_HALF), min(W-PAD, cx+WM_HALF)
    y1, y2 = max(PAD, cy-WM_HALF), min(H-PAD, cy+WM_HALF)
    sh, sw = y2-y1, x2-x1

    top_m = data[y1-PAD:y1, x1:x2].mean(axis=0).astype(float)
    bot_m = data[y2:y2+PAD, x1:x2].mean(axis=0).astype(float)
    lft_m = data[y1:y2, x1-PAD:x1].mean(axis=1).astype(float)
    rgt_m = data[y1:y2, x2:x2+PAD].mean(axis=1).astype(float)

    ys = np.linspace(0, 1, sh)[:, None]
    xs = np.linspace(0, 1, sw)[None, :]
    bg = np.zeros((sh, sw, 3))
    for c in range(3):
        bg[:,:,c] = 0.5*((1-ys)*top_m[:,c] + ys*bot_m[:,c]) + \
                    0.5*((1-xs)*lft_m[:,c][:,None] + xs*rgt_m[:,c][:,None])

    # Step 3: diff → alpha → mask
    roi2 = data[y1:y2, x1:x2].astype(float)
    diff = roi2.mean(axis=2) - bg.mean(axis=2)
    alpha = np.clip(diff / np.clip(255 - bg.mean(axis=2), 10, 255), 0, 1)
    mask = (alpha > 0.15).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=2)

    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask

    # Step 4: TELEA inpainting
    result = cv2.inpaint(data, full_mask, 5, cv2.INPAINT_TELEA)
    return result

@app.get("/", response_class=HTMLResponse)
async def index():
    return open("index.html", encoding="utf-8").read()

@app.post("/remove-watermark")
async def api_remove(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "请上传图片文件")

    raw = await file.read()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(400, "图片不能超过 20MB")

    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "无法读取图片")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result_rgb = remove_watermark(img_rgb)

    pil_img = Image.fromarray(result_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    buf.seek(0)

    fname = file.filename.rsplit(".", 1)[0] + "_no_watermark.png"
    return Response(
        content=buf.read(),
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'}
    )
