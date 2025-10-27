from typing import List, Dict, Any, Optional
import os
import io
import urllib.request
from urllib.parse import urlparse

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request
from PIL import Image
from fastapi.responses import JSONResponse

from face_db import FaceDatabase

try:
    from insightface.app import FaceAnalysis
except Exception as e:
    FaceAnalysis = None

app = FastAPI(title="InsightFace 人脸库 API", description="基于 InsightFace 的人脸库接口")

# 使用单独的 JSON 存储，避免与 face_recognition 的编码混用
INSIGHT_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "insightface_db.json")
db = FaceDatabase(db_path=os.path.abspath(INSIGHT_DB_PATH))

# 初始化 InsightFace 分析器
if FaceAnalysis is None:
    raise RuntimeError("insightface 未安装或无法导入，请先安装 insightface 与 onnxruntime")

face_analyzer = FaceAnalysis(name="buffalo_l", root='/app/models', providers=["CPUExecutionProvider"])  # CPU 推理，跨平台可用
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

# 统一响应封装

def resp_ok(data: Any = None, msg: str = "") -> Dict[str, Any]:
    return {"code": 0, "data": data, "msg": msg}


def resp_fail(msg: str = "", code: int = 1, data: Any = None) -> Dict[str, Any]:
    return {"code": code, "data": data, "msg": msg}


# 全局异常处理，统一错误响应格式

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail if getattr(exc, "detail", None) is not None else str(exc)
    return JSONResponse(status_code=exc.status_code, content=resp_fail(msg=str(detail)))


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # 未捕获异常归为服务器错误
    return JSONResponse(status_code=500, content=resp_fail(msg="服务器内部错误"))


# 图片读取辅助

def _pil_to_bgr_ndarray(upload: UploadFile) -> np.ndarray:
    upload.file.seek(0)
    img = Image.open(upload.file).convert("RGB")
    arr = np.array(img)  # RGB
    bgr = arr[:, :, ::-1].copy()  # 转为 BGR
    return bgr


def _url_to_bgr_ndarray(image_url: str) -> np.ndarray:
    parsed = urlparse(image_url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="请提供有效的图片URL")
    try:
        req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        arr = np.array(img)
        bgr = arr[:, :, ::-1].copy()
        return bgr
    except Exception:
        raise HTTPException(status_code=400, detail="无法从URL加载图片")


@app.get("/people")
def list_people() -> Dict[str, Any]:
    return resp_ok(db.list_people())


@app.post("/people")
async def add_person(
    name: str = Form(...),
    image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
) -> Dict[str, Any]:
    if image is None and not image_url:
        raise HTTPException(status_code=400, detail="请上传图片或提供图片URL")

    if image is not None:
        if not image.content_type or "image" not in image.content_type:
            raise HTTPException(status_code=400, detail="请上传图片文件")
        img_bgr = _pil_to_bgr_ndarray(image)
    else:
        img_bgr = _url_to_bgr_ndarray(image_url)  # type: ignore[arg-type]

    faces = face_analyzer.get(img_bgr)
    if not faces:
        raise HTTPException(status_code=400, detail="图片中未检测到人脸")
    if len(faces) > 1:
        raise HTTPException(status_code=400, detail="图片中检测到多张人脸，请上传仅包含一个人脸的照片")

    emb = faces[0].normed_embedding  # 512 维，已归一化
    if emb is None or len(emb) == 0:
        raise HTTPException(status_code=400, detail="未能提取人脸特征")

    saved = db.add_person(name=name, encoding=emb.tolist())
    return resp_ok({"person": saved, "faces_detected": len(faces)})


@app.delete("/people/{person_id}")
async def delete_person(person_id: str) -> Dict[str, Any]:
    try:
        removed = db.delete_person(person_id)
        return resp_ok({"deleted": removed})
    except ValueError:
        raise HTTPException(status_code=404, detail="人员不存在")


@app.post("/recognize")
async def recognize(
    image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    similarity_threshold: float = Query(0.35, ge=0.1, le=0.99, description="匹配阈值（cosine），默认0.35")
) -> Dict[str, Any]:
    if image is None and not image_url:
        raise HTTPException(status_code=400, detail="请上传图片或提供图片URL")

    if image is not None:
        if not image.content_type or "image" not in image.content_type:
            raise HTTPException(status_code=400, detail="请上传图片文件")
        img_bgr = _pil_to_bgr_ndarray(image)
    else:
        img_bgr = _url_to_bgr_ndarray(image_url)  # type: ignore[arg-type]

    known = db.get_known_encodings()
    if not known:
        return resp_ok({"matches": [], "match_count": 0, "faces_detected": 0, "threshold": similarity_threshold})

    # 准备已知编码矩阵与人员映射
    known_matrix = np.array([k["encoding"] for k in known], dtype=np.float32)  # (N, D)
    person_ids = [k["person_id"] for k in known]
    names = [k["name"] for k in known]

    faces = face_analyzer.get(img_bgr)
    if not faces:
        return resp_ok({"matches": [], "match_count": 0, "faces_detected": 0, "threshold": similarity_threshold})

    results: List[Dict[str, Any]] = []
    for f in faces:
        emb = f.normed_embedding
        if emb is None or len(emb) == 0:
            continue
        q = np.array(emb, dtype=np.float32)  # (D,)

        # cos 相似度：已归一化编码可直接点积
        sims = known_matrix @ q  # (N,)

        # 按人员聚合取最大相似度
        per_person_best: Dict[str, Dict[str, Any]] = {}
        for i, sim in enumerate(sims.tolist()):
            pid = person_ids[i]
            name = names[i]
            cur = per_person_best.get(pid)
            if cur is None or sim > cur["similarity"]:
                per_person_best[pid] = {"person_id": pid, "name": name, "similarity": sim}

        if not per_person_best:
            continue
        # 找到相似度最高的人员
        best = max(per_person_best.values(), key=lambda x: x["similarity"])  # type: ignore
        # 第二佳用于参考（可选）
        others = sorted([v for k, v in per_person_best.items() if k != best["person_id"]], key=lambda x: x["similarity"], reverse=True)
        second_sim = others[0]["similarity"] if others else 0.0

        if best["similarity"] >= similarity_threshold:
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            results.append({
                "person_id": best["person_id"],
                "name": best["name"],
                "similarity": float(best["similarity"]),
                "second_best_similarity": float(second_sim),
                "bbox": {"left": x1, "top": y1, "right": x2, "bottom": y2}
            })

    return resp_ok({
        "matches": results,
        "match_count": len(results),
        "faces_detected": len(faces),
        "threshold": similarity_threshold,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)