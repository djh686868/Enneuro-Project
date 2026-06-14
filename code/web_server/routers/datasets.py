import os
import base64
import numpy as np
import cv2
from fastapi import APIRouter, HTTPException, UploadFile, File

from web_server.schemas import DatasetRegisterRequest
from web_server.services.dataset_manager import (
    register_dataset, list_datasets, UPLOAD_DIR, _datasets,
    _detect_and_load_dataset,
)

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.post("/register")
def register(req: DatasetRegisterRequest):
    if not os.path.exists(req.path):
        raise HTTPException(status_code=400, detail=f"Path does not exist: {req.path}")
    img_size = (req.resize_w, req.resize_h) if req.resize_w > 0 and req.resize_h > 0 else None
    dataset_id = register_dataset(req.name, req.path, req.description, img_size=img_size)
    return {"dataset_id": dataset_id}


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    import zipfile
    safe_name = os.path.basename(file.filename or "dataset")
    dest = os.path.join(UPLOAD_DIR, safe_name.replace(".zip", ""))
    os.makedirs(dest, exist_ok=True)
    zip_path = os.path.join(UPLOAD_DIR, safe_name)
    content = await file.read()
    with open(zip_path, "wb") as f:
        f.write(content)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    dataset_id = register_dataset(name=safe_name.replace(".zip", ""), path=dest)
    return {"dataset_id": dataset_id}


@router.get("")
def get_datasets():
    return list_datasets()


@router.get("/{dataset_id}/preview")
def preview_dataset(dataset_id: str, n: int = 9):
    meta = _datasets.get(dataset_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        ds = _detect_and_load_dataset(meta["path"], img_size=meta.get("img_size"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load dataset: {e}")

    classes = getattr(ds, '_classes', None) or getattr(ds, 'classes', [])
    samples = []
    for i in range(min(n, len(ds))):
        img_arr, label = ds[i]
        # img_arr 可能是 (1,H,W) 或 (H,W) 或 (3,H,W)
        arr = np.array(img_arr)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr[0]  # 取第一通道展示
        elif arr.ndim == 2:
            pass
        img_u8 = (arr * 255).clip(0, 255).astype(np.uint8)
        _, buf = cv2.imencode(".png", img_u8)
        b64 = base64.b64encode(buf).decode()
        samples.append({"label": int(label), "image": b64})
    return {"samples": samples, "classes": list(classes), "total": len(ds)}
