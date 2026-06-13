import os
import json
from fastapi import APIRouter, HTTPException
from web_server.schemas import BuildModelRequest, BuildModelResponse
from web_server.services.model_registry import (
    build_model_from_config, list_models, list_presets, get_model,
    _model_store, _model_configs,
)

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../../saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

router = APIRouter(prefix="/api/models", tags=["models"])


# ── 固定路径（必须在 /{model_id} 之前注册）───────────────────────────────────

@router.get("/presets")
def get_presets():
    return list_presets()


@router.get("")
def get_models():
    return list_models()


@router.post("/build", response_model=BuildModelResponse)
def build_model(req: BuildModelRequest):
    config = req.model_dump(exclude_none=True)
    if "layers" in config:
        config["layers"] = [
            {"type": l["type"], "params": l["params"]}
            for l in config["layers"]
        ]
    try:
        model_id, model = build_model_from_config(config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return BuildModelResponse(model_id=model_id, model_type=type(model).__name__)


@router.get("/saved/list")
def list_saved_models():
    files = [f for f in os.listdir(SAVED_MODELS_DIR) if f.endswith(".json")]
    result = []
    for f in sorted(files):
        try:
            with open(os.path.join(SAVED_MODELS_DIR, f), encoding="utf-8") as fp:
                meta = json.load(fp)
            result.append({
                "file":       f,
                "model_type": meta.get("model_type", "?"),
                "model_id":   meta.get("model_id", "?"),
            })
        except Exception:
            result.append({"file": f, "model_type": "?", "model_id": "?"})
    return result


@router.post("/saved/load")
def load_saved_model(body: dict):
    filename = body.get("file", "")
    if not filename:
        raise HTTPException(status_code=400, detail="file is required")
    path = os.path.join(SAVED_MODELS_DIR, os.path.basename(filename))
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    config = data.get("model_config", {})
    if not config:
        raise HTTPException(status_code=400, detail="No model_config in file")
    try:
        model_id, model = build_model_from_config(config)
        model.from_dict(data["model_state"])
        _model_store[model_id]   = model
        _model_configs[model_id] = config
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "model_id": model_id, "model_type": type(model).__name__}


# ── 参数路径（/{model_id} 必须在固定路径之后）────────────────────────────────

@router.get("/{model_id}")
def get_model_info(model_id: str):
    try:
        model = get_model(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"model_id": model_id, "type": type(model).__name__}


@router.get("/{model_id}/architecture")
def model_architecture(model_id: str):
    try:
        model = get_model(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"model_id": model_id, "layers": _collect_layers(model)}


def _collect_layers(layer, prefix=""):
    from eneuro.nn.module import Layer as _Layer, Parameter as _Param
    result = []
    for name in sorted(getattr(layer, "_params", set())):
        obj = layer.__dict__.get(name)
        if obj is None or isinstance(obj, _Param):
            continue
        if isinstance(obj, _Layer):
            full = f"{prefix}.{name}" if prefix else name
            result.append({
                "attr": name,
                "full": full,
                "type": type(obj).__name__,
                "children": _collect_layers(obj, full),
            })
    return result


@router.post("/{model_id}/save")
def save_model(model_id: str, body: dict = {}):
    if model_id not in _model_store:
        raise HTTPException(status_code=404, detail="Model not found")
    model  = _model_store[model_id]
    config = _model_configs.get(model_id, {})
    name   = (body.get("name") or model_id).strip()
    if not name.endswith(".json"):
        name += ".json"
    path = os.path.join(SAVED_MODELS_DIR, name)
    data = {
        "model_id":     model_id,
        "model_type":   type(model).__name__,
        "model_config": config,
        "model_state":  model.to_dict(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return {"ok": True, "file": name, "path": path}


@router.delete("/{model_id}")
def delete_model(model_id: str):
    if model_id not in _model_store:
        raise HTTPException(status_code=404, detail="Model not found")
    _model_store.pop(model_id, None)
    _model_configs.pop(model_id, None)
    return {"ok": True, "model_id": model_id}
