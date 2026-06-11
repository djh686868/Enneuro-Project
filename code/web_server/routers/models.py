from fastapi import APIRouter, HTTPException
from web_server.schemas import BuildModelRequest, BuildModelResponse
from web_server.services.model_registry import (
    build_model_from_config, list_models, list_presets, get_model,
)

router = APIRouter(prefix="/api/models", tags=["models"])


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


@router.get("/{model_id}")
def get_model_info(model_id: str):
    try:
        model = get_model(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"model_id": model_id, "type": type(model).__name__}
