from fastapi import APIRouter, HTTPException
from web_server.schemas import GradCAMRequest, FeatureMapsRequest
from web_server.services.explain_service import run_gradcam, run_feature_maps

router = APIRouter(prefix="/api/explain", tags=["explain"])


@router.post("/gradcam")
def gradcam(req: GradCAMRequest):
    try:
        return run_gradcam(
            model_id=req.model_id,
            image_b64=req.image_base64,
            layer_name=req.target_layer_name,
            class_idx=req.class_idx,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/feature_maps")
def feature_maps(req: FeatureMapsRequest):
    try:
        return run_feature_maps(
            model_id=req.model_id,
            image_b64=req.image_base64,
            layer_name=req.target_layer_name,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
