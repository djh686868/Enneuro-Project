from pydantic import BaseModel
from typing import Any, Optional


class LayerConfig(BaseModel):
    type: str
    params: dict[str, Any] = {}


class BuildModelRequest(BaseModel):
    layers: Optional[list[LayerConfig]] = None
    preset: Optional[str] = None


class BuildModelResponse(BaseModel):
    model_id: str
    model_type: str


class DatasetRegisterRequest(BaseModel):
    path: str
    name: str
    description: str = ""


class TrainingStartRequest(BaseModel):
    model_id: str
    dataset_id: str
    epochs: int = 10
    batch_size: int = 32
    lr: float = 0.001
    optimizer: str = "adam"
    device: str = "cpu"
    val_split: float = 0.2


class GradCAMRequest(BaseModel):
    model_id: str
    image_base64: str
    target_layer_name: str
    class_idx: Optional[int] = None


class FeatureMapsRequest(BaseModel):
    model_id: str
    image_base64: str
    target_layer_name: str
