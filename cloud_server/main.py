import base64
import io
import os

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

from src.models import load_model
from src.utils import InputPadder
from src.utils.encode_transfer import pack_enc_out


class EncodeRequest(BaseModel):
    frame0: str
    frame1: str


class EncodeResponse(BaseModel):
    blob: str
    difference: float
    height: int
    width: int


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_encoder(config: dict, device: torch.device) -> torch.nn.Module:
    model_name = config["model_name"]
    model_args = config["model_args"]
    ckpt_path = config["pretrained_eden_path"]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = load_model(model_name, **model_args)
    model.load_state_dict(ckpt["eden"])
    model.to(device)
    model.eval()
    return model


def decode_image_b64(b64_str: str) -> torch.Tensor:
    try:
        raw = base64.b64decode(b64_str)
        with Image.open(io.BytesIO(raw)) as img:
            img = img.convert("RGB")
            np_img = np.array(img, dtype=np.uint8)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {exc}") from exc
    tensor = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor


CONFIG_PATH = os.environ.get("EDEN_CONFIG", "configs/eval_eden.yaml")
DEVICE = torch.device(os.environ.get("EDEN_ENCODER_DEVICE", "cuda:0"))
CONFIG = load_config(CONFIG_PATH)
COS_SIM_MEAN = CONFIG.get("cos_sim_mean", 0.0)
COS_SIM_STD = CONFIG.get("cos_sim_std", 1.0)
EDEN_ENCODER = load_encoder(CONFIG, DEVICE)

app = FastAPI(title="EDEN Encoder Service", version="1.0")


@app.post("/encode", response_model=EncodeResponse)
@torch.inference_mode()
def encode_frames(req: EncodeRequest):
    frame0 = decode_image_b64(req.frame0).to(DEVICE)
    frame1 = decode_image_b64(req.frame1).to(DEVICE)

    height, width = frame0.shape[2:]
    padder = InputPadder([height, width])
    cond_frames = padder.pad(torch.cat((frame0, frame1), dim=0))

    cos_sim = torch.mean(torch.cosine_similarity(frame0, frame1), dim=[1, 2])
    difference = ((cos_sim - COS_SIM_MEAN) / COS_SIM_STD).unsqueeze(1)

    enc_out = EDEN_ENCODER.encode(cond_frames)
    blob_bytes = pack_enc_out(enc_out)
    blob_b64 = base64.b64encode(blob_bytes).decode("ascii")

    return EncodeResponse(
        blob=blob_b64,
        difference=float(difference.item()),
        height=height,
        width=width,
    )

