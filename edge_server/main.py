import base64
import io
import os

import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

from src.models import load_model
from src.transport import Sampler, create_transport
from src.utils import InputPadder
from src.utils.encode_transfer import unpack_enc_out


class InterpolateRequest(BaseModel):
    blob: str
    difference: float
    height: int
    width: int


class InterpolateResponse(BaseModel):
    frame: str


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_ditdec(config: dict, device: torch.device) -> torch.nn.Module:
    model_name = config["model_name"]
    model_args = config["model_args"]
    ckpt_path = config["pretrained_eden_path"]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = load_model(model_name, **model_args)
    model.load_state_dict(ckpt["eden"])
    model.to(device)
    model.eval()
    return model


def tensor_to_b64_png(tensor: torch.Tensor) -> str:
    if tensor.dim() != 4 or tensor.shape[0] != 1:
        raise ValueError("tensor must be [1, 3, H, W]")
    array = (tensor.clamp(0.0, 1.0)[0] * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(array)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


CONFIG_PATH = os.environ.get("EDEN_CONFIG", "configs/eval_eden.yaml")
DEVICE = torch.device(os.environ.get("EDEN_DITDEC_DEVICE", "cuda:1"))
CONFIG = load_config(CONFIG_PATH)
EDEN_DITDEC = load_ditdec(CONFIG, DEVICE)

TRANSPORT = create_transport("Linear", "velocity")
SAMPLER = Sampler(TRANSPORT)
SAMPLE_FN = SAMPLER.sample_ode(sampling_method="euler", num_steps=2, atol=1e-6, rtol=1e-3)

LATENT_DIM = CONFIG["model_args"]["latent_dim"]
PATCH_SIZE = CONFIG["model_args"]["patch_size"]
VAE_SCALER = CONFIG.get("vae_scaler", 1.0)
VAE_SHIFT = CONFIG.get("vae_shift", 0.0)

app = FastAPI(title="EDEN DiT+Decoder Service", version="1.0")


@app.post("/interpolate", response_model=InterpolateResponse)
@torch.inference_mode()
def interpolate(req: InterpolateRequest):
    try:
        blob = base64.b64decode(req.blob)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Failed to decode blob: {exc}") from exc

    enc_out = unpack_enc_out(blob, DEVICE)
    difference = torch.tensor([[req.difference]], device=DEVICE, dtype=torch.float32)

    ph = enc_out["ph"]
    pw = enc_out["pw"]
    new_h = ph * PATCH_SIZE
    new_w = pw * PATCH_SIZE
    noise = torch.randn([1, new_h // 32 * new_w // 32, LATENT_DIM], device=DEVICE)

    def denoise_wrapper(query_latents, t):
        timestep = t
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([t], device=DEVICE, dtype=torch.float32)
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        return EDEN_DITDEC.denoise_from_tokens(query_latents, timestep, enc_out, difference)

    samples = SAMPLE_FN(noise, denoise_wrapper)[-1]
    denoise_latents = samples / VAE_SCALER + VAE_SHIFT
    generated = EDEN_DITDEC.decode(denoise_latents)

    padder = InputPadder([req.height, req.width])
    generated = padder.unpad(generated).clamp(0.0, 1.0)
    frame_b64 = tensor_to_b64_png(generated.cpu())
    return InterpolateResponse(frame=frame_b64)

