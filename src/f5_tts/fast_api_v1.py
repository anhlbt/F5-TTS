import random
import sys
from importlib.resources import files
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
import os
import soundfile as sf
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
import logging

from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    transcribe,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)
from f5_tts.model.utils import seed_everything

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="F5-TTS API")

# Cấu hình load balancing


NUM_INSTANCES = min(4, max(1, os.cpu_count() // 2))
MAX_CONCURRENT_REQUESTS = 10  # Giới hạn request đồng thời


class F5TTS:
    def __init__(
        self,
        model="F5TTS_Base",
        ckpt_file="/workspace/F5-TTS/ckpts/vivoice/model_last.pt",
        vocab_file="/workspace/F5-TTS/ckpts/vivoice/vocab.txt",
        ode_method="euler",
        use_ema=True,
        vocoder_local_path=None,
        device=None,
        hf_cache_dir=None,
    ):
        from omegaconf import OmegaConf
        from hydra.utils import get_class

        model_cfg = OmegaConf.load(
            str(files("f5_tts").joinpath(f"configs/{model}.yaml"))
        )
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        model_arc = model_cfg.model.arch

        self.mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
        self.target_sample_rate = model_cfg.model.mel_spec.target_sample_rate
        self.seed = -1
        self.ode_method = ode_method
        self.use_ema = use_ema

        if device is not None:
            self.device = device
        else:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.vocoder = load_vocoder(
            self.mel_spec_type,
            vocoder_local_path is not None,
            vocoder_local_path,
            self.device,
            hf_cache_dir,
        )

        if not ckpt_file:
            repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"
            if model == "F5TTS_Base":
                if self.mel_spec_type == "vocos":
                    ckpt_step = 1200000
                elif self.mel_spec_type == "bigvgan":
                    model = "F5TTS_Base_bigvgan"
                    ckpt_type = "pt"
            elif model == "E2TTS_Base":
                repo_name = "E2-TTS"
                ckpt_step = 1200000

            ckpt_file = str(
                cached_path(
                    f"hf://SWivid/{repo_name}/{model}/model_{ckpt_step}.{ckpt_type}",
                    cache_dir=hf_cache_dir,
                )
            )
        self.ema_model = load_model(
            model_cls,
            model_arc,
            ckpt_file,
            self.mel_spec_type,
            vocab_file,
            self.ode_method,
            self.use_ema,
            self.device,
        )

    def infer(
        self,
        ref_file: str,
        ref_text: str,
        gen_text: str,
        target_rms: float = 0.1,
        cross_fade_duration: float = 0.15,
        sway_sampling_coef: float = -1,
        cfg_strength: float = 2,
        nfe_step: int = 32,
        speed: float = 1.0,
        fix_duration: Optional[float] = None,
        remove_silence: bool = False,
        seed: Optional[int] = None,
    ):
        if seed is None:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)
        self.seed = seed

        ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text)

        wav, sr, spec = infer_process(
            ref_file,
            ref_text,
            gen_text,
            self.ema_model,
            self.vocoder,
            self.mel_spec_type,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=self.device,
        )
        if remove_silence:
            wav = remove_silence_for_generated_wav(wav, sr)

        return wav, sr, spec, seed


# Khởi tạo các instance của F5TTS
f5tts_instances: List[F5TTS] = [F5TTS() for _ in range(NUM_INSTANCES)]
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)  # Giới hạn request đồng thời


# Model cho request body
class TTSRequest(BaseModel):
    ref_file: str
    ref_text: str
    gen_text: str
    target_rms: float = 0.1
    cross_fade_duration: float = 0.15
    sway_sampling_coef: float = -1
    cfg_strength: float = 2
    nfe_step: int = 32
    speed: float = 1.0
    fix_duration: Optional[float] = None
    remove_silence: bool = False
    seed: Optional[int] = None


# Hàm xử lý inference trong thread pool
async def process_tts(request: TTSRequest, instance_idx: int) -> dict:
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        try:
            instance = f5tts_instances[instance_idx]
            wav, sr, spec, seed = await loop.run_in_executor(
                pool,
                instance.infer,
                request.ref_file,
                request.ref_text,
                request.gen_text,
                request.target_rms,
                request.cross_fade_duration,
                request.sway_sampling_coef,
                request.cfg_strength,
                request.nfe_step,
                request.speed,
                request.fix_duration,
                request.remove_silence,
                request.seed,
            )
            # Chuyển wav thành bytes để trả về
            import io

            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, wav, sr, format="WAV")
            wav_bytes = wav_buffer.getvalue()
            wav_buffer.close()

            return {
                "status": "success",
                "sample_rate": sr,
                "seed": seed,
                "wav": wav_bytes.hex(),  # Trả về dưới dạng hex để dễ truyền qua JSON
            }
        except Exception as e:
            logger.error(f"Error in instance {instance_idx}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


# API endpoint
@app.post("/tts-infer")
async def tts_infer(request: TTSRequest):
    async with semaphore:  # Giới hạn số lượng request đồng thời
        # Load balancing: chọn instance ngẫu nhiên
        instance_idx = random.randint(0, NUM_INSTANCES - 1)
        logger.info(f"Processing request with instance {instance_idx}")
        result = await process_tts(request, instance_idx)
        return result


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "instances": NUM_INSTANCES}


if __name__ == "__main__":
    # Chạy server với uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7867, workers=2)
