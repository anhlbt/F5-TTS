from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket
from pydantic import BaseModel
from typing import Optional
from .api import F5TTS
import asyncio
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
import json
import os
import random
import aiofiles
import aiohttp
import shutil
from uuid import uuid4
import redis.asyncio as redis  # Thư viện Redis bất đồng bộ

app = FastAPI(title="F5-TTS API")

# Kafka configuration
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_REQUEST_TOPIC = "tts_requests"
KAFKA_RESULT_TOPIC = "tts_results"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Khởi tạo Redis client
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Initialize TTS model
tts = F5TTS()


class TTSRequest(BaseModel):
    ref_file: str
    ref_text: str
    gen_text: str
    remove_silence: Optional[bool] = False
    seed: Optional[int] = None
    task_id: Optional[str] = None
    output_file: Optional[str] = None


class TTSResponse(BaseModel):
    status: str
    task_id: str
    output_file: str
    seed: int


async def download_file(url: str, dest_path: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(
                    status_code=400, detail=f"Không thể tải file từ {url}"
                )
            async with aiofiles.open(dest_path, "wb") as f:
                await f.write(await response.read())


async def save_uploaded_file(upload_file: UploadFile, dest_path: str):
    async with aiofiles.open(dest_path, "wb") as f:
        content = await upload_file.read()
        await f.write(content)


async def prepare_ref_file(ref_file: str, uploaded_file: Optional[UploadFile] = None):
    ref_file_local = os.path.join(OUTPUT_DIR, f"ref_{random.randint(0, 1000000)}.wav")
    if ref_file.startswith("http://") or ref_file.startswith("https://"):
        await download_file(ref_file, ref_file_local)
    elif uploaded_file:
        await save_uploaded_file(uploaded_file, ref_file_local)
    elif os.path.exists(ref_file):
        shutil.copy(ref_file, ref_file_local)
    else:
        raise HTTPException(
            status_code=400, detail=f"ref_file không hợp lệ: {ref_file}"
        )
    return ref_file_local


async def produce_message(request: TTSRequest):
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BROKER)
    await producer.start()
    try:
        message = request.dict()
        await producer.send_and_wait(
            KAFKA_REQUEST_TOPIC, json.dumps(message).encode("utf-8")
        )
    finally:
        await producer.stop()


@app.post("/generate", response_model=TTSResponse)
async def generate_tts(
    ref_text: str,
    gen_text: str,
    ref_file: Optional[str] = None,
    uploaded_file: Optional[UploadFile] = File(None),
    remove_silence: Optional[bool] = False,
    seed: Optional[int] = None,
):
    try:
        task_id = str(uuid4())
        output_file = os.path.join(OUTPUT_DIR, f"output_{task_id}.wav")
        ref_file_local = await prepare_ref_file(ref_file, uploaded_file)

        request = TTSRequest(
            ref_file=ref_file_local,
            ref_text=ref_text,
            gen_text=gen_text,
            remove_silence=remove_silence,
            seed=seed,
            task_id=task_id,
            output_file=output_file,
        )

        await produce_message(request)

        # Lưu trạng thái vào Redis
        await redis_client.set(
            f"task:{task_id}",
            json.dumps({"status": "queued", "output_file": output_file}),
            ex=3600,  # Hết hạn sau 1 giờ
        )

        return TTSResponse(
            status="queued",
            task_id=task_id,
            output_file=output_file,
            seed=tts.seed if request.seed is None else request.seed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await websocket.accept()
    consumer = AIOKafkaConsumer(
        KAFKA_RESULT_TOPIC, bootstrap_servers=KAFKA_BROKER, group_id="api-group"
    )
    await consumer.start()
    try:
        # Lấy trạng thái từ Redis
        status = await redis_client.get(f"task:{task_id}")
        if status:
            await websocket.send_json(json.loads(status))
        else:
            await websocket.send_json({"status": "not_found", "task_id": task_id})

        async for msg in consumer:
            result = json.loads(msg.value.decode("utf-8"))
            if result["task_id"] == task_id:
                await redis_client.set(f"task:{task_id}", json.dumps(result), ex=3600)
                await websocket.send_json(result)
                break
    finally:
        await consumer.stop()
        await websocket.close()


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    status = await redis_client.get(f"task:{task_id}")
    if status:
        return json.loads(status)
    raise HTTPException(status_code=404, detail="Task not found")
