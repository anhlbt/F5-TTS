import asyncio
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
import json
import os
from .api import F5TTS

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_REQUEST_TOPIC = "tts_requests"
KAFKA_RESULT_TOPIC = "tts_results"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


async def process_tts():
    tts = F5TTS()
    consumer = AIOKafkaConsumer(
        KAFKA_REQUEST_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        group_id="tts-workers",
    )
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BROKER)

    await consumer.start()
    await producer.start()

    try:
        async for msg in consumer:
            request = json.loads(msg.value.decode("utf-8"))
            output_file = request.get("output_file")
            ref_file = request["ref_file"]

            # Xử lý TTS request
            wav, sr, spec = tts.infer(
                ref_file=ref_file,
                ref_text=request["ref_text"],
                gen_text=request["gen_text"],
                remove_silence=request.get("remove_silence", False),
                file_wave=output_file,
                seed=request.get("seed"),
            )

            # Gửi kết quả về topic tts_results
            result = {
                "task_id": request.get("task_id"),
                "status": "completed",
                "output_file": output_file,
                "seed": tts.seed,
            }
            await producer.send_and_wait(
                KAFKA_RESULT_TOPIC, json.dumps(result).encode("utf-8")
            )
            print(f"Processed task {result['task_id']} with seed: {tts.seed}")

            # Xóa file ref_file tạm
            if os.path.exists(ref_file):
                os.remove(ref_file)
                print(f"Deleted temporary file: {ref_file}")

    finally:
        await consumer.stop()
        await producer.stop()


if __name__ == "__main__":
    asyncio.run(process_tts())
