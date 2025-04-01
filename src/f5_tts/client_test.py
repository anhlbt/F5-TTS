import aiohttp
import asyncio
import json


async def main():
    async with aiohttp.ClientSession() as session:
        # Gửi request
        async with session.post(
            "http://localhost:8000/generate",
            json={
                "ref_file": "https://example.com/sample.wav",
                "ref_text": "Reference text",
                "gen_text": "Generated text",
            },
        ) as resp:
            data = await resp.json()
            task_id = data["task_id"]
            print(f"Task queued: {task_id}")

        # Lắng nghe qua WebSocket
        async with session.ws_connect(f"ws://localhost:8000/ws/{task_id}") as ws:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    print(f"Received update: {msg.data}")
                    if json.loads(msg.data)["status"] == "completed":
                        break


async def poll_status(task_id):
    async with aiohttp.ClientSession() as session:
        while True:
            async with session.get(f"http://localhost:8000/status/{task_id}") as resp:
                data = await resp.json()
                print(data)
                if data["status"] == "completed":
                    break
            await asyncio.sleep(2)  # Kiểm tra mỗi 2 giây


# Gọi sau khi gửi /generate
asyncio.run(poll_status("your-task-id"))


asyncio.run(main())
