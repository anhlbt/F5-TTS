#!/bin/bash
set -e  # Exit on any error

# Khởi động infer_gradio trong background
echo "Starting infer_gradio..."
python src/f5_tts/infer/infer_gradio.py #&

# # Khởi động FastAPI với Uvicorn trong background
# echo "Starting FastAPI server..."
# uvicorn src.f5_tts.api:app --host 0.0.0.0 --port 7867 --workers 1 --reload &

# # Chờ tất cả các process background hoàn thành
# wait

echo "All services are running."