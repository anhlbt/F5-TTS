import pandas as pd
import random
import os

# Đường dẫn đến file metadata.csv (thay đổi theo vị trí file của bạn)
input_file = (
    "/media/anhlbt/SSD2/viVoice/metadata.csv"  # Ví dụ: "/path/to/LJSpeech/metadata.csv"
)
output_file = "/media/anhlbt/SSD2/viVoice/metadata_sample.csv"

# Đọc file metadata.csv
# LJSpeech metadata thường có định dạng: id|transcription|normalized_transcription
# Không có header, dùng dấu | làm delimiter
data = pd.read_csv(
    input_file,
    sep="|",
    header=None,
    names=["id", "transcription", "normalized_transcription"],
)

# Tính số lượng dòng cần lấy (10% tổng số dòng)
total_rows = len(data)
sample_size = int(total_rows * 0.1)

# Lấy ngẫu nhiên 10% dữ liệu
sample_data = data.sample(
    n=sample_size, random_state=42
)  # random_state để kết quả có thể tái hiện

# Lưu lại thành file metadata_sample.csv với cùng định dạng
sample_data.to_csv(output_file, sep="|", header=False, index=False)

# In thông tin để kiểm tra
print(f"Tổng số dòng trong file gốc: {total_rows}")
print(f"Số dòng trong file mẫu: {sample_size}")
print(f"Đã lưu file mẫu tại: {os.path.abspath(output_file)}")
