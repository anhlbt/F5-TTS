import pandas as pd
from os.path import join

# Đường dẫn đến thư mục chứa metadata.csv
folder = "/media/anhlbt/Book1/datasets/datasets_voice/VOICE_TASK/data_BaomoiCrawler/data/anhlbt_tts"

# Đọc file metadata.csv
df = pd.read_csv(
    join(folder, "metadata_updated.csv"),
    sep="|",
    header=None,
    names=["audio_id", "text", "normalized_text"],
)


# Hàm thêm đuôi .wav nếu chưa có
def add_wav_extension(audio_id):
    if audio_id.endswith(".wav"):
        return f"{audio_id[:-4]}"
    return audio_id


# Áp dụng hàm để thêm đuôi .wav vào cột audio_id
df["audio_id"] = df["audio_id"].apply(add_wav_extension)

# Lưu lại file metadata mới
output_file = join(folder, "metadata.csv")
df.to_csv(
    output_file,
    sep="|",
    index=False,
    header=False,
    columns=["audio_id", "text", "normalized_text"],
)

# In thông tin để kiểm tra
print(f"Đã cập nhật metadata và lưu vào {output_file}")
print("5 dòng đầu tiên của file mới:")
print(df.head())
