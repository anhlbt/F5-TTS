import pandas as pd
import os
from pydub import AudioSegment
from os.path import join

folder = "/media/anhlbt/Book1/datasets/datasets_voice/VOICE_TASK/data_BaomoiCrawler/data/anhlbt_tts"

# Đọc file metadata.csv
df = pd.read_csv(
    join(folder, "metadata.csv"),
    sep="|",
    header=None,
    names=["audio_id", "text", "normalized_text"],
)

# Trích xuất speaker_id từ audio_id
df["speaker_id"] = df["audio_id"].apply(lambda x: x.lstrip("_").split("_")[0])


# Hàm tính số từ trong text
def count_words(text):
    return len(str(text).split())


# Hàm nhóm các dòng theo speaker_id để đạt độ dài 40-60 từ
def group_lines(df_speaker):
    grouped_lines = []
    current_group = []
    current_word_count = 0

    for index, row in df_speaker.iterrows():
        words = count_words(row["text"])
        if current_word_count + words > 60 and current_group:
            grouped_lines.append(current_group)
            current_group = [row]
            current_word_count = words
        else:
            current_group.append(row)
            current_word_count += words
            if 40 <= current_word_count <= 60:
                grouped_lines.append(current_group)
                current_group = []
                current_word_count = 0

    if current_group and current_word_count >= 40:
        grouped_lines.append(current_group)

    return grouped_lines


# Tạo dataset mới
new_dataset = []
silence_1s = AudioSegment.silent(duration=1000)  # 1 giây silence

# Xử lý từng speaker_id
for speaker_id in df["speaker_id"].unique():
    df_speaker = df[df["speaker_id"] == speaker_id].copy()
    groups = group_lines(df_speaker)

    for i, group in enumerate(groups):
        # Concat text
        new_text = ". ".join(row["text"] for row in group) + "."

        # Concat audio
        combined_audio = AudioSegment.empty()
        for j, row in enumerate(group):
            audio_path = join(folder, "wavs", f"{row['audio_id']}.wav")
            print(f"Checking audio file: {audio_path}")  # Debug
            if os.path.exists(audio_path):
                try:
                    audio = AudioSegment.from_wav(audio_path)
                    print(f"Loaded {audio_path}, duration: {len(audio)/1000}s")  # Debug
                    combined_audio += audio
                    if j < len(group) - 1:
                        combined_audio += silence_1s
                except Exception as e:
                    print(f"Error loading {audio_path}: {e}")
            else:
                print(f"File not found: {audio_path}")
        combined_audio += silence_1s

        # Tạo ID mới
        new_id = f"{speaker_id}_new_{str(i).zfill(3)}"

        # Lưu file audio mới
        output_audio_path = join(folder, "new_wav", f"{new_id}.wav")
        os.makedirs(join(folder, "new_wav"), exist_ok=True)
        print(
            f"Saving to {output_audio_path}, duration: {len(combined_audio)/1000}s"
        )  # Debug
        combined_audio.export(output_audio_path, format="wav")

        # Thêm vào dataset mới
        new_dataset.append(
            {"audio_id": new_id, "text": new_text, "word_count": count_words(new_text)}
        )

# Tạo file metadata mới
new_df = pd.DataFrame(new_dataset)
new_df.to_csv(
    join(folder, "new_metadata.csv"),
    sep="|",
    index=False,
    header=False,
    columns=["audio_id", "text"],
)

# In thông tin kết quả
print(f"Đã tạo {len(new_dataset)} dòng mới")
for item in new_dataset:
    print(
        f"ID: {item['audio_id']}, Words: {item['word_count']}, Text: {item['text'][:50]}..."
    )
