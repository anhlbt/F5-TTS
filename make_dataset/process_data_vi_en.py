import pandas as pd
import os


def merge_2_metadata():
    # Đường dẫn đến 2 file metadata.csv
    metadata1_path = (
        "/media/anhlbt/SSD2/workspace/VOICE_TASK/F5-TTS/make_dataset/vi_en/metadata.csv"
    )
    metadata2_path = "/media/anhlbt/SSD2/viVoice/metadata.csv"

    # Thư mục đích
    output_dir = "/media/anhlbt/SSD2/workspace/VOICE_TASK/F5-TTS/data/vivoice_plus"
    output_metadata_path = os.path.join(output_dir, "metadata.csv")
    output_wavs_dir = os.path.join(output_dir, "wavs")

    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(output_wavs_dir, exist_ok=True)

    # Đọc 2 file metadata.csv
    df1 = pd.read_csv(
        metadata1_path,
        sep="|",
        header=None,
        names=["file_id", "transcript", "normalized_transcript"],
    )
    df2 = pd.read_csv(
        metadata2_path,
        sep="|",
        header=None,
        names=["file_id", "transcript", "normalized_transcript"],
    )

    # Xử lý trùng lặp file_id
    seen_ids = set(df1["file_id"])
    for idx, row in df2.iterrows():
        orig_file_id = row["file_id"]
        if orig_file_id in seen_ids:
            # Thêm hậu tố _v2 để tránh trùng
            df2.at[idx, "file_id"] = f"{orig_file_id}_v2"
        seen_ids.add(df2.at[idx, "file_id"])

    # Merge 2 DataFrame
    merged_df = pd.concat([df1, df2], ignore_index=True)
    print("shape: ", merged_df.shape)

    # Ghi file metadata.csv merged
    merged_df.to_csv(
        output_metadata_path, sep="|", index=False, header=False, encoding="utf-8"
    )

    print(f"Đã merge metadata thành công: {output_metadata_path}")
    print(f"Tổng số dòng: {len(merged_df)}")
    print("Vui lòng chạy lệnh softlink để liên kết file wav vào:", output_wavs_dir)


def convert_bangdream_to_ljspeech():
    # Đường dẫn đến file esd.list
    input_file = (
        "/media/anhlbt/SSD2/workspace/VOICE_TASK/BangStarlight/data/vi_en/esd.list"
    )
    # Đường dẫn để lưu file metadata.csv
    output_file = "./vi_en/metadata.csv"

    # Đọc file esd.list
    # Dùng '|' làm delimiter vì dữ liệu của bạn dùng ký tự này để phân tách
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Tạo list để lưu dữ liệu
    data = []

    # Xử lý từng dòng
    for line in lines:
        # Tách dòng thành các cột
        columns = line.strip().split("|")
        if len(columns) >= 4:  # Đảm bảo dòng có đủ cột
            # Lấy cột đầu (file path) và cột cuối (text)
            audio_path = columns[0]  # ./data/vi_en/wavs/--rWHNfSqhY_0029.wav
            text = columns[-1]  # text

            # Chỉ lấy tên file từ đường dẫn (bỏ phần ./data/vi_en/wavs/)
            file_name = os.path.basename(audio_path).replace(".wav", "")

            # Thêm vào data theo format của LJSpeech: "file_name|text|text"
            # Trong đó cột thứ 2 và 3 giống nhau (text gốc)
            data.append([file_name, text, text])

    # Tạo DataFrame
    df = pd.DataFrame(data, columns=["file_id", "transcript", "normalized_transcript"])

    # Ghi ra file metadata.csv
    df.to_csv(output_file, sep="|", index=False, header=False, encoding="utf-8")

    print(f"Đã tạo file {output_file} thành công!")
    print(f"Tổng số dòng: {len(df)}")


if __name__ == "__main__":
    merge_2_metadata()
