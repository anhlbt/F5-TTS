import os
import glob
import pyarrow.parquet as pq
import argparse
import pandas as pd
from pathlib import Path

import uuid


def generate_id(channel, index):
    return f"{channel}_{index}_{uuid.uuid4().hex[:8]}"


def load_processed_samples(metadata_file):
    """Đọc các sample_id đã xử lý từ file metadata"""
    processed = set()
    if os.path.exists(metadata_file):
        try:
            df = pd.read_csv(
                metadata_file, sep="|", names=["sample_id", "text1", "text2"]
            )
            processed = set(df["text1"])
        except Exception as e:
            print(f"Không thể đọc metadata file: {e}")
    return processed


def remove_duplicates(metadata_file):
    """
    Loại bỏ các record trùng lặp trong file metadata.csv và lưu lại file.

    :param metadata_file: Đường dẫn đến file metadata.csv
    """
    # Đọc file metadata.csv
    df = pd.read_csv(metadata_file, sep="|", names=["sample_id", "text1", "text2"])

    # Loại bỏ các record trùng lặp dựa trên sample_id
    df_unique = df.drop_duplicates(subset=["text1"])

    # Lưu lại file metadata.csv
    df_unique.to_csv(metadata_file, sep="|", index=False, header=False)
    print(df_unique.head())
    print("\n")
    print(df_unique.tail())
    print(f"Đã loại bỏ các record trùng lặp trong {metadata_file}")


def process_parquet_files(
    input_dir, output_dir, mode, max_samples=None, batch_size=100
):
    # Đặt tên thư mục và file dựa trên mode
    if mode == "full":
        wav_dir = os.path.join(output_dir, "wavs")
        metadata_file = os.path.join(output_dir, "metadata.csv")
    else:  # mode == "test"
        wav_dir = os.path.join(output_dir, "wavs_test")
        metadata_file = os.path.join(output_dir, "metadata_test.csv")

    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(wav_dir, exist_ok=True)

    # Tải danh sách các mẫu đã xử lý
    processed_samples = load_processed_samples(metadata_file)
    initial_total = len(processed_samples)
    total_samples = initial_total

    # Tìm tất cả file Parquet trong thư mục
    parquet_files = glob.glob(os.path.join(input_dir, "*"))
    parquet_files = [f for f in parquet_files if not f.endswith((".json", ".lock"))]

    # Mở file metadata ở chế độ append
    with open(metadata_file, "a", encoding="utf-8") as meta_file:
        batch = []
        for parquet_file in parquet_files:
            if max_samples is not None and total_samples >= max_samples:
                break

            print(f"Đang xử lý: {parquet_file}")
            try:
                # Đọc file Parquet
                parquet_reader = pq.ParquetFile(parquet_file)
                row_groups = parquet_reader.num_row_groups

                for rg in range(row_groups):
                    if max_samples is not None and total_samples >= max_samples:
                        break
                    try:
                        table = parquet_reader.read_row_group(
                            rg, columns=["channel", "text", "audio"]
                        )
                        df = table.to_pandas()

                        for idx, row in df.iterrows():
                            if max_samples is not None and total_samples >= max_samples:
                                break

                            channel = row["channel"]
                            text = row["text"]
                            audio = row["audio"]

                            # Tạo ID duy nhất
                            sample_id = generate_id(channel, total_samples)

                            # Bỏ qua nếu mẫu đã được xử lý
                            if text in processed_samples:  # sample_id
                                print("by pass: ", text)
                                continue

                            # Đường dẫn file .wav
                            wav_path = os.path.join(wav_dir, f"{sample_id}.wav")

                            # Chỉ xử lý nếu file wav chưa tồn tại
                            if not os.path.exists(wav_path):
                                with open(wav_path, "wb") as wav_file:
                                    wav_file.write(audio["bytes"])

                            # Thêm vào batch
                            batch.append(f"{sample_id}|{text}|{text}")
                            total_samples += 1
                            processed_samples.add(sample_id)

                            # Ghi batch khi đủ kích thước
                            if len(batch) >= batch_size:
                                meta_file.write("\n".join(batch) + "\n")
                                meta_file.flush()  # Đảm bảo ghi ngay lập tức
                                batch = []
                                print(f"Đã xử lý {total_samples} mẫu")

                    except Exception as e:
                        print(f"Lỗi khi xử lý row group {rg} trong {parquet_file}: {e}")
                        continue

            except Exception as e:
                print(f"Lỗi khi xử lý {parquet_file}: {e}")
                continue

        # Ghi các mẫu còn lại trong batch
        if batch:
            meta_file.write("\n".join(batch) + "\n")

    print(
        f"Hoàn tất! Tổng cộng {total_samples} mẫu đã được xử lý (bao gồm {initial_total} mẫu đã xử lý trước đó)."
    )
    print(f"- File metadata: {metadata_file}")
    print(f"- Thư mục WAVs: {wav_dir}")


def delete_unreferenced_wav_files(metadata_file, wav_dir):
    """
    Xóa các file WAV không có sample_id tương ứng trong file metadata.csv.

    :param metadata_file: Đường dẫn đến file metadata.csv
    :param wav_dir: Đường dẫn đến thư mục chứa các file WAV
    """
    # Đọc file metadata.csv và lấy danh sách sample_id
    if not os.path.exists(metadata_file):
        print(f"File metadata.csv không tồn tại: {metadata_file}")
        return

    df = pd.read_csv(metadata_file, sep="|", names=["sample_id", "text1", "text2"])
    valid_sample_ids = set(df["sample_id"].unique())

    # Liệt kê các file WAV trong thư mục
    wav_files = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]

    # Kiểm tra và xóa các file không có trong metadata
    for wav_file in wav_files:
        sample_id = os.path.splitext(wav_file)[0]  # Lấy sample_id từ tên file
        if sample_id not in valid_sample_ids:
            wav_path = os.path.join(wav_dir, f"{sample_id}.wav")
            try:
                os.remove(wav_path)
                print(f"Đã xóa: {wav_path}")
            except Exception as e:
                print(f"Lỗi khi xóa {wav_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Xử lý file Parquet thành WAV và metadata.csv"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/media/anhlbt/Book/datasets/viVoice/downloads",
        help="Đường dẫn thư mục chứa file Parquet",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "test"],
        default="full",
        help="Chế độ chạy: 'full' để xử lý toàn bộ, 'test' để xử lý một phần nhỏ",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Số mẫu tối đa để test (chỉ áp dụng với mode=test)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/media/anhlbt/SSD2/viVoice",
        help="Đường dẫn thư mục đầu ra chứa WAVs và metadata",
    )
    args = parser.parse_args()

    process_parquet_files(
        args.input_dir,
        args.output_dir,
        args.mode,
        args.max_samples if args.mode == "test" else None,
    )

    # metadata_file = os.path.join(args.output_dir, "metadata.csv")
    # wav_dir = os.path.join(args.output_dir, "wavs")
    # # remove_duplicates(metadata_file)

    # delete_unreferenced_wav_files(metadata_file, wav_dir)
