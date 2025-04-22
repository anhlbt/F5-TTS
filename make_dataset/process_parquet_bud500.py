# import os
# import glob
# import pyarrow.parquet as pq
# import argparse
# import pandas as pd
# from pathlib import Path

# import uuid


# def generate_id(channel, index):
#     return f"{channel}_{index}_{uuid.uuid4().hex[:8]}"


# def load_processed_samples(metadata_file):
#     """Đọc các sample_id đã xử lý từ file metadata"""
#     processed = set()
#     if os.path.exists(metadata_file):
#         try:
#             df = pd.read_csv(
#                 metadata_file, sep="|", names=["sample_id", "text1", "text2"]
#             )
#             processed = set(df["text1"])
#         except Exception as e:
#             print(f"Không thể đọc metadata file: {e}")
#     return processed


# def process_parquet_files(
#     input_dir, output_dir, mode, max_samples=None, batch_size=100
# ):
#     # Đặt tên thư mục và file dựa trên mode
#     if mode == "full":
#         wav_dir = os.path.join(output_dir, "wavs")
#         metadata_file = os.path.join(output_dir, "metadata.csv")
#     else:  # mode == "test"
#         wav_dir = os.path.join(output_dir, "wavs_test")
#         metadata_file = os.path.join(output_dir, "metadata_test.csv")

#     # Tạo thư mục đầu ra nếu chưa tồn tại
#     os.makedirs(wav_dir, exist_ok=True)

#     # Tải danh sách các mẫu đã xử lý
#     processed_samples = load_processed_samples(metadata_file)
#     initial_total = len(processed_samples)
#     total_samples = initial_total

#     # Tìm tất cả file Parquet trong thư mục
#     parquet_files = glob.glob(os.path.join(input_dir, "*"))
#     parquet_files = [f for f in parquet_files if not f.endswith((".json", ".lock"))]

#     # Mở file metadata ở chế độ append
#     with open(metadata_file, "a", encoding="utf-8") as meta_file:
#         batch = []
#         for parquet_file in parquet_files:
#             if max_samples is not None and total_samples >= max_samples:
#                 break

#             print(f"Đang xử lý: {parquet_file}")
#             try:
#                 # Đọc file Parquet
#                 parquet_reader = pq.ParquetFile(parquet_file)
#                 row_groups = parquet_reader.num_row_groups

#                 for rg in range(row_groups):
#                     if max_samples is not None and total_samples >= max_samples:
#                         break
#                     try:
#                         table = parquet_reader.read_row_group(
#                             rg, columns=["audio", "transcription"]
#                         )
#                         df = table.to_pandas()

#                         for idx, row in df.iterrows():
#                             if max_samples is not None and total_samples >= max_samples:
#                                 break

#                             text = row["transcription"]
#                             audio = row["audio"]

#                             # Tạo ID duy nhất
#                             sample_id = generate_id("bud500", total_samples)

#                             # Bỏ qua nếu mẫu đã được xử lý
#                             if text in processed_samples:  # sample_id
#                                 print("by pass: ", text)
#                                 continue

#                             # Đường dẫn file .wav
#                             wav_path = os.path.join(wav_dir, f"{sample_id}.wav")

#                             # Chỉ xử lý nếu file wav chưa tồn tại
#                             if not os.path.exists(wav_path):
#                                 with open(wav_path, "wb") as wav_file:
#                                     wav_file.write(audio["bytes"])

#                             # Thêm vào batch
#                             batch.append(f"{sample_id}|{text}|{text}")
#                             total_samples += 1
#                             processed_samples.add(sample_id)

#                             # Ghi batch khi đủ kích thước
#                             if len(batch) >= batch_size:
#                                 meta_file.write("\n".join(batch) + "\n")
#                                 meta_file.flush()  # Đảm bảo ghi ngay lập tức
#                                 batch = []
#                                 print(f"Đã xử lý {total_samples} mẫu")

#                     except Exception as e:
#                         print(f"Lỗi khi xử lý row group {rg} trong {parquet_file}: {e}")
#                         continue

#             except Exception as e:
#                 print(f"Lỗi khi xử lý {parquet_file}: {e}")
#                 continue

#         # Ghi các mẫu còn lại trong batch
#         if batch:
#             meta_file.write("\n".join(batch) + "\n")

#     print(
#         f"Hoàn tất! Tổng cộng {total_samples} mẫu đã được xử lý (bao gồm {initial_total} mẫu đã xử lý trước đó)."
#     )
#     print(f"- File metadata: {metadata_file}")
#     print(f"- Thư mục WAVs: {wav_dir}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Xử lý file Parquet thành WAV và metadata.csv"
#     )
#     parser.add_argument(
#         "--input_dir",
#         type=str,
#         default="/media/anhlbt/Book1/datasets/viet_bud500/data",
#         help="Đường dẫn thư mục chứa file Parquet",
#     )
#     parser.add_argument(
#         "--mode",
#         type=str,
#         choices=["full", "test"],
#         default="test",
#         help="Chế độ chạy: 'full' để xử lý toàn bộ, 'test' để xử lý một phần nhỏ",
#     )
#     parser.add_argument(
#         "--max_samples",
#         type=int,
#         default=100,
#         help="Số mẫu tối đa để test (chỉ áp dụng với mode=test)",
#     )
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         default="/media/anhlbt/SSD2/bud500",
#         help="Đường dẫn thư mục đầu ra chứa WAVs và metadata",
#     )
#     args = parser.parse_args()

#     process_parquet_files(
#         args.input_dir,
#         args.output_dir,
#         args.mode,
#         args.max_samples if args.mode == "test" else None,
#     )

import os
import glob
import pyarrow.parquet as pq
import argparse
import pandas as pd
from pathlib import Path
import uuid
import numpy as np
from pydub import AudioSegment
import random
import io  # Thêm import này


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
            processed = set(df["sample_id"])
        except Exception as e:
            print(f"Không thể đọc metadata file: {e}")
    return processed


def concatenate_audios(audio_list, output_path):
    """Nối các file audio với khoảng trống ngẫu nhiên"""
    combined = AudioSegment.empty()

    for i, audio_bytes in enumerate(audio_list):
        # Chuyển bytes thành file-like object và tạo AudioSegment
        audio_file = io.BytesIO(audio_bytes["bytes"])
        audio = AudioSegment.from_file(audio_file, format="wav")
        combined += audio

        # Thêm khoảng trống ngẫu nhiên (trừ file cuối cùng)
        if i < len(audio_list) - 1:
            silence_duration = random.uniform(0.3, 0.5) * 1000  # đổi sang milliseconds
            silence = AudioSegment.silent(duration=silence_duration)
            combined += silence

    # Xuất file audio đã nối
    combined.export(output_path, format="wav")
    return combined


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
        audio_buffer = []
        text_buffer = []

        for parquet_file in parquet_files:
            if max_samples is not None and total_samples >= max_samples:
                break

            print(f"Đang xử lý: {parquet_file}")
            try:
                parquet_reader = pq.ParquetFile(parquet_file)
                row_groups = parquet_reader.num_row_groups

                for rg in range(row_groups):
                    if max_samples is not None and total_samples >= max_samples:
                        break

                    table = parquet_reader.read_row_group(
                        rg, columns=["audio", "transcription"]
                    )
                    df = table.to_pandas()

                    for idx, row in df.iterrows():
                        if max_samples is not None and total_samples >= max_samples:
                            break

                        text = row["transcription"]
                        audio = row["audio"]
                        sample_id = generate_id("bud500", total_samples)

                        if sample_id in processed_samples:
                            continue

                        audio_buffer.append(audio)
                        text_buffer.append(text)

                        # Khi đủ số lượng sample để concatenate
                        if len(audio_buffer) >= random.randint(10, 15):
                            # Tạo ID mới cho sample đã nối
                            combined_id = generate_id("bud500", total_samples)
                            wav_path = os.path.join(wav_dir, f"{combined_id}.wav")

                            # Nối audio
                            concatenate_audios(audio_buffer, wav_path)

                            # Nối text bằng dấu phẩy
                            combined_text = ", ".join(text_buffer)

                            # Thêm vào batch
                            batch.append(
                                f"{combined_id}|{combined_text}|{combined_text}"
                            )
                            total_samples += 1
                            processed_samples.add(combined_id)

                            # Reset buffer
                            audio_buffer = []
                            text_buffer = []

                            if len(batch) >= batch_size:
                                meta_file.write("\n".join(batch) + "\n")
                                meta_file.flush()
                                batch = []
                                print(f"Đã xử lý {total_samples} mẫu")

            except Exception as e:
                print(f"Lỗi khi xử lý {parquet_file}: {e}")
                continue

        # Xử lý các mẫu còn lại trong buffer
        if audio_buffer:
            combined_id = generate_id("bud500", total_samples)
            wav_path = os.path.join(wav_dir, f"{combined_id}.wav")
            concatenate_audios(audio_buffer, wav_path)
            combined_text = ", ".join(text_buffer)
            batch.append(f"{combined_id}|{combined_text}|{combined_text}")
            total_samples += 1

        # Ghi batch cuối cùng
        if batch:
            meta_file.write("\n".join(batch) + "\n")

    print(
        f"Hoàn tất! Tổng cộng {total_samples} mẫu đã được xử lý (bao gồm {initial_total} mẫu đã xử lý trước đó)."
    )
    print(f"- File metadata: {metadata_file}")
    print(f"- Thư mục WAVs: {wav_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Xử lý file Parquet thành WAV và metadata.csv"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/media/anhlbt/Book1/datasets/viet_bud500/data",
        help="Đường dẫn thư mục chứa file Parquet",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "test"],
        default="test",
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
        default="/media/anhlbt/SSD2/bud500",
        help="Đường dẫn thư mục đầu ra chứa WAVs và metadata",
    )
    args = parser.parse_args()

    process_parquet_files(
        args.input_dir,
        args.output_dir,
        args.mode,
        args.max_samples if args.mode == "test" else None,
    )
