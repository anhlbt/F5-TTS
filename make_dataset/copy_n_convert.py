import os
import librosa
import soundfile as sf
from multiprocessing import Pool
from pathlib import Path


def convert_audio_file(args):
    """Hàm xử lý từng file audio"""
    input_path, output_path, target_sr = args
    try:
        # Đọc file audio
        audio, sr = librosa.load(input_path, sr=None)
        # Resample về target sampling rate
        audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        # Ghi file mới
        sf.write(output_path, audio_resampled, target_sr)
        print(f"Đã xử lý: {input_path} -> {output_path}")
    except Exception as e:
        print(f"Lỗi khi xử lý {input_path}: {str(e)}")


def process_wavs_parallel(input_dir="wavs", output_dir="output_wavs", target_sr=24000):
    """
    Hàm chính để xử lý song song tất cả file WAV
    Args:
        input_dir: thư mục chứa file WAV đầu vào
        output_dir: thư mục đầu ra
        target_sr: sampling rate mục tiêu (mặc định 24000 Hz)
    """
    # Tạo thư mục output nếu chưa tồn tại
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Tìm tất cả file WAV trong thư mục input
    wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".wav")]
    if not wav_files:
        print("Không tìm thấy file WAV nào trong thư mục!")
        return

    # Tạo danh sách tác vụ
    tasks = []
    for wav_file in wav_files:
        input_path = os.path.join(input_dir, wav_file)
        output_path = os.path.join(output_dir, wav_file)
        tasks.append((input_path, output_path, target_sr))

    # Sử dụng Pool để xử lý song song
    num_processes = min(
        os.cpu_count(), len(wav_files)
    )  # Số process tối đa bằng số CPU hoặc số file
    with Pool(processes=num_processes) as pool:
        pool.map(convert_audio_file, tasks)

    print("Hoàn tất xử lý tất cả file!")


# Cách sử dụng
if __name__ == "__main__":
    # Ví dụ chạy hàm
    process_wavs_parallel(
        input_dir="/media/anhlbt/Book/train_bert/wavs",
        output_dir="/media/anhlbt/SSD2/viVoice/wavs",
        target_sr=24000,
    )
