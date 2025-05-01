# A unified script for inference process
# Make adjustments inside functions, and consider both gradio and cli scripts if need to change func output format
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from underthesea import word_tokenize

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # for MPS device compatibility
sys.path.append(
    f"{os.path.dirname(os.path.abspath(__file__))}/../../third_party/BigVGAN/"
)
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../third_party")
from vinorm import TTSnorm
import hashlib
import re
import tempfile
from importlib.resources import files

import matplotlib

matplotlib.use("Agg")

import matplotlib.pylab as plt
import numpy as np
import torch
import torchaudio
import tqdm
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, silence
from transformers import pipeline
from vocos import Vocos
import re

# from vfastpunct import VFastPunct


from f5_tts.model import CFM
from f5_tts.model.utils import (
    get_tokenizer,
    convert_char_to_pinyin,
)

_ref_audio_cache = {}

device = (
    "cuda"
    if torch.cuda.is_available()
    else (
        "xpu"
        if torch.xpu.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
)

# -----------------------------------------

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"
target_rms = 0.1
cross_fade_duration = 0.15
ode_method = "euler"
nfe_step = 32  # 16, 32
cfg_strength = 2.0
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None

# -----------------------------------------
break_words = [
    "không chỉ thế mà còn",
    "không chỉ vậy mà còn",
    "trong khi đó",
    "chẳng hạn như",
    "ví dụ như",
    "nói cách khác",
    "cụ thể là",
    "không chỉ",
    "không những",
    "mà còn",
    "mặt khác",
    "ngoài ra",
    "hơn nữa",
    "trái lại",
    "ngược lại",
    "trong khi",
    "dẫu cho",
    "dù cho",
    "mặc cho",
    "mặc kệ",
    "miễn là",
    "miễn sao",
    "chỉ cần",
    "đối với",
    "như thể",
    "theo đó",
    "theo như",
    "cùng lúc đó",
    "cùng lúc",
    "thậm chí",
    "để cho",
    "để",
    "vì vậy",
    "bởi vì",
    "vì thế",
    "do đó",
    "do vậy",
    "nếu như",
    "hay",
    "vì",
    "nên",
    "và",
    "nhưng",
    "rồi",
    "lại",
    "cũng",
    "vẫn",
    "như",
    "nếu",
    "khi",
    "lúc",
    "bởi",
    "do",
]


def post_process(text):
    replacements = {" . ": ". ", " .. ": ". ", " , ": ", ", " ! ": "! ", " ? ": "? "}
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Thay thế một hoặc nhiều khoảng trắng bằng một khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()
    return text


# chunk text into smaller pieces
def chunk_text(text, max_chars=200):
    print("max_chars: ", max_chars)
    # Define length limits
    threshold_2 = int(max_chars * 1.2)  # 120% của max_chars - độ dài tối đa cho chunk
    min_chunk_length = int(max_chars * 0.3)  # Ngưỡng tối thiểu để xem xét chunk
    final_punctuation = ".?!"  # Dấu câu ưu tiên để kết thúc chunk
    english_punctuation = ".?!,:;)}…"  # Dấu câu tiếng Anh để kiểm tra chunk tạm
    punctuation_marks = (
        "。？！，、；：”’》」』）】…—"  # Dấu câu khác để kiểm tra chunk tạm
    )

    # Results list
    result = []
    # Starting position
    pos = 0
    text = text.strip()
    text_length = len(text)
    if text_length < max_chars:
        return [text]

    while pos < text_length:
        # Nếu đây là đoạn cuối cùng (phần còn lại nhỏ hơn hoặc bằng threshold_2), append luôn
        if pos + max_chars >= text_length:
            new_chunk = text[pos:].strip()
            if new_chunk:  # Chỉ append nếu chunk không rỗng
                result.append(new_chunk)
            break

        chunk_temp = None
        last_valid_end = None

        # Duyệt từ vị trí hiện tại để tìm điểm cắt tối ưu
        for i in range(pos, min(pos + threshold_2, text_length)):
            current_length = i - pos + 1
            char = text[i]

            # Ưu tiên 1: Kết thúc bằng final_punctuation, dài nhất có thể (tối đa threshold_2)
            if (
                char in final_punctuation
                and current_length <= threshold_2
                and current_length > min_chunk_length
            ):
                last_valid_end = i
                if (
                    i == text_length - 1 or text[i + 1].isspace()
                ):  # Đảm bảo không cắt giữa từ
                    chunk_temp = text[pos : i + 1].strip()
                    break

            # Ưu tiên 2: Lưu chunk tạm nếu thoả min_chunk_length < length < max_chars
            # và kết thúc bằng dấu trong english_punctuation hoặc punctuation_marks
            elif (
                min_chunk_length < current_length <= max_chars
                and char in (english_punctuation + punctuation_marks)
                and (i == text_length - 1 or text[i + 1].isspace())
            ):
                if not chunk_temp:  # Chỉ lưu chunk_temp đầu tiên thoả mãn
                    chunk_temp = text[pos : i + 1].strip()

        # Xử lý chunk đã tìm được
        if last_valid_end is not None:  # Ưu tiên 1: Có final_punctuation
            new_chunk = text[pos : last_valid_end + 1].strip()
            result.append(new_chunk)
            pos = last_valid_end + 1
        elif chunk_temp:  # Ưu tiên 2: Có chunk tạm thoả mãn
            result.append(chunk_temp)
            pos += len(chunk_temp)
        else:  # Ưu tiên 3: Không có dấu, cắt dựa trên penultimate word
            chunk = text[pos : pos + max_chars]
            last_space = chunk.rfind(" ")
            if last_space != -1:
                sub_chunk = text[pos : pos + last_space]
                tokenized_words = word_tokenize(sub_chunk)
                if len(tokenized_words) >= 2:
                    penultimate_word = tokenized_words[-2]
                    cut_pos = sub_chunk.rfind(penultimate_word) + len(penultimate_word)
                    new_chunk = text[pos : pos + cut_pos].strip()
                    result.append(new_chunk)
                    pos = pos + cut_pos
                else:
                    new_chunk = sub_chunk.strip()
                    result.append(new_chunk)
                    pos = pos + last_space
            else:  # Nếu không có khoảng trắng, cắt cứng ở max_chars
                new_chunk = chunk.strip()
                result.append(new_chunk)
                pos += max_chars

        # Bỏ qua khoảng trắng sau chunk
        while pos < text_length and text[pos].isspace():
            pos += 1

    return result


def chunk_text_with_break_words(text, max_chars=115):
    # Define length limits
    threshold = int(max_chars * 0.8)  # 80% của max_chars
    threshold_2 = int(max_chars * 1.2)  # 120% của max_chars
    punctuation_marks = "。？！，、；：”’》」』）】…—"
    english_punctuation = ".?!,:;)}…"
    final_punctuation = ".?"  # Chỉ dùng các dấu này để cắt khi vượt ngưỡng 80%
    # Results list
    result = []
    # Starting location
    pos = 0
    text = text.strip()
    text_length = len(text)

    i = 0
    last_punctuation_pos = None
    last_space_pos = None

    while i < text_length:
        char = text[i]
        current_length = i - pos + 1

        # Ghi nhớ vị trí dấu câu và dấu trắng
        if char in punctuation_marks or char in english_punctuation:
            if char == "." and i < text_length - 1 and re.match(r"\d", text[i + 1]):
                i += 1
                continue
            last_punctuation_pos = i
        elif char.isspace():
            last_space_pos = i

        # Trường hợp đặc biệt: cho phép vượt max_chars nếu trong threshold_2 và kết thúc bằng "."
        if (
            current_length <= threshold_2
            and char == "."
            and (i == text_length - 1 or text[i + 1].isspace())
        ):
            result.append(text[pos : i + 1].strip())
            pos = i + 1
            i = pos
            last_punctuation_pos = None
            last_space_pos = None
            continue

        # Kiểm tra khi vượt quá max_chars
        if current_length > max_chars:
            # Nếu vượt threshold_2 thì không áp dụng trường hợp đặc biệt
            if current_length > threshold_2:
                # Ưu tiên cắt ở dấu câu trước max_chars
                if (
                    last_punctuation_pos is not None
                    and last_punctuation_pos - pos + 1 <= max_chars
                ):
                    result.append(text[pos : last_punctuation_pos + 1].strip())
                    pos = last_punctuation_pos + 1
                # Nếu không có dấu câu, cắt ở dấu trắng
                elif (
                    last_space_pos is not None and last_space_pos - pos + 1 <= max_chars
                ):
                    result.append(text[pos:last_space_pos].strip())
                    pos = last_space_pos + 1
                # Nếu không có điểm cắt nào, tìm dấu trắng gần nhất trước max_chars
                else:
                    chunk = text[pos : pos + max_chars]
                    last_space = chunk.rfind(" ")
                    if last_space != -1:
                        result.append(chunk[:last_space].strip())
                        pos = pos + last_space + 1
                    else:
                        # Nếu không có dấu trắng nào trong max_chars, cắt ở dấu trắng cuối cùng trước đó
                        if pos > 0:
                            prev_chunk = text[:pos]
                            last_space_before = prev_chunk.rfind(" ")
                            if last_space_before != -1:
                                result[-1] = prev_chunk[:last_space_before].strip()
                                pos = last_space_before + 1
                            else:
                                pos += max_chars
                        else:
                            result.append(chunk.strip())
                            pos += max_chars
                i = pos
                last_punctuation_pos = None
                last_space_pos = None
            # Nếu trong threshold_2, tiếp tục để kiểm tra dấu chấm
        # Cắt ở ngưỡng 80% với dấu câu ưu tiên
        elif current_length >= threshold and char in final_punctuation:
            result.append(text[pos : i + 1].strip())
            pos = i + 1
            i = pos
            last_punctuation_pos = None
            last_space_pos = None

        i += 1

    # Xử lý phần text còn lại
    if pos < text_length:
        remaining = text[pos:].strip()
        while remaining:
            if len(remaining) <= max_chars or (
                len(remaining) <= threshold_2 and remaining.endswith(".")
            ):
                result.append(remaining)
                break

            # Tìm điểm cắt cho phần còn lại
            last_punctuation_pos = None
            last_space_pos = None
            for j, char in enumerate(remaining[:max_chars]):
                if char in punctuation_marks or char in english_punctuation:
                    if (
                        char == "."
                        and j < len(remaining) - 1
                        and re.match(r"\d", remaining[j + 1])
                    ):
                        continue
                    last_punctuation_pos = j
                elif char.isspace():
                    last_space_pos = j

            if last_punctuation_pos is not None:
                result.append(remaining[: last_punctuation_pos + 1].strip())
                remaining = remaining[last_punctuation_pos + 1 :].strip()
            elif last_space_pos is not None:
                result.append(remaining[:last_space_pos].strip())
                remaining = remaining[last_space_pos + 1 :].strip()
            else:
                chunk = remaining[:max_chars]
                last_space = chunk.rfind(" ")
                if last_space != -1:
                    result.append(chunk[:last_space].strip())
                    remaining = remaining[last_space + 1 :].strip()
                else:
                    if result:
                        last_chunk = result.pop()
                        last_space_before = last_chunk.rfind(" ")
                        if last_space_before != -1:
                            result.append(last_chunk[:last_space_before].strip())
                            remaining = last_chunk[last_space_before + 1 :] + remaining
                        else:
                            result.append(last_chunk)
                            result.append(chunk.strip())
                            remaining = remaining[max_chars:].strip()
                    else:
                        result.append(chunk.strip())
                        remaining = remaining[max_chars:].strip()

    return result


def chunk_text_v1(text, max_chars=115):
    """
    Splits the input text into chunks, each with a maximum number of characters.

    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if (
            len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8"))
            <= max_chars
        ):
            current_chunk += (
                sentence + " "
                if sentence and len(sentence[-1].encode("utf-8")) == 1
                else sentence
            )
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = (
                sentence + " "
                if sentence and len(sentence[-1].encode("utf-8")) == 1
                else sentence
            )

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# load vocoder
def load_vocoder(
    vocoder_name="vocos",
    is_local=False,
    local_path="",
    device=device,
    hf_cache_dir=None,
):
    if vocoder_name == "vocos":
        # vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        if is_local:
            print(f"Load vocos from local path {local_path}")
            config_path = f"{local_path}/config.yaml"
            model_path = f"{local_path}/pytorch_model.bin"
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            repo_id = "charactr/vocos-mel-24khz"
            config_path = hf_hub_download(
                repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml"
            )
            model_path = hf_hub_download(
                repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin"
            )
        vocoder = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        from vocos.feature_extractors import EncodecFeatures

        if isinstance(vocoder.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in vocoder.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        vocoder.load_state_dict(state_dict)
        vocoder = vocoder.eval().to(device)
    elif vocoder_name == "bigvgan":
        try:
            from third_party.BigVGAN import bigvgan
        except ImportError:
            print(
                "You need to follow the README to init submodule and change the BigVGAN source code."
            )
        if is_local:
            # download generator from https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x/tree/main
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)
        else:
            vocoder = bigvgan.BigVGAN.from_pretrained(
                "nvidia/bigvgan_v2_24khz_100band_256x",
                use_cuda_kernel=False,
                cache_dir=hf_cache_dir,
            )

        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
    return vocoder


# load asr pipeline

asr_pipe = None


def initialize_asr_pipeline(device: str = device, dtype=None):
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 7
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    global asr_pipe
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=dtype,
        device=device,
    )


# transcribe


def transcribe(ref_audio, language=None):
    global asr_pipe
    if asr_pipe is None:
        initialize_asr_pipeline(device=device)
    return asr_pipe(
        ref_audio,
        chunk_length_s=30,
        batch_size=128,
        generate_kwargs=(
            {"task": "transcribe", "language": language}
            if language
            else {"task": "transcribe"}
        ),
        return_timestamps=False,
    )["text"].strip()


# load model checkpoint for inference


def load_checkpoint(model, ckpt_path, device: str, dtype=None, use_ema=True):
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 7
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    model = model.to(dtype)

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(ckpt_path, device=device)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    print(f"load checkpoint with use_ema={use_ema}")

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }

        # patch for backward compatibility, 305e3ea
        for key in [
            "mel_spec.mel_stft.mel_scale.fb",
            "mel_spec.mel_stft.spectrogram.window",
        ]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"])

    del checkpoint
    torch.cuda.empty_cache()

    return model.to(device)


# load model for inference


def load_model(
    model_cls,
    model_cfg,
    ckpt_path,
    mel_spec_type=mel_spec_type,
    vocab_file="",
    ode_method=ode_method,
    use_ema=True,  #
    device=device,
):
    if vocab_file == "":
        vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
    tokenizer = "custom"

    print("\nvocab : ", vocab_file)
    print("token : ", tokenizer)
    print("model : ", ckpt_path, "\n")

    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)
    model = CFM(
        transformer=model_cls(
            **model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels
        ),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    return model


def remove_silence_edges(audio, silence_threshold=-42):
    # Remove silence from the start
    non_silent_start_idx = silence.detect_leading_silence(
        audio, silence_threshold=silence_threshold
    )
    audio = audio[non_silent_start_idx:]

    # Remove silence from the end
    non_silent_end_duration = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        non_silent_end_duration -= 0.001
    trimmed_audio = audio[: int(non_silent_end_duration * 1000)]

    return trimmed_audio


# preprocess reference audio and text


def preprocess_ref_audio_text(
    ref_audio_orig, ref_text, clip_short=True, show_info=print
):
    show_info("Converting audio...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio_orig)

        if clip_short:
            # 1. try to find long silence for clipping
            non_silent_segs = silence.split_on_silence(
                aseg,
                min_silence_len=1000,
                silence_thresh=-50,
                keep_silence=1000,
                seek_step=10,
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                if (
                    len(non_silent_wave) > 6000
                    and len(non_silent_wave + non_silent_seg) > 12000
                ):
                    show_info("Audio is over 12s, clipping short. (1)")
                    break
                non_silent_wave += non_silent_seg

            # 2. try to find short silence for clipping if 1. failed
            if len(non_silent_wave) > 12000:
                non_silent_segs = silence.split_on_silence(
                    aseg,
                    min_silence_len=100,
                    silence_thresh=-40,
                    keep_silence=1000,
                    seek_step=10,
                )
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    if (
                        len(non_silent_wave) > 6000
                        and len(non_silent_wave + non_silent_seg) > 12000
                    ):
                        show_info("Audio is over 12s, clipping short. (2)")
                        break
                    non_silent_wave += non_silent_seg

            aseg = non_silent_wave

            # 3. if no proper silence found for clipping
            if len(aseg) > 12000:
                aseg = aseg[:12000]
                show_info("Audio is over 12s, clipping short. (3)")

        aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
        aseg.export(f.name, format="wav")
        ref_audio = f.name

    # Compute a hash of the reference audio file
    with open(ref_audio, "rb") as audio_file:
        audio_data = audio_file.read()
        audio_hash = hashlib.md5(audio_data).hexdigest()

    if not ref_text.strip():
        global _ref_audio_cache
        if audio_hash in _ref_audio_cache:
            # Use cached asr transcription
            show_info("Using cached reference text...")
            ref_text = _ref_audio_cache[audio_hash]
        else:
            show_info("No reference text provided, transcribing reference audio...")
            ref_text = transcribe(ref_audio)
            # Cache the transcribed text (not caching custom ref_text, enabling users to do manual tweak)
            _ref_audio_cache[audio_hash] = ref_text
    else:
        show_info("Using custom reference text...")

    # Ensure ref_text ends with a proper sentence-ending punctuation
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    print("\nref_text  ", ref_text)

    return ref_audio, ref_text


# infer process: chunk text -> infer batches [i.e. infer_batch_process()]

# infer batches


def infer_process(
    ref_audio,
    ref_text,
    gen_text,
    model_obj,
    vocoder,
    mel_spec_type=mel_spec_type,
    show_info=print,
    progress=tqdm,
    target_rms=target_rms,
    cross_fade_duration=cross_fade_duration,
    nfe_step=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
    speed=speed,
    fix_duration=fix_duration,
    device=device,
):
    # Chuỗi văn bản nháp để warm-up mô hình
    warmup_text = (
        "Đây là một câu văn bản nháp để khởi động mô hình."  # Có thể tùy chỉnh
    )
    warmup_text = post_process(TTSnorm(warmup_text, rule=False))
    print(f"Warm-up text: {warmup_text}")

    # Split the input text into batches
    audio, sr = torchaudio.load(ref_audio)
    ref_text = post_process(TTSnorm(ref_text, rule=False))
    print(f"Norm ref_text: {ref_text}")
    gen_text = post_process(TTSnorm(gen_text, rule=False))
    print(f"Norm gen_text: {gen_text}")

    max_chars = int(
        len(ref_text.encode("utf-8"))
        / (audio.shape[-1] / sr)
        * (22 - audio.shape[-1] / sr)
    )

    # Thêm chuỗi nháp vào danh sách batches
    gen_text_batches = [warmup_text] + chunk_text(gen_text)
    for i, gen_text in enumerate(gen_text_batches):
        print(f"gen_text {i}-{len(gen_text)}", gen_text)
    print("\n")

    show_info(
        f"Generating audio in {len(gen_text_batches)} batches (including warm-up)..."
    )

    # Gọi infer_batch_process
    result = next(
        infer_batch_process(
            (audio, sr),
            ref_text,
            gen_text_batches,
            model_obj,
            vocoder,
            mel_spec_type=mel_spec_type,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=device,
        )
    )

    # Kiểm tra kết quả từ infer_batch_process
    final_wave, final_sample_rate, combined_spectrogram = result
    return final_wave, final_sample_rate, combined_spectrogram


def infer_batch_process(
    ref_audio,
    ref_text,
    gen_text_batches,
    model_obj,
    vocoder,
    mel_spec_type="vocos",
    progress=tqdm,
    target_rms=0.1,
    cross_fade_duration=0.15,
    nfe_step=32,
    cfg_strength=2.0,
    sway_sampling_coef=-1,
    speed=1,
    fix_duration=None,
    device=None,
    streaming=False,
    chunk_size=2048,
):
    audio, sr = ref_audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    generated_waves = []
    spectrograms = []

    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "

    def process_batch(gen_text):
        local_speed = speed
        if len(gen_text.encode("utf-8")) < 10:
            local_speed = 0.3

        # Prepare the text
        text_list = [ref_text + gen_text]
        final_text_list = convert_char_to_pinyin(text_list)

        ref_audio_len = audio.shape[-1] // hop_length
        if fix_duration is not None:
            duration = int(fix_duration * target_sample_rate / hop_length)
        else:
            # Calculate duration
            ref_text_len = len(ref_text.encode("utf-8"))
            gen_text_len = len(gen_text.encode("utf-8"))
            duration = ref_audio_len + int(
                ref_audio_len / ref_text_len * gen_text_len / local_speed
            )

        # inference
        with torch.inference_mode():
            generated, _ = model_obj.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )
            del _

            generated = generated.to(torch.float32)  # generated mel spectrogram
            generated = generated[:, ref_audio_len:, :]
            generated = generated.permute(0, 2, 1)
            if mel_spec_type == "vocos":
                generated_wave = vocoder.decode(generated)
            elif mel_spec_type == "bigvgan":
                generated_wave = vocoder(generated)
            if rms < target_rms:
                generated_wave = generated_wave * rms / target_rms

            # wav -> numpy
            generated_wave = generated_wave.squeeze().cpu().numpy()

            if streaming:
                for j in range(0, len(generated_wave), chunk_size):
                    yield generated_wave[j : j + chunk_size], target_sample_rate
            else:
                generated_cpu = generated[0].cpu().numpy()
                del generated
                yield generated_wave, generated_cpu

    if streaming:
        for gen_text in (
            progress.tqdm(gen_text_batches)
            if progress is not None
            else gen_text_batches
        ):
            for chunk in process_batch(gen_text):
                yield chunk
    else:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_batch, gen_text)
                for gen_text in gen_text_batches
            ]
            for i, future in enumerate(
                progress.tqdm(futures) if progress is not None else futures
            ):
                result = future.result()
                if result:
                    generated_wave, generated_mel_spec = next(result)
                    # Bỏ qua batch đầu tiên (warm-up) nếu i == 0
                    if i == 0:
                        continue
                    generated_waves.append(generated_wave)
                    spectrograms.append(generated_mel_spec)

        if generated_waves:
            if cross_fade_duration <= 0:
                # Simply concatenate
                final_wave = np.concatenate(generated_waves)
            else:
                # Combine all generated waves with cross-fading
                final_wave = generated_waves[0]
                for i in range(1, len(generated_waves)):
                    prev_wave = final_wave
                    next_wave = generated_waves[i]

                    # Calculate cross-fade samples, ensuring it does not exceed wave lengths
                    cross_fade_samples = int(cross_fade_duration * target_sample_rate)
                    cross_fade_samples = min(
                        cross_fade_samples, len(prev_wave), len(next_wave)
                    )

                    if cross_fade_samples <= 0:
                        # No overlap possible, concatenate
                        final_wave = np.concatenate([prev_wave, next_wave])
                        continue

                    # Overlapping parts
                    prev_overlap = prev_wave[-cross_fade_samples:]
                    next_overlap = next_wave[:cross_fade_samples]

                    # Fade out and fade in
                    fade_out = np.linspace(1, 0, cross_fade_samples)
                    fade_in = np.linspace(0, 1, cross_fade_samples)

                    # Cross-faded overlap
                    cross_faded_overlap = (
                        prev_overlap * fade_out + next_overlap * fade_in
                    )

                    # Combine
                    new_wave = np.concatenate(
                        [
                            prev_wave[:-cross_fade_samples],
                            cross_faded_overlap,
                            next_wave[cross_fade_samples:],
                        ]
                    )

                    final_wave = new_wave

            # Create a combined spectrogram
            combined_spectrogram = np.concatenate(spectrograms, axis=1)

            yield final_wave, target_sample_rate, combined_spectrogram

        else:
            yield None, target_sample_rate, None


# remove silence from generated wav


def remove_silence_for_generated_wav(filename):
    aseg = AudioSegment.from_file(filename)
    non_silent_segs = silence.split_on_silence(
        aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500, seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        non_silent_wave += non_silent_seg
    aseg = non_silent_wave
    aseg.export(filename, format="wav")


# save spectrogram


def save_spectrogram(spectrogram, path):
    plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    plt.colorbar()
    plt.savefig(path)
    plt.close()
