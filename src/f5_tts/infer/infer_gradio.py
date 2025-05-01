# ruff: noqa: E402
# Above allows ruff to ignore E402: module level import not at top of file

import gc
import json
import re
import tempfile
from collections import OrderedDict
from importlib.resources import files

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio
from cached_path import cached_path
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
import nltk
import json
from nltk.tokenize import sent_tokenize
import magic
from mutagen.id3 import ID3, APIC, error, TIT2, TPE1
import subprocess
import shutil

#####################

import markitdown
import zipfile
import xml.etree.ElementTree as ET
import csv
from urllib.parse import urlparse
from typing import Tuple, Optional


from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)

DEFAULT_TTS_MODEL = "F5-TTS_v1"  # "Custom"  #
tts_model_choice = DEFAULT_TTS_MODEL
use_ema_value = True  # mặc định


# DEFAULT_TTS_MODEL_CFG = [
#     "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
#     "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
#     json.dumps(
#         dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
#     ),
# ]

DEFAULT_TTS_MODEL_CFG = [
    "/workspace/F5-TTS/ckpts/vivoice/model_last.pt",
    "/workspace/F5-TTS/ckpts/vivoice/vocab.txt",
    json.dumps(
        dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    ),
]


# NLTK 'punkt' resource check
try:
    nltk.data.find("tokenizers/punkt")
    print("NLTK 'punkt' resource found.")
except LookupError:
    print("NLTK 'punkt' resource not found. Attempting download...")
    try:
        nltk_data_path = os.environ.get("NLTK_DATA", None)
        if nltk_data_path:
            print(f"Using NLTK_DATA path: {nltk_data_path}")
            if ":" in nltk_data_path:
                nltk_data_path = nltk_data_path.split(":")[0]
            os.makedirs(os.path.join(nltk_data_path, "tokenizers"), exist_ok=True)
            nltk.download("punkt", quiet=False, download_dir=nltk_data_path)
        else:
            nltk.download("punkt", quiet=False)
        nltk.data.find("tokenizers/punkt")
        print("'punkt' resource downloaded successfully.")
    except Exception as download_e:
        print(f"Failed to download NLTK 'punkt' resource: {download_e}.")


# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Spaces Check
try:
    import spaces

    USING_SPACES = True
except ImportError:
    USING_SPACES = False

# Constants
OUTPUT_DIR = os.path.join("Working_files", "Book")
TEMP_CONVERT_DIR = os.path.join("Working_files", "temp_converted")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_CONVERT_DIR, exist_ok=True)
DEFAULT_REF_AUDIO_PATH = "/workspace/F5-TTS/tools/ebook/default_voice.mp3"
DEFAULT_REF_TEXT = "For thirty-six years I was the confidential secretary of the Roman statesman Cicero. At first this was exciting, then astonishing, then arduous, and finally extremely dangerous."

# def create_audiobooks_tab():
"""Create the Audiobooks tab for integration into a TabbedInterface."""
default_audio_exists = os.path.exists(DEFAULT_REF_AUDIO_PATH)
available_bitrates = ["128k", "192k", "256k", "320k"]
default_bitrate = "320k"


# GPU Decorator
def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    return func


# load models
vocoder = load_vocoder()


def get_ckpt_choices(folder_path):
    # Lấy danh sách file kết thúc bằng .pt trong thư mục chỉ định
    choices = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".pt")
    ]
    return sorted(choices)  # Tuỳ chọn: sắp xếp theo tên file


ckpt_folder = "/workspace/F5-TTS/ckpts/vivoice"
ckpt_choices = get_ckpt_choices(ckpt_folder)


def on_checkbox_change(new_value):
    global use_ema_value
    use_ema_value = new_value
    print("Checkbox updated:", use_ema_value)


def load_f5tts():
    global use_ema_value
    # ckpt_path = str(cached_path(DEFAULT_TTS_MODEL_CFG[0]))
    ckpt_path = str(DEFAULT_TTS_MODEL_CFG[0])
    F5TTS_model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    vocab_file = str(DEFAULT_TTS_MODEL_CFG[1])
    return load_model(
        DiT, F5TTS_model_cfg, ckpt_path, vocab_file=vocab_file, use_ema=use_ema_value
    )


def load_e2tts():
    global use_ema_value
    ckpt_path = str(
        cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors")
    )
    E2TTS_model_cfg = dict(
        dim=1024, depth=24, heads=16, ff_mult=4, text_mask_padding=False, pe_attn_head=1
    )
    return load_model(UNetT, E2TTS_model_cfg, ckpt_path, use_ema=use_ema_value)


def load_custom(ckpt_path: str, vocab_path="", model_cfg=None):
    global use_ema_value
    ckpt_path, vocab_path = ckpt_path.strip(), vocab_path.strip()
    if ckpt_path.startswith("hf://"):
        ckpt_path = str(cached_path(ckpt_path))
    if vocab_path.startswith("hf://"):
        vocab_path = str(cached_path(vocab_path))
    if model_cfg is None:
        model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    return load_model(
        DiT, model_cfg, ckpt_path, vocab_file=vocab_path, use_ema=use_ema_value
    )


F5TTS_ema_model = load_f5tts()
E2TTS_ema_model = load_e2tts() if USING_SPACES else None
custom_ema_model, pre_custom_path = None, ""

chat_model_state = None
chat_tokenizer_state = None


@gpu_decorator
def generate_response(messages, model, tokenizer):
    """Generate response using Qwen"""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
    )

    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def choose_model(
    model,
    show_info=gr.Info,
):
    if model == DEFAULT_TTS_MODEL:
        ema_model = F5TTS_ema_model
    elif model == "E2-TTS":
        global E2TTS_ema_model
        if E2TTS_ema_model is None:
            show_info("Loading E2-TTS model...")
            E2TTS_ema_model = load_e2tts()
        ema_model = E2TTS_ema_model
    elif isinstance(model, list) and model[0] == "Custom":
        assert not USING_SPACES, "Only official checkpoints allowed in Spaces."
        global custom_ema_model, pre_custom_path
        if pre_custom_path != model[1]:
            show_info("Loading Custom TTS model...")
            custom_ema_model = load_custom(
                model[1], vocab_path=model[2], model_cfg=model[3]
            )
            pre_custom_path = model[1]
        ema_model = custom_ema_model
    return ema_model


@gpu_decorator
def infer(
    ref_audio_orig,
    ref_text,
    gen_text,
    model,
    remove_silence,
    cross_fade_duration=0.15,
    nfe_step=32,
    speed=1,
    show_info=gr.Info,
):
    if not ref_audio_orig:
        gr.Warning("Please provide reference audio.")
        return gr.update(), gr.update(), ref_text

    if not gen_text.strip():
        gr.Warning("Please enter text to generate.")
        return gr.update(), gr.update(), ref_text

    ref_audio, ref_text = preprocess_ref_audio_text(
        ref_audio_orig, ref_text, show_info=show_info
    )

    ema_model = choose_model(model, show_info=show_info)

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
        show_info=show_info,
        progress=gr.Progress(),
    )

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Save the spectrogram
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave), spectrogram_path, ref_text


################ audiobooks ################


# Utility Functions
def extract_metadata_and_cover(ebook_path):
    print(f"Attempting to extract cover from: {ebook_path}")
    cover_path = None
    try:
        temp_cover_dir = TEMP_CONVERT_DIR
        ensure_directory(temp_cover_dir)
        with tempfile.NamedTemporaryFile(
            dir=temp_cover_dir, suffix=".jpg", delete=False
        ) as tmp_cover:
            cover_path = tmp_cover.name
        command = ["ebook-meta", ebook_path, "--get-cover", cover_path]
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if (
            result.returncode == 0
            and os.path.exists(cover_path)
            and os.path.getsize(cover_path) > 0
        ):
            print(f"Cover extracted to {cover_path}")
            return cover_path
        else:
            print(f"Failed to extract cover. RC: {result.returncode}")
            if cover_path and os.path.exists(cover_path):
                os.remove(cover_path)
            return None
    except FileNotFoundError:
        print("Error: 'ebook-meta' not found.")
        if cover_path and os.path.exists(cover_path):
            try:
                os.remove(cover_path)
            except OSError:
                pass
        return None
    except Exception as e:
        print(f"Error extracting cover: {e}")
        if cover_path and os.path.exists(cover_path):
            try:
                os.remove(cover_path)
            except OSError:
                pass
        return None


def embed_metadata_into_mp3(mp3_path, cover_image_path, title, author):
    if not mp3_path or not os.path.exists(mp3_path):
        print("MP3 path invalid.")
        return
    print(f"Embedding metadata into {mp3_path}")
    try:
        audio = ID3(mp3_path)
    except error as e:
        print(f"Error loading ID3 tags: {e}")
        audio = ID3()
    if cover_image_path and os.path.exists(cover_image_path):
        try:
            audio.delall("APIC")
            with open(cover_image_path, "rb") as img:
                image_data = img.read()
            mime_type = magic.from_buffer(image_data, mime=True)
            audio.add(
                APIC(
                    encoding=3,
                    mime=mime_type,
                    type=3,
                    desc="Front cover",
                    data=image_data,
                )
            )
            print("Prepared cover image.")
        except Exception as e:
            print(f"Failed to prepare cover: {e}")
    if title:
        audio.delall("TIT2")
        audio.add(TIT2(encoding=3, text=title))
        print(f"Prepared title '{title}'.")
    if author:
        audio.delall("TPE1")
        audio.add(TPE1(encoding=3, text=author))
        print(f"Prepared author '{author}'.")
    try:
        audio.save(mp3_path, v2_version=3)
        print(f"Saved metadata to {mp3_path}")
    except Exception as e:
        print(f"Failed to save metadata: {e}")


def extract_text_title_author_from_epub(file_path: str) -> Tuple[str, str, str]:
    """
    Extract text, title, and author from various file formats using markitdown and specialized libraries.

    Args:
        file_path: Path to the file (local or URL for YouTube).

    Returns:
        Tuple of (full_text, title, author).

    Raises:
        ValueError: If no text can be extracted.
        RuntimeError: If file reading fails.
    """
    file_extension = (
        os.path.splitext(file_path)[1].lower()
        if not file_path.startswith("http")
        else "youtube"
    )
    print(f"file_extension: {file_extension}")
    text_content = []
    title = (
        os.path.splitext(os.path.basename(file_path))[0]
        if not file_path.startswith("http")
        else "YouTube Video"
    )
    author = "Unknown Author"

    def clean_text(text: str) -> str:
        """Clean extracted text by normalizing whitespace."""
        text = re.sub(r"[ \t]{2,}", " ", text.strip())
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    # EPUB handling
    if file_extension == ".epub":
        try:
            book = epub.read_epub(file_path)
            print(f"EPUB '{os.path.basename(file_path)}' read.")
        except Exception as e:
            raise RuntimeError(f"Failed to read EPUB: {e}")

        try:
            metadata_title = book.get_metadata("DC", "title")
            title = metadata_title[0][0] if metadata_title else title
            print(f"Using title: {title}")
        except Exception as e:
            print(f"Could not get title: {e}")

        try:
            metadata_creator = book.get_metadata("DC", "creator")
            author = (
                ", ".join([creator[0] for creator in metadata_creator])
                if metadata_creator
                else author
            )
            print(f"Using author: {author}")
        except Exception as e:
            print(f"Could not get author: {e}")

        items_processed = 0
        spine_ids = [item[0] for item in book.spine] if book.spine else []
        ordered_items = []
        if spine_ids:
            item_map = {item.id: item for item in book.get_items()}
            for item_id in spine_ids:
                if (
                    item_id in item_map
                    and item_map[item_id].get_type() == ITEM_DOCUMENT
                ):
                    ordered_items.append(item_map[item_id])
            for item in book.get_items_of_type(ITEM_DOCUMENT):
                if item.id not in spine_ids:
                    ordered_items.append(item)
            print(f"Processing {len(ordered_items)} items.")
        else:
            ordered_items = list(book.get_items_of_type(ITEM_DOCUMENT))
            print("Warning: EPUB spine missing.")

        for item in ordered_items:
            try:
                soup = BeautifulSoup(item.get_content(), "html.parser")
                for script_or_style in soup(["script", "style", "head", "title"]):
                    script_or_style.decompose()
                paragraphs = soup.find_all(
                    ["p", "h1", "h2", "h3", "h4", "h5", "h6", "div"]
                )
                item_text_parts = []
                for tag in paragraphs:
                    text_part = tag.get_text(separator=" ", strip=True)
                    if text_part:
                        item_text_parts.append(text_part)
                text = "\n\n".join(item_text_parts)
                text = clean_text(text)
                if text:
                    text_content.append(text)
                    items_processed += 1
            except Exception as e:
                print(f"Error parsing item {item.get_id()}: {e}")

        if not text_content:
            raise ValueError(f"No text extracted from EPUB: {file_path}")
        full_text = "\n\n".join(text_content)
        print(f"Extracted {len(full_text)} chars from {items_processed} documents.")
        return full_text, title, author

    # Markitdown-supported formats
    elif file_extension in [".pdf", ".docx", ".pptx", ".xlsx", ".html"]:
        try:
            result = markitdown.parse(file_path)
            text_content.append(clean_text(result["markdown"]))
            title = result.get("metadata", {}).get("title", title)
            author = result.get("metadata", {}).get("author", author)
            full_text = "\n\n".join(text_content)
            print(f"Extracted {len(full_text)} chars from {file_extension} file.")
            return full_text, title, author
        except Exception as e:
            raise RuntimeError(f"Failed to parse {file_extension} with markitdown: {e}")

    # Text files
    elif file_extension == ".txt":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text_content.append(clean_text(f.read()))
            full_text = "\n".join(text_content)
            if not full_text:
                raise ValueError(f"No text extracted from TXT: {file_path}")
            print(f"Extracted {len(full_text)} chars from TXT.")
            print(f"full text: {full_text}")
            return full_text, title, author
        except Exception as e:
            raise RuntimeError(f"Failed to parse TXT: {e}")

    # Text-based formats (CSV, JSON, XML)
    elif file_extension == ".csv":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                text_content = [" ".join(row) for row in reader]
                full_text = clean_text("\n\n".join(text_content))
                print(f"Extracted {len(full_text)} chars from CSV.")
                return full_text, title, author
        except Exception as e:
            raise RuntimeError(f"Failed to parse CSV: {e}")

    elif file_extension == ".json":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                text_content.append(clean_text(json.dumps(data, indent=2)))
                full_text = "\n\n".join(text_content)
                print(f"Extracted {len(full_text)} chars from JSON.")
                return full_text, title, author
        except Exception as e:
            raise RuntimeError(f"Failed to parse JSON: {e}")

    elif file_extension == ".xml":
        try:
            tree = ET.parse(file_path)
            text_content.append(
                clean_text(
                    ET.tostring(tree.getroot(), encoding="unicode", method="text")
                )
            )
            full_text = "\n\n".join(text_content)
            print(f"Extracted {len(full_text)} chars from XML.")
            return full_text, title, author
        except Exception as e:
            raise RuntimeError(f"Failed to parse XML: {e}")

    # ZIP files
    elif file_extension == ".zip":
        try:
            with zipfile.ZipFile(file_path, "r") as z:
                for file_name in z.namelist():
                    if file_name.endswith((".txt", ".md", ".html")):
                        with z.open(file_name) as f:
                            text_content.append(clean_text(f.read().decode("utf-8")))
            full_text = "\n\n".join(text_content)
            if not full_text:
                raise ValueError(f"No text extracted from ZIP: {file_path}")
            print(f"Extracted {len(full_text)} chars from ZIP.")
            return full_text, title, author
        except Exception as e:
            raise RuntimeError(f"Failed to parse ZIP: {e}")


def extract_text_title_author_from_epub_v1(epub_path):
    try:
        book = epub.read_epub(epub_path)
        print(f"EPUB '{os.path.basename(epub_path)}' read.")
    except Exception as e:
        raise RuntimeError(f"Failed to read EPUB: {e}")
    text_content = []
    title = "Untitled Audiobook"
    author = "Unknown Author"
    try:
        metadata_title = book.get_metadata("DC", "title")
        title = (
            metadata_title[0][0]
            if metadata_title
            else os.path.splitext(os.path.basename(epub_path))[0]
        )
        print(f"Using title: {title}")
    except Exception as e:
        print(f"Could not get title: {e}")
        title = os.path.splitext(os.path.basename(epub_path))[0]
    try:
        metadata_creator = book.get_metadata("DC", "creator")
        author = (
            ", ".join([creator[0] for creator in metadata_creator])
            if metadata_creator
            else "Unknown Author"
        )
        print(f"Using author: {author}")
    except Exception as e:
        print(f"Could not get author: {e}")
    items_processed = 0
    spine_ids = [item[0] for item in book.spine] if book.spine else []
    ordered_items = []
    if spine_ids:
        item_map = {item.id: item for item in book.get_items()}
        for item_id in spine_ids:
            if item_id in item_map and item_map[item_id].get_type() == ITEM_DOCUMENT:
                ordered_items.append(item_map[item_id])
        for item in book.get_items_of_type(ITEM_DOCUMENT):
            if item.id not in spine_ids:
                ordered_items.append(item)
        print(f"Processing {len(ordered_items)} items.")
    else:
        ordered_items = list(book.get_items_of_type(ITEM_DOCUMENT))
        print("Warning: EPUB spine missing.")
    for item in ordered_items:
        try:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            for script_or_style in soup(["script", "style", "head", "title"]):
                script_or_style.decompose()
            paragraphs = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "div"])
            item_text_parts = []
            for tag in paragraphs:
                text_part = tag.get_text(separator=" ", strip=True)
                text_part = re.sub(r"\s+", " ", text_part).strip()
                if text_part:
                    item_text_parts.append(text_part)
            text = "\n\n".join(item_text_parts)
            text = re.sub(r"[ \t]{2,}", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            if text:
                text_content.append(text)
                items_processed += 1
        except Exception as e:
            print(f"Error parsing item {item.get_id()}: {e}")
    if not text_content:
        raise ValueError(f"No text extracted from EPUB: {epub_path}")
    full_text = "\n\n".join(text_content)
    print(f"Extracted {len(full_text)} chars from {items_processed} documents.")
    return full_text, title, author


def convert_to_epub(input_path, output_dir):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    sanitized_base = sanitize_filename(base_name)
    output_path = os.path.join(output_dir, f"{sanitized_base}.epub")
    ensure_directory(output_dir)
    try:
        print(f"Converting '{input_path}' to EPUB...")
        command = [
            "ebook-convert",
            input_path,
            output_path,
            "--enable-heuristics",
            "--keep-ligatures",
            "--input-encoding=utf-8",
        ]
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError(f"Output EPUB empty. RC: {result.returncode}.")
        print(f"Converted to '{output_path}'.")
        return output_path
    except FileNotFoundError:
        raise RuntimeError("Error: 'ebook-convert' not found.")
    except Exception as e:
        raise RuntimeError(f"Conversion failed: {e}")


def detect_file_type(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        mime_type = magic.Magic(mime=True).from_file(file_path)
        if mime_type is None:
            ext = os.path.splitext(file_path)[1].lower()
            mime_map = {
                ".epub": "application/epub+zip",
                ".mobi": "application/x-mobipocket-ebook",
                ".pdf": "application/pdf",
                ".txt": "text/plain",
                ".html": "text/html",
                ".azw3": "application/vnd.amazon.ebook",
                ".fb2": "application/x-fictionbook+xml",
            }
            return mime_map.get(ext, None)
        return mime_type
    except Exception as e:
        print(f"Error detecting file type: {e}")
        return None


def ensure_directory(directory_path):
    if not directory_path:
        raise ValueError("Directory path cannot be empty.")
    try:
        os.makedirs(directory_path, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Error creating directory: {e}")


def sanitize_filename(filename):
    if not filename:
        return "default_filename"
    sanitized = re.sub(r'[\\/*?:"<>|]', "", filename)
    sanitized = re.sub(r"\s+", "_", sanitized)
    sanitized = sanitized.strip("_ ")
    return sanitized if sanitized else "sanitized_filename"


def show_converted_audiobooks():
    if not os.path.exists(OUTPUT_DIR):
        print(f"Output directory '{OUTPUT_DIR}' does not exist.")
        return []
    try:
        files = [
            os.path.join(OUTPUT_DIR, f)
            for f in os.listdir(OUTPUT_DIR)
            if f.lower().endswith(".mp3")
        ]
        files.sort(key=os.path.getmtime, reverse=True)
        return files
    except Exception as e:
        print(f"Error listing audiobooks: {e}")
        return []


def split_text_into_chunks(text, max_length):
    chunks = []
    current_chunk = ""
    print(f"Splitting text with max_length = {max_length}")
    try:
        sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        sentences = sent_tokenizer.tokenize(text)
        print(f"Tokenized into {len(sentences)} sentences.")
    except Exception as e:
        print(f"NLTK tokenization failed: {e}. Using fallback.")
        sentences = [p.strip() for p in text.split("\n\n") if p.strip()] or [
            p.strip() for p in text.split("\n") if p.strip()
        ]
        if not sentences and len(text) > max_length:
            sentences = [
                text[i : i + max_length] for i in range(0, len(text), max_length)
            ]
        elif not sentences and text:
            sentences = [text]
        print(f"Split into {len(sentences)} segments.")
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_length:
            print(f"Sentence exceeds max_length ({len(sentence)}). Splitting.")
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            for i in range(0, len(sentence), max_length):
                chunks.append(sentence[i : i + max_length])
        elif len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += (" " + sentence) if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    chunks = [c for c in chunks if c]
    print(f"Split into {len(chunks)} chunks.")
    return chunks


# @gpu_decorator
# def infer_chunk(
#     ref_audio_orig, ref_text, text_chunk, model, cross_fade_duration, speed, show_info
# ):
#     try:
#         effective_show_info = (
#             show_info if callable(show_info) else (lambda *args, **kwargs: None)
#         )
#         ref_audio, processed_ref_text = preprocess_ref_audio_text(
#             ref_audio_orig, ref_text, show_info=effective_show_info
#         )
#         if not ref_text and not processed_ref_text:
#             raise RuntimeError("Transcription failed or empty.")
#         elif not ref_text and processed_ref_text:
#             print("Transcription successful.")
#     except Exception as e:
#         import traceback

#         print(f"Preprocessing error: {e}\n{traceback.format_exc()}")
#         raise RuntimeError(f"Preprocessing failed: {e}")
#     if not text_chunk or text_chunk.isspace():
#         print("Skipping empty chunk.")
#         return None
#     try:
#         ema_model = choose_model(model, show_info=show_info)
#         with torch.no_grad():
#             final_wave, final_sample_rate, _ = infer_process(
#                 ref_audio,
#                 processed_ref_text,
#                 text_chunk,
#                 ema_model,
#                 vocoder,
#                 cross_fade_duration=cross_fade_duration,
#                 speed=speed,
#                 show_info=effective_show_info,
#             )
#         if final_wave is None or final_sample_rate is None:
#             print("Inference returned None.")
#             return None
#         if isinstance(final_wave, torch.Tensor) and final_wave.numel() == 0:
#             print("Inference returned empty tensor.")
#             return None
#         if isinstance(final_wave, np.ndarray) and final_wave.size == 0:
#             print("Inference returned empty array.")
#             return None
#         print(f"Generated chunk: {len(final_wave)} samples @ {final_sample_rate} Hz.")
#         return (final_sample_rate, final_wave)
#     except Exception as e:
#         import traceback

#         print(f"Inference error: {e}\n{traceback.format_exc()}")
#         return None


def process_ebook_to_audio(
    ref_audio_input,
    ref_text_input,
    ebook_path,
    cross_fade_duration,
    nfe_slider,
    speed,
    max_chunk_length,
    mp3_bitrate,
    model,
    progress=gr.Progress(track_tqdm=True),
):
    temp_dir = None
    temp_chunk_files = []
    final_mp3_path = None
    converted_epub_path = None
    extracted_cover_path = None
    sample_rate = 24000
    ebook_title = "Untitled"
    ebook_author = "Unknown Author"
    try:
        if not ebook_path or not os.path.exists(ebook_path):
            yield None, f"Error: File not found: {ebook_path}"
            return
        progress(0, desc=f"Starting: {os.path.basename(ebook_path)}")
        original_input_path = ebook_path
        file_type = detect_file_type(ebook_path)
        print(f"Detected file type: {file_type}")
        if file_type not in ["application/epub+zip"]:  # "text/plain",
            progress(0.05, desc="Converting to EPUB...")
            converted_epub_path = convert_to_epub(ebook_path, TEMP_CONVERT_DIR)
            epub_path_to_process = converted_epub_path
        else:
            epub_path_to_process = ebook_path
        progress(0.1, desc="Extracting text/metadata...")
        gen_text, ebook_title, ebook_author = extract_text_title_author_from_epub(
            epub_path_to_process
        )

        extracted_cover_path = extract_metadata_and_cover(epub_path_to_process)
        ref_text = ref_text_input
        sanitized_title = sanitize_filename(ebook_title)
        sanitized_author = sanitize_filename(ebook_author)
        base_filename = (
            f"{sanitized_title}_by_{sanitized_author}"
            if sanitized_author.lower() != "unknown_author"
            else sanitized_title
        )
        final_mp3_path = os.path.join(OUTPUT_DIR, f"{base_filename}.mp3")
        ensure_directory(os.path.dirname(final_mp3_path))
        progress(0.2, desc="Splitting text...")
        text_chunks = split_text_into_chunks(gen_text, max_length=max_chunk_length)
        if not text_chunks:
            yield None, "Error: No chunks generated."
            return
        temp_dir = tempfile.mkdtemp(prefix="audio_chunks_")
        successful_chunks = 0
        first_chunk_processed = False
        dummy_show_info = lambda *args, **kwargs: None
        for i, chunk in enumerate(text_chunks):
            chunk_start_progress = 0.25 + (i / len(text_chunks)) * 0.5
            progress(chunk_start_progress, desc=f"Chunk {i+1}/{len(text_chunks)}")
            # chunk = re.sub(r"\s+", " ", chunk.strip())
            if not chunk:  # clean_chunk
                continue
            audio_out_chunk_data = None
            try:
                audio_out_chunk_data, _, _ = infer(
                    ref_audio_input,
                    ref_text,
                    chunk,  # clean_chunk
                    model,
                    remove_silence,
                    cross_fade_duration=cross_fade_duration,
                    nfe_step=nfe_slider,
                    speed=speed,
                )
            except Exception as e:
                import traceback

                print(f"Inference error: {e}\n{traceback.format_exc()}")
                print(f"Chunk {i+1} error: {e}")
                return None

            if audio_out_chunk_data:
                chunk_sample_rate, wave = audio_out_chunk_data
                if (
                    wave is not None
                    and chunk_sample_rate is not None
                    and hasattr(wave, "size")
                    and wave.size > 0
                ):
                    chunk_filename = os.path.join(temp_dir, f"chunk_{i:05d}.wav")
                    try:
                        if isinstance(wave, torch.Tensor):
                            wave = wave.squeeze().cpu().numpy()
                        if wave.ndim > 1:
                            wave = np.mean(wave, axis=1)
                        if wave.dtype != np.float32 and wave.dtype != np.int16:
                            wave = (
                                wave.astype(np.float32)
                                if np.issubdtype(wave.dtype, np.floating)
                                else wave.astype(np.int16)
                            )
                        if not first_chunk_processed:
                            sample_rate = chunk_sample_rate
                            first_chunk_processed = True
                        sf.write(chunk_filename, wave, chunk_sample_rate)
                        temp_chunk_files.append(chunk_filename)
                        successful_chunks += 1
                    except Exception as e:
                        print(f"Error saving chunk {i+1}: {e}")
            del audio_out_chunk_data, wave
            gc.collect()
            if device == torch.device("cuda"):
                torch.cuda.empty_cache()
        if not temp_chunk_files:
            yield None, "Error: No chunks generated."
            return
        progress(0.8, desc="Stitching audio...")
        list_file_path = os.path.join(temp_dir, "mylist.txt")
        with open(list_file_path, "w", encoding="utf-8") as f:
            for chunk_file_path in temp_chunk_files:
                escaped_path = chunk_file_path.replace("'", "'\\''")
                normalized_path = escaped_path.replace(os.sep, "/")
                f.write(f"file '{normalized_path}'\n")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file_path,
            "-vn",
            "-c:a",
            "libmp3lame",
            "-b:a",
            str(mp3_bitrate),
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            final_mp3_path,
        ]
        try:
            proc = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                check=False,
                encoding="utf-8",
                errors="ignore",
            )
            if proc.returncode != 0:
                yield None, f"ffmpeg error (RC={proc.returncode}): {proc.stderr or proc.stdout}"
                return
        except FileNotFoundError:
            yield None, "Error: ffmpeg not found."
            return
        progress(0.95, desc="Adding metadata...")
        if os.path.exists(final_mp3_path):
            embed_metadata_into_mp3(
                final_mp3_path, extracted_cover_path, ebook_title, ebook_author
            )
        progress(1, desc=f"Completed: {os.path.basename(final_mp3_path)}")
        yield final_mp3_path, None
    except Exception as e:
        import traceback

        yield None, f"Error: {e}\n{traceback.format_exc()}"
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Could not remove temp dir: {e}")
        if converted_epub_path and os.path.exists(converted_epub_path):
            try:
                os.remove(converted_epub_path)
            except Exception as e:
                print(f"Could not remove temp EPUB: {e}")
        if (
            extracted_cover_path
            and os.path.exists(extracted_cover_path)
            and TEMP_CONVERT_DIR in extracted_cover_path
        ):
            try:
                os.remove(extracted_cover_path)
            except Exception as e:
                print(f"Could not remove temp cover: {e}")
        gc.collect()
        if device == torch.device("cuda"):
            torch.cuda.empty_cache()


def batch_process_ebooks(
    ref_audio_input,
    ref_text_input,
    gen_file_inputs,
    cross_fade_duration,
    nfe_slider,
    speed,
    max_chunk_length,
    mp3_bitrate,
    progress=gr.Progress(track_tqdm=True),
):
    if not gen_file_inputs:
        gr.Warning("No eBook files provided.")
        yield None, show_converted_audiobooks()
        return
    if not ref_audio_input and not ref_text_input:
        gr.Error("Reference Audio required if Reference Text not provided.")
        yield None, show_converted_audiobooks()
        return
    ebook_paths = (
        [f.name for f in gen_file_inputs]
        if isinstance(gen_file_inputs, list)
        else [gen_file_inputs.name] if hasattr(gen_file_inputs, "name") else []
    )
    if not ebook_paths:
        gr.Warning("No valid eBook paths.")
        yield None, show_converted_audiobooks()
        return
    processed_paths = []
    last_successful_output_path = None
    errors = []
    for idx, ebook_path in enumerate(ebook_paths):
        print(
            f"Processing eBook {idx+1}/{len(ebook_paths)}: {os.path.basename(ebook_path)}"
        )
        try:
            ebook_processor_gen = process_ebook_to_audio(
                ref_audio_input,
                ref_text_input,
                ebook_path,
                cross_fade_duration,
                nfe_slider,
                speed,
                int(max_chunk_length),
                mp3_bitrate,
                tts_model_choice,
                progress,
            )
            final_path, error_msg = next(ebook_processor_gen)
            if error_msg:
                errors.append(f"'{os.path.basename(ebook_path)}': Failed - {error_msg}")
                yield last_successful_output_path, show_converted_audiobooks()
            elif final_path and os.path.exists(final_path):
                processed_paths.append(final_path)
                last_successful_output_path = final_path
                yield last_successful_output_path, show_converted_audiobooks()
            else:
                errors.append(f"'{os.path.basename(ebook_path)}': No output generated.")
                yield last_successful_output_path, show_converted_audiobooks()
        except Exception as e:
            errors.append(f"'{os.path.basename(ebook_path)}': Error - {e}")
            yield last_successful_output_path, show_converted_audiobooks()
    if errors:
        gr.Warning("Batch processing errors:\n" + "\n".join(errors))
    yield last_successful_output_path, show_converted_audiobooks()


def clear_ref_text_on_audio_change(audio_filepath):
    if audio_filepath and os.path.exists(audio_filepath):
        print(f"Ref audio changed: {os.path.basename(audio_filepath)}")
        return ""
    print("Ref audio cleared.")
    return ""


with gr.Blocks() as audiobooks_tab:
    gr.Markdown("## 📚 eBook to Audiobook Conversion")
    gr.Markdown("Upload eBooks and a voice sample to generate audiobooks.")

    with gr.Row():
        # Left Column: Inputs
        with gr.Column(scale=1, min_width=350):
            gr.Markdown("### Reference Voice")
            ref_audio_input = gr.Audio(
                label="Upload Voice Sample (<15s) or Record",
                sources=["upload", "microphone"],
                type="filepath",
                value=DEFAULT_REF_AUDIO_PATH if default_audio_exists else None,
            )
            ref_text_input = gr.Textbox(
                label="Reference Text (Optional - Leave Blank for Auto-Transcription)",
                lines=3,
                placeholder="Enter EXACT transcript or leave blank...",
                value=DEFAULT_REF_TEXT if default_audio_exists else "",
            )
            gr.Markdown("### Upload eBooks")
            gen_file_input = gr.Files(
                label="Upload eBook File(s)",
                file_types=[
                    ".epub",
                    ".mobi",
                    ".pdf",
                    ".txt",
                    ".html",
                    ".azw3",
                    ".fb2",
                ],
                file_count="multiple",
            )
            gr.Markdown("### Settings")
            speed_slider_book = gr.Slider(
                label="Speech Speed", minimum=0.5, maximum=2.0, value=1.0, step=0.05
            )
            nfe_slider_book = gr.Slider(
                label="NFE Steps",
                minimum=4,
                maximum=64,
                value=32,
                step=2,
                info="Set the number of denoising steps.",
            )
            cross_fade_duration_slider = gr.Slider(
                label="Chunk Cross-Fade (Seconds)",
                minimum=0.0,
                maximum=0.5,
                value=0.0,
                step=0.01,
            )
            max_chunk_length_input = gr.Slider(
                label="Max Text Chunk Length (Characters)",
                minimum=100,
                maximum=5000,
                value=2000,
                step=50,
            )
            mp3_bitrate_input = gr.Dropdown(
                label="Output MP3 Bitrate",
                choices=available_bitrates,
                value=default_bitrate,
                interactive=True,
            )

        # Right Column: Outputs
        with gr.Column(scale=2, min_width=400):
            gr.Markdown("### Generated Audiobooks")
            player = gr.Audio(label="Listen to Latest Audiobook", interactive=False)
            audiobooks_output = gr.Files(
                label="Completed Audiobooks (Download Links)",
                interactive=False,
                file_count="multiple",
            )
            with gr.Row():
                show_audiobooks_btn = gr.Button(
                    "Refresh Audiobook List", variant="secondary"
                )
                generate_btn = gr.Button("Generate Audiobook(s)", variant="primary")

    # Event Listeners
    ref_audio_input.change(
        fn=clear_ref_text_on_audio_change,
        inputs=[ref_audio_input],
        outputs=[ref_text_input],
        queue=False,
    )
    generate_btn.click(
        fn=batch_process_ebooks,
        inputs=[
            ref_audio_input,
            ref_text_input,
            gen_file_input,
            cross_fade_duration_slider,
            nfe_slider_book,
            speed_slider_book,
            max_chunk_length_input,
            mp3_bitrate_input,
        ],
        outputs=[player, audiobooks_output],
    )
    show_audiobooks_btn.click(
        fn=show_converted_audiobooks,
        inputs=[],
        outputs=[audiobooks_output],
        queue=False,
    )
    audiobooks_tab.load(
        fn=show_converted_audiobooks,
        inputs=None,
        outputs=[audiobooks_output],
        queue=False,
    )

# return audiobooks_tab


############################################


with gr.Blocks() as app_credits:
    gr.Markdown(
        """
# Credits

* [mrfakename](https://github.com/fakerybakery) for the original [online demo](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
* [RootingInLoad](https://github.com/RootingInLoad) for initial chunk generation and podcast app exploration
* [jpgallegoar](https://github.com/jpgallegoar) for multiple speech-type generation & voice chat
"""
    )
with gr.Blocks() as app_tts:
    gr.Markdown("# Batched TTS")
    ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
    gen_text_input = gr.Textbox(label="Text to Generate", lines=10)
    generate_btn = gr.Button("Synthesize", variant="primary")
    with gr.Accordion("Advanced Settings", open=False):
        ref_text_input = gr.Textbox(
            label="Reference Text",
            info="Leave blank to automatically transcribe the reference audio. If you enter text it will override automatic transcription.",
            lines=2,
        )
        remove_silence = gr.Checkbox(
            label="Remove Silences",
            info="The model tends to produce silences, especially on longer audio. We can manually remove silences if needed. Note that this is an experimental feature and may produce strange results. This will also increase generation time.",
            value=False,
        )
        speed_slider = gr.Slider(
            label="Speed",
            minimum=0.3,
            maximum=2.0,
            value=1.0,
            step=0.1,
            info="Adjust the speed of the audio.",
        )
        nfe_slider = gr.Slider(
            label="NFE Steps",
            minimum=4,
            maximum=64,
            value=32,
            step=2,
            info="Set the number of denoising steps.",
        )
        cross_fade_duration_slider = gr.Slider(
            label="Cross-Fade Duration (s)",
            minimum=0.0,
            maximum=1.0,
            value=0.0,
            step=0.01,
            info="Set the duration of the cross-fade between audio clips.",
        )

    audio_output = gr.Audio(label="Synthesized Audio")
    spectrogram_output = gr.Image(label="Spectrogram")

    @gpu_decorator
    def basic_tts(
        ref_audio_input,
        ref_text_input,
        gen_text_input,
        remove_silence,
        cross_fade_duration_slider,
        nfe_slider,
        speed_slider,
    ):
        audio_out, spectrogram_path, ref_text_out = infer(
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            tts_model_choice,
            remove_silence,
            cross_fade_duration=cross_fade_duration_slider,
            nfe_step=nfe_slider,
            speed=speed_slider,
        )
        return audio_out, spectrogram_path, ref_text_out

    generate_btn.click(
        basic_tts,
        inputs=[
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            remove_silence,
            cross_fade_duration_slider,
            nfe_slider,
            speed_slider,
        ],
        outputs=[audio_output, spectrogram_output, ref_text_input],
    )


def parse_speechtypes_text(gen_text):
    # Pattern to find {speechtype}
    pattern = r"\{(.*?)\}"

    # Split the text by the pattern
    tokens = re.split(pattern, gen_text)

    segments = []

    current_style = "Regular"

    for i in range(len(tokens)):
        if i % 2 == 0:
            # This is text
            text = tokens[i].strip()
            if text:
                segments.append({"style": current_style, "text": text})
        else:
            # This is style
            style = tokens[i].strip()
            current_style = style

    return segments


with gr.Blocks() as app_multistyle:
    # New section for multistyle generation
    gr.Markdown(
        """
    # Multiple Speech-Type Generation

    This section allows you to generate multiple speech types or multiple people's voices. Enter your text in the format shown below, and the system will generate speech using the appropriate type. If unspecified, the model will use the regular speech type. The current speech type will be used until the next speech type is specified.
    """
    )

    with gr.Row():
        gr.Markdown(
            """
            **Example Input:**                                                                      
            {Regular} Hello, I'd like to order a sandwich please.                                                         
            {Surprised} What do you mean you're out of bread?                                                                      
            {Sad} I really wanted a sandwich though...                                                              
            {Angry} You know what, darn you and your little shop!                                                                       
            {Whisper} I'll just go back home and cry now.                                                                           
            {Shouting} Why me?!                                                                         
            """
        )

        gr.Markdown(
            """
            **Example Input 2:**                                                                                
            {Speaker1_Happy} Hello, I'd like to order a sandwich please.                                                            
            {Speaker2_Regular} Sorry, we're out of bread.                                                                                
            {Speaker1_Sad} I really wanted a sandwich though...                                                                             
            {Speaker2_Whisper} I'll give you the last one I was hiding.                                                                     
            """
        )

    gr.Markdown(
        "Upload different audio clips for each speech type. The first speech type is mandatory. You can add additional speech types by clicking the 'Add Speech Type' button."
    )

    # Regular speech type (mandatory)
    with gr.Row() as regular_row:
        with gr.Column():
            regular_name = gr.Textbox(value="Regular", label="Speech Type Name")
            regular_insert = gr.Button("Insert Label", variant="secondary")
        regular_audio = gr.Audio(label="Regular Reference Audio", type="filepath")
        regular_ref_text = gr.Textbox(label="Reference Text (Regular)", lines=2)

    # Regular speech type (max 100)
    max_speech_types = 100
    speech_type_rows = [regular_row]
    speech_type_names = [regular_name]
    speech_type_audios = [regular_audio]
    speech_type_ref_texts = [regular_ref_text]
    speech_type_delete_btns = [None]
    speech_type_insert_btns = [regular_insert]

    # Additional speech types (99 more)
    for i in range(max_speech_types - 1):
        with gr.Row(visible=False) as row:
            with gr.Column():
                name_input = gr.Textbox(label="Speech Type Name")
                delete_btn = gr.Button("Delete Type", variant="secondary")
                insert_btn = gr.Button("Insert Label", variant="secondary")
            audio_input = gr.Audio(label="Reference Audio", type="filepath")
            ref_text_input = gr.Textbox(label="Reference Text", lines=2)
        speech_type_rows.append(row)
        speech_type_names.append(name_input)
        speech_type_audios.append(audio_input)
        speech_type_ref_texts.append(ref_text_input)
        speech_type_delete_btns.append(delete_btn)
        speech_type_insert_btns.append(insert_btn)

    # Button to add speech type
    add_speech_type_btn = gr.Button("Add Speech Type")

    # Keep track of autoincrement of speech types, no roll back
    speech_type_count = 1

    # Function to add a speech type
    def add_speech_type_fn():
        row_updates = [gr.update() for _ in range(max_speech_types)]
        global speech_type_count
        if speech_type_count < max_speech_types:
            row_updates[speech_type_count] = gr.update(visible=True)
            speech_type_count += 1
        else:
            gr.Warning(
                "Exhausted maximum number of speech types. Consider restart the app."
            )
        return row_updates

    add_speech_type_btn.click(add_speech_type_fn, outputs=speech_type_rows)

    # Function to delete a speech type
    def delete_speech_type_fn():
        return gr.update(visible=False), None, None, None

    # Update delete button clicks
    for i in range(1, len(speech_type_delete_btns)):
        speech_type_delete_btns[i].click(
            delete_speech_type_fn,
            outputs=[
                speech_type_rows[i],
                speech_type_names[i],
                speech_type_audios[i],
                speech_type_ref_texts[i],
            ],
        )

    # Text input for the prompt
    gen_text_input_multistyle = gr.Textbox(
        label="Text to Generate",
        lines=10,
        placeholder="Enter the script with speaker names (or emotion types) at the start of each block, e.g.:\n\n{Regular} Hello, I'd like to order a sandwich please.\n{Surprised} What do you mean you're out of bread?\n{Sad} I really wanted a sandwich though...\n{Angry} You know what, darn you and your little shop!\n{Whisper} I'll just go back home and cry now.\n{Shouting} Why me?!",
    )

    def make_insert_speech_type_fn(index):
        def insert_speech_type_fn(current_text, speech_type_name):
            current_text = current_text or ""
            speech_type_name = speech_type_name or "None"
            updated_text = current_text + f"{{{speech_type_name}}} "
            return updated_text

        return insert_speech_type_fn

    for i, insert_btn in enumerate(speech_type_insert_btns):
        insert_fn = make_insert_speech_type_fn(i)
        insert_btn.click(
            insert_fn,
            inputs=[gen_text_input_multistyle, speech_type_names[i]],
            outputs=gen_text_input_multistyle,
        )

    with gr.Accordion("Advanced Settings", open=False):
        remove_silence_multistyle = gr.Checkbox(
            label="Remove Silences",
            value=True,
        )

    # Generate button
    generate_multistyle_btn = gr.Button(
        "Generate Multi-Style Speech", variant="primary"
    )

    # Output audio
    audio_output_multistyle = gr.Audio(label="Synthesized Audio")

    @gpu_decorator
    def generate_multistyle_speech(
        gen_text,
        *args,
    ):
        speech_type_names_list = args[:max_speech_types]
        speech_type_audios_list = args[max_speech_types : 2 * max_speech_types]
        speech_type_ref_texts_list = args[2 * max_speech_types : 3 * max_speech_types]
        remove_silence = args[3 * max_speech_types]
        # Collect the speech types and their audios into a dict
        speech_types = OrderedDict()

        ref_text_idx = 0
        for name_input, audio_input, ref_text_input in zip(
            speech_type_names_list, speech_type_audios_list, speech_type_ref_texts_list
        ):
            if name_input and audio_input:
                speech_types[name_input] = {
                    "audio": audio_input,
                    "ref_text": ref_text_input,
                }
            else:
                speech_types[f"@{ref_text_idx}@"] = {"audio": "", "ref_text": ""}
            ref_text_idx += 1

        # Parse the gen_text into segments
        segments = parse_speechtypes_text(gen_text)

        # For each segment, generate speech
        generated_audio_segments = []
        current_style = "Regular"

        for segment in segments:
            style = segment["style"]
            text = segment["text"]

            if style in speech_types:
                current_style = style
            else:
                gr.Warning(
                    f"Type {style} is not available, will use Regular as default."
                )
                current_style = "Regular"

            try:
                ref_audio = speech_types[current_style]["audio"]
            except KeyError:
                gr.Warning(f"Please provide reference audio for type {current_style}.")
                return [None] + [
                    speech_types[style]["ref_text"] for style in speech_types
                ]
            ref_text = speech_types[current_style].get("ref_text", "")

            # Generate speech for this segment
            audio_out, _, ref_text_out = infer(
                ref_audio,
                ref_text,
                text,
                tts_model_choice,
                remove_silence,
                0,
                show_info=print,
            )  # show_info=print no pull to top when generating
            sr, audio_data = audio_out

            generated_audio_segments.append(audio_data)
            speech_types[current_style]["ref_text"] = ref_text_out

        # Concatenate all audio segments
        if generated_audio_segments:
            final_audio_data = np.concatenate(generated_audio_segments)
            return [(sr, final_audio_data)] + [
                speech_types[style]["ref_text"] for style in speech_types
            ]
        else:
            gr.Warning("No audio generated.")
            return [None] + [speech_types[style]["ref_text"] for style in speech_types]

    generate_multistyle_btn.click(
        generate_multistyle_speech,
        inputs=[
            gen_text_input_multistyle,
        ]
        + speech_type_names
        + speech_type_audios
        + speech_type_ref_texts
        + [
            remove_silence_multistyle,
        ],
        outputs=[audio_output_multistyle] + speech_type_ref_texts,
    )

    # Validation function to disable Generate button if speech types are missing
    def validate_speech_types(gen_text, regular_name, *args):
        speech_type_names_list = args

        # Collect the speech types names
        speech_types_available = set()
        if regular_name:
            speech_types_available.add(regular_name)
        for name_input in speech_type_names_list:
            if name_input:
                speech_types_available.add(name_input)

        # Parse the gen_text to get the speech types used
        segments = parse_speechtypes_text(gen_text)
        speech_types_in_text = set(segment["style"] for segment in segments)

        # Check if all speech types in text are available
        missing_speech_types = speech_types_in_text - speech_types_available

        if missing_speech_types:
            # Disable the generate button
            return gr.update(interactive=False)
        else:
            # Enable the generate button
            return gr.update(interactive=True)

    gen_text_input_multistyle.change(
        validate_speech_types,
        inputs=[gen_text_input_multistyle, regular_name] + speech_type_names,
        outputs=generate_multistyle_btn,
    )


with gr.Blocks() as app_chat:
    gr.Markdown(
        """
# Voice Chat
Have a conversation with an AI using your reference voice! 
1. Upload a reference audio clip and optionally its transcript.
2. Load the chat model.
3. Record your message through your microphone.
4. The AI will respond using the reference voice.
"""
    )

    chat_model_name_list = [
        "Qwen/Qwen2.5-3B-Instruct",
        "microsoft/Phi-4-mini-instruct",
    ]

    @gpu_decorator
    def load_chat_model(chat_model_name):
        show_info = gr.Info
        global chat_model_state, chat_tokenizer_state
        if chat_model_state is not None:
            chat_model_state = None
            chat_tokenizer_state = None
            gc.collect()
            torch.cuda.empty_cache()

        show_info(f"Loading chat model: {chat_model_name}")
        chat_model_state = AutoModelForCausalLM.from_pretrained(
            chat_model_name, torch_dtype="auto", device_map="auto"
        )
        chat_tokenizer_state = AutoTokenizer.from_pretrained(chat_model_name)
        show_info(f"Chat model {chat_model_name} loaded successfully!")

        return gr.update(visible=False), gr.update(visible=True)

    if USING_SPACES:
        load_chat_model(chat_model_name_list[0])

    chat_model_name_input = gr.Dropdown(
        choices=chat_model_name_list,
        value=chat_model_name_list[0],
        label="Chat Model Name",
        info="Enter the name of a HuggingFace chat model",
        allow_custom_value=not USING_SPACES,
    )
    load_chat_model_btn = gr.Button(
        "Load Chat Model", variant="primary", visible=not USING_SPACES
    )
    chat_interface_container = gr.Column(visible=USING_SPACES)

    chat_model_name_input.change(
        lambda: gr.update(visible=True),
        None,
        load_chat_model_btn,
        show_progress="hidden",
    )
    load_chat_model_btn.click(
        load_chat_model,
        inputs=[chat_model_name_input],
        outputs=[load_chat_model_btn, chat_interface_container],
    )

    with chat_interface_container:
        with gr.Row():
            with gr.Column():
                ref_audio_chat = gr.Audio(label="Reference Audio", type="filepath")
            with gr.Column():
                with gr.Accordion("Advanced Settings", open=False):
                    remove_silence_chat = gr.Checkbox(
                        label="Remove Silences",
                        value=True,
                    )
                    ref_text_chat = gr.Textbox(
                        label="Reference Text",
                        info="Optional: Leave blank to auto-transcribe",
                        lines=2,
                    )
                    system_prompt_chat = gr.Textbox(
                        label="System Prompt",
                        value="You are not an AI assistant, you are whoever the user says you are. You must stay in character. Keep your responses concise since they will be spoken out loud.",
                        lines=2,
                    )

        chatbot_interface = gr.Chatbot(label="Conversation")

        with gr.Row():
            with gr.Column():
                audio_input_chat = gr.Microphone(
                    label="Speak your message",
                    type="filepath",
                )
                audio_output_chat = gr.Audio(autoplay=True)
            with gr.Column():
                text_input_chat = gr.Textbox(
                    label="Type your message",
                    lines=1,
                )
                send_btn_chat = gr.Button("Send Message")
                clear_btn_chat = gr.Button("Clear Conversation")

        conversation_state = gr.State(
            value=[
                {
                    "role": "system",
                    "content": "You are not an AI assistant, you are whoever the user says you are. You must stay in character. Keep your responses concise since they will be spoken out loud.",
                }
            ]
        )

        # Modify process_audio_input to use model and tokenizer from state
        @gpu_decorator
        def process_audio_input(audio_path, text, history, conv_state):
            """Handle audio or text input from user"""

            if not audio_path and not text.strip():
                return history, conv_state, ""

            if audio_path:
                text = preprocess_ref_audio_text(audio_path, text)[1]

            if not text.strip():
                return history, conv_state, ""

            conv_state.append({"role": "user", "content": text})
            history.append((text, None))

            response = generate_response(
                conv_state, chat_model_state, chat_tokenizer_state
            )

            conv_state.append({"role": "assistant", "content": response})
            history[-1] = (text, response)

            return history, conv_state, ""

        @gpu_decorator
        def generate_audio_response(history, ref_audio, ref_text, remove_silence):
            """Generate TTS audio for AI response"""
            if not history or not ref_audio:
                return None

            last_user_message, last_ai_response = history[-1]
            if not last_ai_response:
                return None

            audio_result, _, ref_text_out = infer(
                ref_audio,
                ref_text,
                last_ai_response,
                tts_model_choice,
                remove_silence,
                cross_fade_duration=0.15,
                speed=1.0,
                show_info=print,  # show_info=print no pull to top when generating
            )
            return audio_result, ref_text_out

        def clear_conversation():
            """Reset the conversation"""
            return [], [
                {
                    "role": "system",
                    "content": "You are not an AI assistant, you are whoever the user says you are. You must stay in character. Keep your responses concise since they will be spoken out loud.",
                }
            ]

        def update_system_prompt(new_prompt):
            """Update the system prompt and reset the conversation"""
            new_conv_state = [{"role": "system", "content": new_prompt}]
            return [], new_conv_state

        # Handle audio input
        audio_input_chat.stop_recording(
            process_audio_input,
            inputs=[
                audio_input_chat,
                text_input_chat,
                chatbot_interface,
                conversation_state,
            ],
            outputs=[chatbot_interface, conversation_state],
        ).then(
            generate_audio_response,
            inputs=[
                chatbot_interface,
                ref_audio_chat,
                ref_text_chat,
                remove_silence_chat,
            ],
            outputs=[audio_output_chat, ref_text_chat],
        ).then(
            lambda: None,
            None,
            audio_input_chat,
        )

        # Handle text input
        text_input_chat.submit(
            process_audio_input,
            inputs=[
                audio_input_chat,
                text_input_chat,
                chatbot_interface,
                conversation_state,
            ],
            outputs=[chatbot_interface, conversation_state],
        ).then(
            generate_audio_response,
            inputs=[
                chatbot_interface,
                ref_audio_chat,
                ref_text_chat,
                remove_silence_chat,
            ],
            outputs=[audio_output_chat, ref_text_chat],
        ).then(
            lambda: None,
            None,
            text_input_chat,
        )

        # Handle send button
        send_btn_chat.click(
            process_audio_input,
            inputs=[
                audio_input_chat,
                text_input_chat,
                chatbot_interface,
                conversation_state,
            ],
            outputs=[chatbot_interface, conversation_state],
        ).then(
            generate_audio_response,
            inputs=[
                chatbot_interface,
                ref_audio_chat,
                ref_text_chat,
                remove_silence_chat,
            ],
            outputs=[audio_output_chat, ref_text_chat],
        ).then(
            lambda: None,
            None,
            text_input_chat,
        )

        # Handle clear button
        clear_btn_chat.click(
            clear_conversation,
            outputs=[chatbot_interface, conversation_state],
        )

        # Handle system prompt change and reset conversation
        system_prompt_chat.change(
            update_system_prompt,
            inputs=system_prompt_chat,
            outputs=[chatbot_interface, conversation_state],
        )


with gr.Blocks() as app:
    gr.Markdown(
        f"""
# E2/F5 TTS

This is {"a local web UI for [F5 TTS](https://github.com/SWivid/F5-TTS)" if not USING_SPACES else "an online demo for [F5-TTS](https://github.com/SWivid/F5-TTS)"} with advanced batch processing support. This app supports the following TTS models:

* [F5-TTS](https://arxiv.org/abs/2410.06885) (A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching)
* [E2 TTS](https://arxiv.org/abs/2406.18009) (Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS)

The checkpoints currently support English and Chinese.

If you're having issues, try converting your reference audio to WAV or MP3, clipping it to 12s with  ✂  in the bottom right corner (otherwise might have non-optimal auto-trimmed result).

**NOTE: Reference text will be automatically transcribed with Whisper if not provided. For best results, keep your reference clips short (<12s). Ensure the audio is fully uploaded before generating.**
"""
    )

    last_used_custom = files("f5_tts").joinpath(
        "infer/.cache/last_used_custom_model_info_v1.txt"
    )

    def load_last_used_custom():
        try:
            custom = []
            with open(last_used_custom, "r", encoding="utf-8") as f:
                for line in f:
                    custom.append(line.strip())
            return custom
        except FileNotFoundError:
            last_used_custom.parent.mkdir(parents=True, exist_ok=True)
            return DEFAULT_TTS_MODEL_CFG

    def switch_tts_model(new_choice):
        global tts_model_choice
        if new_choice == "Custom":  # override in case webpage is refreshed
            custom_ckpt_path, custom_vocab_path, custom_model_cfg = (
                load_last_used_custom()
            )
            tts_model_choice = [
                "Custom",
                custom_ckpt_path,
                custom_vocab_path,
                json.loads(custom_model_cfg),
            ]
            return (
                gr.update(visible=True, value=custom_ckpt_path),
                gr.update(visible=True, value=custom_vocab_path),
                gr.update(visible=True, value=custom_model_cfg),
            )
        else:
            tts_model_choice = new_choice
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

    def set_custom_model(custom_ckpt_path, custom_vocab_path, custom_model_cfg):
        global tts_model_choice

        tts_model_choice = [
            "Custom",
            custom_ckpt_path,
            custom_vocab_path,
            json.loads(custom_model_cfg),
        ]
        with open(last_used_custom, "w", encoding="utf-8") as f:
            f.write(
                custom_ckpt_path
                + "\n"
                + custom_vocab_path
                + "\n"
                + custom_model_cfg
                + "\n"
            )

    with gr.Row():

        if not USING_SPACES:
            choose_tts_model = gr.Radio(
                choices=[DEFAULT_TTS_MODEL, "E2-TTS", "Custom"],
                label="Choose TTS Model",
                value=DEFAULT_TTS_MODEL,
            )
        else:
            choose_tts_model = gr.Radio(
                choices=[DEFAULT_TTS_MODEL, "E2-TTS"],
                label="Choose TTS Model",
                value=DEFAULT_TTS_MODEL,
            )

        ch_use_ema = gr.Checkbox(
            label="Use EMA",
            value=True,
            interactive=True,
            info="Turn off at early stage",
        )
        ch_use_ema.change(fn=on_checkbox_change, inputs=ch_use_ema, outputs=[])

        custom_ckpt_path = gr.Dropdown(
            choices=ckpt_choices,  # [DEFAULT_TTS_MODEL_CFG[0]],
            value=load_last_used_custom()[0],
            allow_custom_value=True,
            label="Model: local_path | hf://user_id/repo_id/model_ckpt",
            visible=False,
        )
        custom_vocab_path = gr.Dropdown(
            choices=[DEFAULT_TTS_MODEL_CFG[1]],
            value=load_last_used_custom()[1],
            allow_custom_value=True,
            label="Vocab: local_path | hf://user_id/repo_id/vocab_file",
            visible=False,
        )
        custom_model_cfg = gr.Dropdown(
            choices=[
                DEFAULT_TTS_MODEL_CFG[2],
                json.dumps(
                    dict(
                        dim=1024,
                        depth=22,
                        heads=16,
                        ff_mult=2,
                        text_dim=512,
                        text_mask_padding=False,
                        conv_layers=4,
                        pe_attn_head=1,
                    )
                ),
                json.dumps(
                    dict(
                        dim=768,
                        depth=18,
                        heads=12,
                        ff_mult=2,
                        text_dim=512,
                        text_mask_padding=False,
                        conv_layers=4,
                        pe_attn_head=1,
                    )
                ),
            ],
            value=load_last_used_custom()[2],
            allow_custom_value=True,
            label="Config: in a dictionary form",
            visible=False,
        )

    choose_tts_model.change(
        switch_tts_model,
        inputs=[choose_tts_model],
        outputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_ckpt_path.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_vocab_path.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_model_cfg.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )

    gr.TabbedInterface(
        [app_tts, app_multistyle, audiobooks_tab, app_chat, app_credits],
        ["Basic-TTS", "Multi-Speech", "Audiobook Creator", "Voice-Chat", "Credits"],
    )


def main():
    global app
    print("Starting app...")
    app.queue(api_open=False).launch(
        server_name="0.0.0.0",
        server_port=7866,
        share=True,
        show_api=False,
        root_path=None,
        inbrowser=True,
    )


if __name__ == "__main__":
    if not USING_SPACES:
        main()
    else:
        app.queue().launch()
