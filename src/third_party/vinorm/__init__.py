# import subprocess, os
# import imp

### verson 1
# def TTSnorm(text, punc=False, unknown=False, lower=True, rule=False):
#     A = imp.find_module("vinorm")[1]

#     # print(A)
#     I = A + "/input.txt"
#     with open(I, "w+", encoding="utf-8") as fw:
#         fw.write(text)

#     myenv = os.environ.copy()
#     myenv["LD_LIBRARY_PATH"] = A + "/lib"

#     E = A + "/main"
#     Command = [E]
#     if punc:
#         Command.append("-punc")
#     if unknown:
#         Command.append("-unknown")
#     if lower:
#         Command.append("-lower")
#     if rule:
#         Command.append("-rule")
#     subprocess.check_call(Command, env=myenv, cwd=A)

#     O = A + "/output.txt"
#     with open(O, "r", encoding="utf-8") as fr:
#         text = fr.read()
#     TEXT = ""
#     S = text.split("#line#")
#     for s in S:
#         if s == "":
#             continue
#         # TEXT+=s+". "
#         TEXT += s

#     return TEXT


import subprocess
import os
import uuid
import time
from fastapi import FastAPI
from threading import Lock
import unicodedata
import logging
import fcntl
import shutil
from multiprocessing import Pool

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def to_normalized_unicode(text):
    return unicodedata.normalize("NFC", text)


app = FastAPI()

# Configuration
MAX_TEMP_DIRS = 10  # Increased to handle higher concurrency
BASE_DIR = os.path.dirname(__file__)
TEMP_DIRS = [os.path.join(BASE_DIR, f"temp_dir_{i}") for i in range(MAX_TEMP_DIRS)]
LOCK_FILES = [os.path.join(dir_path, "lockfile") for dir_path in TEMP_DIRS]
WAITING_TIME = 0.1  # Time to wait (seconds) if all directories are busy

# Required directories to symlink
REQUIRED_DIRS = ["RegexRule", "Mapping", "Dict", "lib"]

# Lock for managing directory access
availability_lock = Lock()
# Availability status of directories (True = available, False = busy)
temp_dir_status = [True] * MAX_TEMP_DIRS
# Track last assigned directory for round-robin
last_assigned_dir = -1


# Initialize temporary directories
def initialize_temp_dirs():
    main_path = os.path.join(BASE_DIR, "main")
    if not os.path.exists(main_path):
        raise FileNotFoundError(f"File 'main' not found in {BASE_DIR}")

    for i, temp_dir in enumerate(TEMP_DIRS):
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            # Create symlink for main
            main_link = os.path.join(temp_dir, "main")
            os.symlink(main_path, main_link)
            # Create symlinks for required directories
            for req_dir in REQUIRED_DIRS:
                src_dir = os.path.join(BASE_DIR, req_dir)
                dst_dir = os.path.join(temp_dir, req_dir)
                if os.path.exists(src_dir):
                    os.symlink(src_dir, dst_dir)
                else:
                    logger.warning(f"Directory {src_dir} not found, skipping...")

        lock_file = LOCK_FILES[i]
        if not os.path.exists(lock_file):
            with open(lock_file, "a"):
                pass  # Create lock file if it doesn't exist


def get_available_temp_dir():
    global last_assigned_dir
    while True:
        with availability_lock:
            # Start from the next directory after the last assigned one
            start_index = (last_assigned_dir + 1) % MAX_TEMP_DIRS
            for i in range(MAX_TEMP_DIRS):
                index = (start_index + i) % MAX_TEMP_DIRS
                if temp_dir_status[index]:
                    temp_dir_status[index] = False  # Mark as busy
                    last_assigned_dir = index  # Update last assigned directory
                    logger.info(f"Assigned temp_dir_{index}")
                    return index
            logger.warning("All directories busy, waiting...")
        time.sleep(WAITING_TIME)


def release_temp_dir(index):
    with availability_lock:
        temp_dir_status[index] = True  # Mark as available
        logger.info(f"Released temp_dir_{index}")


def TTSnorm(text, punc=False, unknown=False, lower=True, rule=False):
    """
    lower: If true, get normalization with lowercase
    rule: If true, just get normalization with Regex, not using Dictionary Checking
    punc: If true, do not replace punctuation with dot and coma
    unknown: If true, replace unknown word, discard word undefined and do not contain vowel
    """
    temp_dir_index = get_available_temp_dir()
    temp_dir = TEMP_DIRS[temp_dir_index]
    lock_file = LOCK_FILES[temp_dir_index]
    text = to_normalized_unicode(text)

    # Generate unique file names for internal use
    unique_id = str(uuid.uuid4())
    temp_input_path = os.path.join(temp_dir, f"temp_input_{unique_id}.txt")
    temp_output_path = os.path.join(temp_dir, f"temp_output_{unique_id}.txt")
    fixed_input_path = os.path.join(temp_dir, "input.txt")
    fixed_output_path = os.path.join(temp_dir, "output.txt")

    try:
        # Lock file to ensure exclusive access to directory
        with open(lock_file, "a") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)  # Exclusive lock
            logger.info(f"Locked temp_dir_{temp_dir_index} for processing {unique_id}")

            # Write to unique temporary input file
            with open(temp_input_path, "w+", encoding="utf-8") as fw:
                fw.write(text)

            # Copy to fixed input.txt
            shutil.copy(temp_input_path, fixed_input_path)

            myenv = os.environ.copy()
            myenv["LD_LIBRARY_PATH"] = os.path.join(temp_dir, "lib")

            E = os.path.join(temp_dir, "main")
            Command = [E]
            if punc:
                Command.append("-punc")
            if unknown:
                Command.append("-unknown")
            if lower:
                Command.append("-lower")
            if rule:
                Command.append("-rule")

            logger.info(f"Running command in temp_dir_{temp_dir_index}: {Command}")
            subprocess.check_call(Command, env=myenv, cwd=temp_dir)
            time.sleep(1)  # Simulate processing delay (remove if unnecessary)

            # Copy output.txt to unique temporary output file
            shutil.copy(fixed_output_path, temp_output_path)

            with open(temp_output_path, "r", encoding="utf-8") as fr:
                text = fr.read()

            TEXT = ""
            S = text.split("#line#")
            for s in S:
                if s == "":
                    continue
                TEXT += s + ". "

            fcntl.flock(lf, fcntl.LOCK_UN)  # Unlock
            logger.info(f"Unlocked temp_dir_{temp_dir_index} for {unique_id}")

        return TEXT

    finally:
        # Clean up all temporary and fixed files
        try:
            for file_path in [
                temp_input_path,
                temp_output_path,
                fixed_input_path,
                fixed_output_path,
            ]:
                if os.path.exists(file_path):
                    os.remove(file_path)
            logger.info(
                f"Cleaned up files for {unique_id} in temp_dir_{temp_dir_index}"
            )
        except Exception as e:
            logger.error(f"Error cleaning up files for {unique_id}: {e}")
        release_temp_dir(temp_dir_index)


def worker(args):
    text = args
    return TTSnorm(text)


@app.get("/tts-norm")
async def tts_norm(
    text: str,
    punc: bool = False,
    unknown: bool = False,
    lower: bool = True,
    rule: bool = False,
):
    result = TTSnorm(text, punc, unknown, lower, rule)
    return {"result": result}


# Enhanced test for concurrent directory usage
def test_concurrent_dirs():
    initialize_temp_dirs()
    texts = [
        f"Test text {i} for concurrent processing" for i in range(10)  # More tasks
    ]
    logger.info("Starting concurrent test with 10 tasks")
    with Pool(processes=5) as pool:  # Use more processes to stress the system
        results = pool.map(worker, texts)
    logger.info("Concurrent test completed")
    return results


# Initialize directories on startup
initialize_temp_dirs()

if __name__ == "__main__":
    # Run the concurrent test
    results = test_concurrent_dirs()
    for i, result in enumerate(results):
        logger.info(f"Result {i}: {result}")

    # Original test
    full_text = """
    Chủ tịch VINASA FPT IOT; Nguyễn Văn Khoa CNTT KHCN AI (trung bình)
    50% GDP, Tổng giám đốc FPT IoT Iot a b c 7 USD TS ai đó
    """
    full_result = TTSnorm(full_text, punc=False, unknown=False, lower=True, rule=False)
    logger.info(f"Full text result: {full_result}")
