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
import tempfile
import shutil
from multiprocessing import Pool

import subprocess
import os
import fcntl
import time
from fastapi import FastAPI
from threading import Lock
import shutil
import unicodedata


def to_normalized_unicode(text):
    return unicodedata.normalize("NFC", text)


app = FastAPI()

# Cấu hình
MAX_TEMP_DIRS = 5
BASE_DIR = os.path.dirname(__file__)
TEMP_DIRS = [os.path.join(BASE_DIR, f"temp_dir_{i}") for i in range(MAX_TEMP_DIRS)]
LOCK_FILES = [os.path.join(dir_path, "lockfile") for dir_path in TEMP_DIRS]
WAITING_TIME = 0.1  # Thời gian chờ (giây) nếu tất cả thư mục bận

# Các thư mục cần sao chép
REQUIRED_DIRS = ["RegexRule", "Mapping", "Dict", "lib"]

# Khóa để quản lý truy cập danh sách thư mục
availability_lock = Lock()
# Trạng thái sẵn sàng của các thư mục (True = sẵn sàng, False = đang bận)
temp_dir_status = [True] * MAX_TEMP_DIRS


# Tạo sẵn các thư mục tạm và sao chép file main cùng các thư mục cần thiết
def initialize_temp_dirs():
    main_path = os.path.join(BASE_DIR, "main")
    if not os.path.exists(main_path):
        raise FileNotFoundError(f"File 'main' not found in {BASE_DIR}")

    for i, temp_dir in enumerate(TEMP_DIRS):
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            # Tạo softlink cho file main
            main_link = os.path.join(temp_dir, "main")
            os.symlink(main_path, main_link)
            # Tạo softlink cho các thư mục cần thiết
            for req_dir in REQUIRED_DIRS:
                src_dir = os.path.join(BASE_DIR, req_dir)
                dst_dir = os.path.join(temp_dir, req_dir)
                if os.path.exists(src_dir):
                    os.symlink(src_dir, dst_dir)
                else:
                    print(f"Warning: Directory {src_dir} not found, skipping...")

        lock_file = LOCK_FILES[i]
        if not os.path.exists(lock_file):
            with open(lock_file, "a"):
                pass  # Tạo file lock nếu chưa tồn tại


def get_available_temp_dir():
    while True:
        with availability_lock:
            for i in range(MAX_TEMP_DIRS):
                if temp_dir_status[i]:
                    temp_dir_status[i] = False  # Đánh dấu là đang sử dụng
                    return i
        # Nếu không có thư mục nào sẵn sàng, chờ một chút rồi thử lại
        time.sleep(WAITING_TIME)


def release_temp_dir(index):
    with availability_lock:
        temp_dir_status[index] = True  # Đánh dấu là sẵn sàng


def TTSnorm(text, punc=False, unknown=False, lower=True, rule=False):
    """
    lower: If true, get normalization with lowercase
    rule: If true, just get normalization wit Regex, not using Dictionary Checking (this flag is not used with another flag)
    punc: If true, do not replace punctuation with dot and coma
    unknown: If true, replace unknown word, discard word undefine and do not contain vowel, do not spell word with vowel
    """
    # Lấy một thư mục tạm sẵn sàng
    temp_dir_index = get_available_temp_dir()
    temp_dir = TEMP_DIRS[temp_dir_index]
    lock_file = LOCK_FILES[temp_dir_index]
    text = to_normalized_unicode(text)

    try:
        # Khóa file để đảm bảo chỉ một tiến trình sử dụng thư mục này
        with open(lock_file, "a") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)  # Khóa độc quyền

            input_path = os.path.join(temp_dir, "input.txt")
            output_path = os.path.join(temp_dir, "output.txt")

            with open(input_path, "w+", encoding="utf-8") as fw:
                fw.write(text)

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

            subprocess.check_call(Command, env=myenv, cwd=temp_dir)

            with open(output_path, "r", encoding="utf-8") as fr:
                text = fr.read()

            # TEXT = "".join(s for s in text.split("#line#") if s)
            TEXT = ""
            S = text.split("#line#")
            for s in S:
                if s == "":
                    continue
                TEXT += s + ". "
                # TEXT += s

            fcntl.flock(lf, fcntl.LOCK_UN)  # Mở khóa

        return TEXT

    finally:
        # Giải phóng thư mục để sử dụng lại
        release_temp_dir(temp_dir_index)


def worker(args):
    text = args
    return TTSnorm(text)


## test api
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


initialize_temp_dirs()

if __name__ == "__main__":
    # Khởi tạo các thư mục tạm trước khi chạy API
    initialize_temp_dirs()
    # import uvicorn

    # uvicorn.run(app, host="0.0.0.0", port=8000)

    texts = ["what's your name", "Lập trình song song 2", "Python multiprocessing 33"]
    # Tạo danh sách tuples chứa text và uuid duy nhất cho mỗi text
    tasks = [text for text in texts]
    with Pool(processes=3) as pool:
        results = pool.map(worker, tasks)
    print(results)
