# F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching

[![python](https://img.shields.io/badge/Python-3.10-brightgreen)](https://github.com/SWivid/F5-TTS)
[![arXiv](https://img.shields.io/badge/arXiv-2410.06885-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.06885)
[![demo](https://img.shields.io/badge/GitHub-Demo%20page-orange.svg)](https://swivid.github.io/F5-TTS/)
[![hfspace](https://img.shields.io/badge/🤗-Space%20demo-yellow)](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
[![msspace](https://img.shields.io/badge/🤖-Space%20demo-blue)](https://modelscope.cn/studios/modelscope/E2-F5-TTS)
[![lab](https://img.shields.io/badge/X--LANCE-Lab-grey?labelColor=lightgrey)](https://x-lance.sjtu.edu.cn/)
[![lab](https://img.shields.io/badge/Peng%20Cheng-Lab-grey?labelColor=lightgrey)](https://www.pcl.ac.cn)
<!-- <img src="https://github.com/user-attachments/assets/12d7749c-071a-427c-81bf-b87b91def670" alt="Watermark" style="width: 40px; height: auto"> -->

**F5-TTS**: Diffusion Transformer with ConvNeXt V2, faster trained and inference.

**E2 TTS**: Flat-UNet Transformer, closest reproduction from [paper](https://arxiv.org/abs/2406.18009).

**Sway Sampling**: Inference-time flow step sampling strategy, greatly improves performance

### Thanks to all the contributors !

## News
- **2025/03/12**: 🔥 F5-TTS v1 base model with better training and inference performance. [Few demo](https://swivid.github.io/F5-TTS_updates).
- **2024/10/08**: F5-TTS & E2 TTS base models on [🤗 Hugging Face](https://huggingface.co/SWivid/F5-TTS), [🤖 Model Scope](https://www.modelscope.cn/models/SWivid/F5-TTS_Emilia-ZH-EN), [🟣 Wisemodel](https://wisemodel.cn/models/SJTU_X-LANCE/F5-TTS_Emilia-ZH-EN).

## Installation

### Create a separate environment if needed

```bash
# Create a python 3.10 conda env (you could also use virtualenv)
conda create -n f5-tts python=3.10
conda activate f5-tts
```

### Install PyTorch with matched device

<details>
<summary>NVIDIA GPU</summary>

> ```bash
> # Install pytorch with your CUDA version, e.g.
> pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
> ```

</details>

<details>
<summary>AMD GPU</summary>

> ```bash
> # Install pytorch with your ROCm version (Linux only), e.g.
> pip install torch==2.5.1+rocm6.2 torchaudio==2.5.1+rocm6.2 --extra-index-url https://download.pytorch.org/whl/rocm6.2
> ```

</details>

<details>
<summary>Intel GPU</summary>

> ```bash
> # Install pytorch with your XPU version, e.g.
> # Intel® Deep Learning Essentials or Intel® oneAPI Base Toolkit must be installed
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/test/xpu
> 
> # Intel GPU support is also available through IPEX (Intel® Extension for PyTorch)
> # IPEX does not require the Intel® Deep Learning Essentials or Intel® oneAPI Base Toolkit
> # See: https://pytorch-extension.intel.com/installation?request=platform
> ```

</details>

<details>
<summary>Apple Silicon</summary>

> ```bash
> # Install the stable pytorch, e.g.
> pip install torch torchaudio
> ```

</details>

### Then you can choose one from below:

> ### 1. As a pip package (if just for inference)
> 
> ```bash
> pip install f5-tts
> ```
> 
> ### 2. Local editable (if also do training, finetuning)
> 
> ```bash
> git clone https://github.com/SWivid/F5-TTS.git
> cd F5-TTS
> # git submodule update --init --recursive  # (optional, if need > bigvgan)
> pip install -e .
> ```

### Docker usage also available
```bash
# Build from Dockerfile
docker build -t f5tts:v1 .

# Run from GitHub Container Registry
docker container run --rm -it --gpus=all --mount 'type=volume,source=f5-tts,target=/root/.cache/huggingface/hub/' -p 7860:7860 ghcr.io/swivid/f5-tts:main

# Quickstart if you want to just run the web interface (not CLI)
docker container run --rm -it --gpus=all --mount 'type=volume,source=f5-tts,target=/root/.cache/huggingface/hub/' -p 7860:7860 ghcr.io/swivid/f5-tts:main f5-tts_infer-gradio --host 0.0.0.0
```

### Runtime

Deployment solution with Triton and TensorRT-LLM.

#### Benchmark Results
Decoding on a single L20 GPU, using 26 different prompt_audio & target_text pairs, 16 NFE.

| Model               | Concurrency    | Avg Latency | RTF    | Mode            |
|---------------------|----------------|-------------|--------|-----------------|
| F5-TTS Base (Vocos) | 2              | 253 ms      | 0.0394 | Client-Server   |
| F5-TTS Base (Vocos) | 1 (Batch_size) | -           | 0.0402 | Offline TRT-LLM |
| F5-TTS Base (Vocos) | 1 (Batch_size) | -           | 0.1467 | Offline Pytorch |

See [detailed instructions](src/f5_tts/runtime/triton_trtllm/README.md) for more information.


## Inference

- In order to achieve desired performance, take a moment to read [detailed guidance](src/f5_tts/infer).
- By properly searching the keywords of problem encountered, [issues](https://github.com/SWivid/F5-TTS/issues?q=is%3Aissue) are very helpful.

### 1. Gradio App

Currently supported features:

- Basic TTS with Chunk Inference
- Multi-Style / Multi-Speaker Generation
- Voice Chat powered by Qwen2.5-3B-Instruct
- [Custom inference with more language support](src/f5_tts/infer/SHARED.md)

```bash
# Launch a Gradio app (web interface)
f5-tts_infer-gradio

# Specify the port/host
f5-tts_infer-gradio --port 7860 --host 0.0.0.0

# Launch a share link
f5-tts_infer-gradio --share
```

<details>
<summary>NVIDIA device docker compose file example</summary>

```yaml
services:
  f5-tts:
    image: ghcr.io/swivid/f5-tts:main
    ports:
      - "7860:7860"
    environment:
      GRADIO_SERVER_PORT: 7860
    entrypoint: ["f5-tts_infer-gradio", "--port", "7860", "--host", "0.0.0.0"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  f5-tts:
    driver: local
```

</details>

### 2. CLI Inference

```bash
# Run with flags
# Leave --ref_text "" will have ASR model transcribe (extra GPU memory usage)
f5-tts_infer-cli --model F5TTS_v1_Base \
--ref_audio "provide_prompt_wav_path_here.wav" \
--ref_text "The content, subtitle or transcription of reference audio." \
--gen_text "Some text you want TTS model generate for you."

# Run with default setting. src/f5_tts/infer/examples/basic/basic.toml
f5-tts_infer-cli
# Or with your own .toml file
f5-tts_infer-cli -c custom.toml

# Multi voice. See src/f5_tts/infer/README.md
f5-tts_infer-cli -c src/f5_tts/infer/examples/multi/story.toml
```


## Training

### 1. With Hugging Face Accelerate

Refer to [training & finetuning guidance](src/f5_tts/train) for best practice.

### 2. With Gradio App

```bash
# Quick start with Gradio web interface
f5-tts_finetune-gradio
```

Read [training & finetuning guidance](src/f5_tts/train) for more instructions.


## [Evaluation](src/f5_tts/eval)


## Development

Use pre-commit to ensure code quality (will run linters and formatters automatically):

```bash
pip install pre-commit
pre-commit install
```

When making a pull request, before each commit, run: 

```bash
pre-commit run --all-files
```

Note: Some model components have linting exceptions for E722 to accommodate tensor notation.


## Acknowledgements

- [E2-TTS](https://arxiv.org/abs/2406.18009) brilliant work, simple and effective
- [Emilia](https://arxiv.org/abs/2407.05361), [WenetSpeech4TTS](https://arxiv.org/abs/2406.05763), [LibriTTS](https://arxiv.org/abs/1904.02882), [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) valuable datasets
- [lucidrains](https://github.com/lucidrains) initial CFM structure with also [bfs18](https://github.com/bfs18) for discussion
- [SD3](https://arxiv.org/abs/2403.03206) & [Hugging Face diffusers](https://github.com/huggingface/diffusers) DiT and MMDiT code structure
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq) as ODE solver, [Vocos](https://huggingface.co/charactr/vocos-mel-24khz) and [BigVGAN](https://github.com/NVIDIA/BigVGAN) as vocoder
- [FunASR](https://github.com/modelscope/FunASR), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [UniSpeech](https://github.com/microsoft/UniSpeech), [SpeechMOS](https://github.com/tarepan/SpeechMOS) for evaluation tools
- [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) for speech edit test
- [mrfakename](https://x.com/realmrfakename) huggingface space demo ~
- [f5-tts-mlx](https://github.com/lucasnewman/f5-tts-mlx/tree/main) Implementation with MLX framework by [Lucas Newman](https://github.com/lucasnewman)
- [F5-TTS-ONNX](https://github.com/DakeQQ/F5-TTS-ONNX) ONNX Runtime version by [DakeQQ](https://github.com/DakeQQ)
- [Yuekai Zhang](https://github.com/yuekaizhang) Triton and TensorRT-LLM support ~

## Citation
If our work and codebase is useful for you, please cite as:
```
@article{chen-etal-2024-f5tts,
      title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching}, 
      author={Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang and Jian Zhao and Kai Yu and Xie Chen},
      journal={arXiv preprint arXiv:2410.06885},
      year={2024},
}
```
## License

Our code is released under MIT License. The pre-trained models are licensed under the CC-BY-NC license due to the training data Emilia, which is an in-the-wild dataset. Sorry for any inconvenience this may cause.



## setups



ln -s /media/anhlbt/Book/datasets/datasets_voice/VOICE_TASK/vivos_edited/* vivos_char
unlink 

_letters =  '0123456789aáảàãạâấẩầẫậăắẳằẵặbcdđeéẻèẽẹêếểềễệfghiíỉìĩịjklmnoóỏòõọôốổồỗộơớởờỡợpqrstuúủùũụưứửừữựvwxyýỷỳỹỵz,.:'

vocab = 0,1,2,3,4,5,6,7,8,9,a,á,ả,à,ã,ạ,â,ấ,ẩ,ầ,ẫ,ậ,ă,ắ,ẳ,ằ,ẵ,ặ,b,c,d,đ,e,é,ẻ,è,ẽ,ẹ,ê,ế,ể,ề,ễ,ệ,f,g,h,i,í,ỉ,ì,ĩ,ị,j,k,l,m,n,o,ó,ỏ,ò,õ,ọ,ô,ố,ổ,ồ,ỗ,ộ,ơ,ớ,ở,ờ,ỡ,ợ,p,q,r,s,t,u,ú,ủ,ù,ũ,ụ,ư,ứ,ử,ừ,ữ,ự,v,w,x,y,ý,ỷ,ỳ,ỹ,ỵ,z,,,.,:

install deepspeed==0.14.4

## vivoice_plus
ln -s /media/anhlbt/SSD2/viVoice/* /media/anhlbt/SSD2/workspace/VOICE_TASK/F5-TTS/data/vivoice_char

ln -s /media/anhlbt/SSD2/viVoice/* /media/anhlbt/SSD2/workspace/VOICE_TASK/F5-TTS/data/vivoicebase_char

## anhlbt_tts
ln -s /media/anhlbt/Book1/datasets/datasets_voice/VOICE_TASK/data_BaomoiCrawler/data/anhlbt_tts/* /media/anhlbt/SSD2/workspace/VOICE_TASK/F5-TTS/data/anhlbt_tts_char

## ljspeech_dataset
ln -s /media/anhlbt/SSD2/viVoice/* /media/anhlbt/SSD2/workspace/VOICE_TASK/F5-TTS/data/vivoice_char

# # Liên kết tất cả file từ nguồn 1
ln -s /media/anhlbt/Book/train_bert/wavs/*.wav /media/anhlbt/SSD2/workspace/VOICE_TASK/F5-TTS/data/vivoice_plus/wavs/
# # Liên kết tất cả file từ nguồn viVoice
ln -s /media/anhlbt/SSD2/viVoice/*.wav /media/anhlbt/SSD2/workspace/VOICE_TASK/F5-TTS/data/vivoice_plus/wavs/


```bash
for file in /media/anhlbt/SSD2/viVoice/wavs/*.wav; do
    echo "$file"
    ln -sf "$file" /media/anhlbt/SSD2/workspace/VOICE_TASK/F5-TTS/data/vivoice_plus/wavs/
done


```




```bash
curl -X POST "http://0.0.0.0:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
         "ref_file": "src/f5_tts/infer/examples/basic/basic_ref_en.wav",
         "ref_text": "Some call me nature, others call me mother nature.",
         "gen_text": "I don\u0027t really care what you call me. I\u0027ve been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring.",
         "remove_silence": false
     }'
```


### Random Seed
3510892055103586970

8184481127414169600
2203761066509910022