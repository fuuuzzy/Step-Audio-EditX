# Step-Audio-EditX
<p align="center">
  <img src="assets/logo.png"  height=100>
</p>

<div align="center">
    <a href="https://stepaudiollm.github.io/step-audio-editx/"><img src="https://img.shields.io/static/v1?label=Demo%20Page&message=Web&color=green"></a> &ensp;
  <a href="https://arxiv.org/abs/2511.03601"><img src="https://img.shields.io/static/v1?label=Tech%20Report&message=Arxiv&color=red"></a> &ensp;
  <a href="https://huggingface.co/stepfun-ai/Step-Audio-EditX"><img src="https://img.shields.io/static/v1?label=Step-Audio-EditX&message=HuggingFace&color=yellow"></a> &ensp;
  <a href="https://huggingface.co/spaces/stepfun-ai/Step-Audio-EditX"><img src="https://img.shields.io/static/v1?label=Space%20Playground&message=HuggingFace&color=yellow"></a> &ensp;
</div>

## 🔥🔥🔥 News!!
* Nov 07, 2025: ✨ [Demo Page](https://stepaudiollm.github.io/step-audio-editx/) ; 🎮  [HF Space Playground](https://huggingface.co/spaces/stepfun-ai/Step-Audio-EditX)
* Nov 06, 2025: 👋 We release the technical report of [Step-Audio-EditX](https://arxiv.org/abs/2511.03601).

## Introduction
We are open-sourcing Step-Audio-EditX, a powerful **3B parameters** LLM-based audio model specialized in expressive and iterative audio editing. It excels at editing emotion, speaking style, and paralinguistics, and also features robust zero-shot text-to-speech (TTS) capabilities. 

## 📑 Open-source Plan
- [x] Inference Code
- [x] Online demo (Gradio)
- [ ] Step-Audio-Edit-Benchmark
- [x] Model Checkpoints
  - [x] Step-Audio-Tokenizer
  - [x] Step-Audio-EditX
  - [ ] Step-Audio-EditX-Int8
- [ ] Training Code
  - [ ] SFT training
  - [ ] PPO training

## Features
- **Zero-Shot TTS**
  - Excellent zero-shot TTS cloning for Mandarin, English, Sichuanese, and Cantonese.
  - To use a dialect, just add a **[Sichuanese]** or **[Cantonese]** tag before your text.
 
    
- **Emotion and Speaking Style Editing**
  - Remarkably effective iterative control over emotions and styles, supporting **dozens** of options for editing.
    - Emotion Editing : [ *Angry*, *Happy*, *Sad*, *Excited*, *Fearful*, *Surprised*, *Disgusted*, etc. ]
    - Speaking Style Editing: [ *Act_coy*, *Older*, *Child*, *Whisper*, *Serious*, *Generous*, *Exaggerated*, etc.]
    - Editing with more emotion and more speaking styles is on the way. **Get Ready!** 🚀 
    

- **Paralinguistic Editing**:
  -  Precise control over 10 types of paralinguistic features for more natural, human-like, and expressive synthetic audio.
  - Supporting Tags:
    - [ *Breathing*, *Laughter*, *Suprise-oh*, *Confirmation-en*, *Uhm*, *Suprise-ah*, *Suprise-wa*, *Sigh*, *Question-ei*, *Dissatisfaction-hnn* ]


## Demos

<table>
  <tr>
    <th style="vertical-align : middle;text-align: center">Task</th>
    <th style="vertical-align : middle;text-align: center">Text</th>
    <th style="vertical-align : middle;text-align: center">Source</th>
    <th style="vertical-align : middle;text-align: center">Edited</th>
  </tr>

  <tr>
    <td align="center"> Emotion-Fear</td>
    <td align="center"> 我总觉得，有人在跟着我，我能听到奇怪的脚步声。</td>
    <td align="center">

  [fear_zh_female_prompt.webm](https://github.com/user-attachments/assets/a088c059-032c-423f-81d6-3816ba347ff5) 
  </td>
    <td align="center">
      
  [fear_zh_female_output.webm](https://github.com/user-attachments/assets/917494ac-5913-4949-8022-46cf55ca05dd)
  </td>
  </tr>


  <tr>
    <td align="center"> Style-Whisper</td>
    <td align="center"> 比如在工作间隙，做一些简单的伸展运动，放松一下身体，这样，会让你更有精力。</td>
    <td align="center">
      
  [whisper_prompt.webm](https://github.com/user-attachments/assets/ed9e22f1-1bac-417b-913a-5f1db31f35c9)
  </td>
    <td align="center">
      
  [whisper_output.webm](https://github.com/user-attachments/assets/e0501050-40db-4d45-b380-8bcc309f0b5f)
  </td>
  </tr>


  <tr>
    <td align="center"> Paralinguistics</td>
    <td align="center"> 你这次又忘记带钥匙了 [Dissatisfaction-hnn]，真是拿你没办法。</td>
    <td align="center">
      
  [paralingustic_prompt.webm](https://github.com/user-attachments/assets/21e831a3-8110-4c64-a157-60e0cf6735f0)
  </td>
    <td align="center">
      
  [paralingustic_output.webm](https://github.com/user-attachments/assets/a82f5a40-c6a3-409b-bbe6-271180b20d7b)
  </td>
  </tr>


  <tr>
    <td align="center"> Denoising</td>
    <td align="center"> Such legislation was clarified and extended from time to time thereafter. No, the man was not drunk, he wondered how we got tied up with this stranger. Suddenly, my reflexes had gone. It's healthier to cook without sugar.</td>
    <td align="center">
      
  [denoising_prompt.webm](https://github.com/user-attachments/assets/70464bf4-ebde-44a3-b2a6-8c292333319b)
  </td>
    <td align="center">
      
  [denoising_output.webm](https://github.com/user-attachments/assets/7cd0ae8d-1bf0-40fc-9bcd-f419bd4b2d21)
  </td>
  </tr>

  <tr>
    <td align="center"> Speed-Faster</td>
    <td align="center"> 上次你说鞋子有点磨脚，我给你买了一双软软的鞋垫。</td>
    <td align="center">
      
  [speed_faster_prompt.webm](https://github.com/user-attachments/assets/db46609e-1b98-48d8-99c8-e166cfdfc6e3)
  </td>
    <td align="center">
      
  [speed_faster_output.webm](https://github.com/user-attachments/assets/0fbc14ca-dd4a-4362-aadc-afe0629f4c9f)
  </td>
  </tr>
  
</table>


For more examples, see [demo page](https://stepaudiollm.github.io/step-audio-editx/).


## Model Download

| Models   | 🤗 Hugging Face | ModelScope |
|-------|-------|-------|
| Step-Audio-EditX | [stepfun-ai/Step-Audio-EditX](https://huggingface.co/stepfun-ai/Step-Audio-EditX) | [stepfun-ai/Step-Audio-EditX](https://modelscope.cn/models/stepfun-ai/Step-Audio-EditX) |
| Step-Audio-Tokenizer | [stepfun-ai/Step-Audio-Tokenizer](https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer) | [stepfun-ai/Step-Audio-Tokenizer](https://modelscope.cn/models/stepfun-ai/Step-Audio-Tokenizer) |


## Model Usage
### 📜 Requirements
The following table shows the requirements for running Step-Audio-EditX model (batch size = 1):

|     Model    | Parameters |  Setting<br/>(sample frequency) | GPU Optimal Memory  |
|------------|------------|--------------------------------|----------------|
| Step-Audio-EditX   | 3B|         41.6Hz          |       12 GB        |

* An NVIDIA GPU with CUDA support is required.
  * The model is tested on a single L40S GPU.
* Tested operating system: Linux

### 🔧 Dependencies and Installation
- Python >= 3.10.0 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.4.1-cu121](https://pytorch.org/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

```bash
git clone https://github.com/stepfun-ai/Step-Audio-EditX.git
conda create -n stepaudioedit python=3.10
conda activate stepaudioedit

cd Step-Audio-EditX
pip install -r requirements.txt

git lfs install
git clone https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer
git clone https://huggingface.co/stepfun-ai/Step-Audio-EditX

```

After downloading the models, where_you_download_dir should have the following structure:
```
where_you_download_dir
├── Step-Audio-Tokenizer
├── Step-Audio-EditX
```

#### Run with Docker

You can set up the environment required for running Step-Audio-EditX using the provided Dockerfile.

```bash
# build docker
docker build . -t step-audio-editx

# run docker
docker run --rm --gpus all \
    -v /your/code/path:/app \
    -v /your/model/path:/model \
    -p 7860:7860 \
    step-audio-editx
```


#### Launch Web Demo
Start a local server for online inference.
Assume you have one GPU with at least 12GB memory available and have already downloaded all the models.

```bash
# Step-Audio-EditX demo
python app.py --model-path where_you_download_dir --model-source local

# Memory-efficient options with quantization
# For systems with limited GPU memory, you can use quantization to reduce memory usage:

# INT8 quantization
python app.py --model-path where_you_download_dir --model-source local --quantization int8

# INT4 quantization
python app.py --model-path where_you_download_dir --model-source local --quantization int4

# Example with custom settings:
python app.py --model-path where_you_download_dir --model-source local --torch-dtype float16
```

#### Local Inference Demo
> [!TIP]
> For optimal performance, keep audio under 30 seconds per inference.

```bash
# zero-shot cloning
python3 tts_infer.py \
    --model-path where_you_download_dir \
    --output-dir ./output \
    --prompt-text "your prompt text"\
    --prompt-audio your_prompt_audio_path \
    --generated-text "your target text" \
    --edit-type "clone"

# edit
python3 tts_infer.py \
    --model-path where_you_download_dir \
    --output-dir ./output \
    --prompt-text "your promt text" \
    --prompt-audio your_prompt_audio_path \
    --generated-text "" \ # for para-linguistic editing, you need to specify the generatedd text
    --edit-type "emotion" \
    --edit-info "sad" \
    --n-edit-iter 2
```

## Technical Details
<img src="assets/architechture.png" width=900>
Step-Audio-EditX comprises three primary components: 

- A dual-codebook audio tokenizer, which converts reference or input audio into discrete tokens.
- An audio LLM that generates dual-codebook token sequences.
- An audio decoder, which converts the dual-codebook token sequences predicted by the audio LLM back into audio waveforms using a flow matching approach.

Audio-Edit enables iterative control over emotion and speaking style across all voices, leveraging large-margin data during SFT and PPO training.

## Evaluation

### Comparison between Step-Audio-EditX and Closed-Source models.

- Step-Audio-EditX demonstrates superior performance over Minimax and Doubao in both zero-shot cloning and emotion control.
- Emotion editing of Step-Audio-EditX significantly improves the emotion-controlled audio outputs of all three models after just one iteration. With further iterations, their overall performance continues to improve.


<img src="assets/emotion-eval.png" width=800 >


## Acknowledgements

Part of the code for this project comes from:
* [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
* [transformers](https://github.com/huggingface/transformers)
* [FunASR](https://github.com/modelscope/FunASR)

Thank you to all the open-source projects for their contributions to this project!

## License Agreement
+ The code in this open-source repository is licensed under the [Apache 2.0](LICENSE) License.

## Citation

```
@misc{yan2025stepaudioeditxtechnicalreport,
      title={Step-Audio-EditX Technical Report}, 
      author={Chao Yan and Boyong Wu and Peng Yang and Pengfei Tan and Guoqiang Hu and Yuxin Zhang and Xiangyu and Zhang and Fei Tian and Xuerui Yang and Xiangyu Zhang and Daxin Jiang and Gang Yu},
      year={2025},
      eprint={2511.03601},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.03601}, 
}

```
## ⚠️ Usage Disclaimer
- Do not use this model for any unauthorized activities, including but not limited to:
  - Voice cloning without permission
  - Identity impersonation
  - Fraud
  - Deepfakes or any other illegal purposes
- Ensure compliance with local laws and regulations, and adhere to ethical guidelines when using this model.
- The model developers are not responsible for any misuse or abuse of this technology.

We advocate for responsible generative AI research and urge the community to uphold safety and ethical standards in AI development and application. If you have any concerns regarding the use of this model, please feel free to contact us.

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=stepfun-ai/Step-Audio-EditX&type=Date)](https://star-history.com/#stepfun-ai/Step-Audio-EditX&Date)
