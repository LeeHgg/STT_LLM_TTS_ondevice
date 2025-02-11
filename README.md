# 온디바이스 STT, LLM, TTS 환경 셋팅
각 stt, llm, tts 모델 사용 코드, 환경을 합치면서 생기는 문제 해결

## 사용 모델

- STT: **faster-whisper**
- LLM: **Llama-3.1-8B-int4**
- TTS: **OpenVoice v2 기반(zeroshot 학습 지원)**

**→ 로컬 사용 가능, 무료**

## 사용 pc 환경

- 운영체제: Windows 10
- CPU: 12th Gen Intel(R) Core(TM) i7-12700F   2.10 GHz
- GPU: **RTX 3060**
- Python 버전: 3.9(anaconda 가상 환경)

```
conda create -n voice-teach python=3.9
conda activate voice-teach
cd voiceTeaching
pip install -r requirements.txt
```

## **requirements.txt**

```jsx
librosa==0.9.1
pydub==0.25.1
wavmark==0.0.3
numpy==1.22.0
eng_to_ipa==0.0.2
inflect==7.0.0
unidecode==1.3.7
whisper-timestamped==1.14.2
openai
python-dotenv
pypinyin==0.50.0
cn2an==0.5.22
jieba==0.42.1
gradio==3.48.0
langid==1.1.6
pygame==2.6.1
sentence-transformers
chromadb
langchain-huggingface
pyaudio==0.2.14
faster-whisper==1.0.2
SpeechRecognition==3.10.4
noisereduce==3.0.2
webrtcvad==2.0.10
langchain-community
PyQt5
```

## STT 준비

- 라이브러리 충돌 에러
    
    ```
    OMP: Error #15: Initializing libiomp5md.dll, ~~~
    ```
    
- → https://2-54.tistory.com/59
- 해당 코드를 실행 →  .ipynb 파일에서는 해결됐지만 아직 .py 파일에서는 해결되지 않음
    
    ```
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    ```
    

## LLM 준비

### 1. 언어 모델 gguf 파일 다운로드

- hugging face 페이지에서 원하는 모델의 gguf를 다운받아 ollama를 통해 로컬에서 사용할 수 있다.
- llama-3-Korean-Bllossom-8B-Q4_K_M.gguf 다운로드 버튼 클릭

https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M/tree/main

### 2. Modelfile 작성 - gguf 파일 경로와 같은 위치

```
FROM llama-3-Korean-Bllossom-8B-Q4_K_M.gguf

TEMPLATE """{{- if .System }}
<s>{{ .System }}</s>
{{- end }}
<s>Human:
{{ .Prompt }}</s>
<s>Assistant:
"""

SYSTEM """You are a helpful AI Assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요."""

PARAMETER temperature 0
PARAMETER num_predict 3000
PARAMETER num_ctx 4096
PARAMETER stop <s>
PARAMETER stop </s>
```

### 3. Ollama 설치

- 아래 링크에서 다운로드
- 설치되면 cmd 에서 ollama 명령어 사용 가능

https://ollama.com/

### 4. gguf 파일 Ollama에 로딩

- Modelfile 경로에서 명령어 실행
    
    ```
    ollama create llama3-ko -f Modelfile
    ```

- 이 외에도 ollama 홈페이지에서 사용 가능한 모델과 명령어를 확인하고 로딩 가능
    
    ex)
    
    ```
    ollama run deepseek-r1:671b
    ```
    

### 5. Lang Chain 프레임워크 사용

- lang chain을 통해 ollama에 로딩된 모델을 쉽게 사용 가능, 챗봇 기능 구현

- model_name을 변경하는 방식으로 Ollama에 등록되어 있는 언어 모델 변경 가능

## TTS 준비

- CUDA 및 PyTorch 설치:
    
    ```
    conda install -c conda-forge cudatoolkit=11.8 cudnn=8.9
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
    
- CUDA 동작 확인
    
    ```
    python
    import torch
    torch.cuda.is_available()
    ```
    
    → True가 출력되어야 함.
    
- MELO TTS 설치
    
    ```
    cd MeloTTS
    pip install .
    python -m unidic download
    ```
    

## demo.ipynb 실행

- 세 가지 기능을 연결해서 사용할 수 있는 테스트용 노트 demo.ipynb를 간단히 작성 후 실행해보았다.

### 결과

1. STT
    - 사용자 음성을 텍스트로 변환
    
2. LLM
    - 답변 생성
    
3. TTS
    - 생성된 답변을 음성으로 변환
    