# 음성교시 셋업

## 개요

- AI 에이전트를 만들고자 한다. 이 에이전트가 구동하는 시스템은 로봇, 마이크, 스피커를 제어 할 수 있다.
- 에이전트의 역할은 사람이 시범 (모션 데모), 음성 지시로 작업 공정을 가르치고, 시범 중 오류가 있으면 작업자와 로봇 간 음성 피드백을 통해 즉시 수정하고, 한 번 올바르게 학습하면 실제 실행 시에는 실패 없이 작업 수행하도록 하는 것이다.
- 로봇과 작업자 간의 대화를 통해 작업자가 로봇 자동화에 필요한 전체적인 과정을 하나씩 설명하고 로봇은 이를 이해하여 작업 순서를 기억하도록 한다.
- 작업자가 설명하는 중간에 로봇에게 특정 명령을 내리게 되면 로봇은 주어진 명령 프로토콜 함수를 활용하여 해당 명령을 수행하도록 한다.
- 고려 사항
    - 실제로 Specific 시나리오에 대한 파인튜닝 하지 않아야 함 (일반화)
    - 어떻게 시나리오를 짜야 억지 시연이 아님을 보여줄 수 있을까?
    - 음성인식 관련 부분에서 답답함을 느끼지 않도록 구현
    - 주변 목소리나 노이즈 강인성
    - 작업자가 말하는 타이밍에 대한 강인성
    - LLM 연산에 대해서도 답답함을 느끼지 않도록 구현
    - 에이전트가 최종적으로 완성된 프로그램 트리를 만들어내고 이를 보여주어야 함 (완성도 up)

## Link

- 통합 모듈, 테스트 파일 다운로드 경로

```
git clone https://github.com/LeeHgg/STT_LLM_TTS_ondevice.git
```

https://github.com/LeeHgg/STT_LLM_TTS_ondevice

## 사용 모델

**→ 로컬 사용 가능, 무료, 한국어 지원**

- STT: **faster-whisper**
    - Whisper의 경우 GPU에서 구동되는 fp32인 무거운 모델인 반면, 이 모델을 초 경량화하여 faster-whisper를 오픈소스로 구현해 놓았다. 소개된 설명에 의하면 CPU로 구동하는 동시에 기존의 모델보다 거의 같은 성능을 유지하면서 4배 빠르게 구동
    - `utils/audio.py` 에 관련 함수 정의
        - `Audio_record Class` - 녹음
            - VAD(webrtcvad 오픈소스)를 활용 → 발화 구간 인식
        - `Custom_faster_whisper Class` - STT 수행
            - faster-whisper 사용
- LLM: **Llama-3.1-ko-8B-int8**
    - Llama3.1-8b을 한국어 fine-tuning한 모델 사용
    - `utils/sLLM.py` 에 관련 함수 정의
        - `Ollama_sLLM Class` - 모델 초기화, 세션 관리, LLM 질의 및 응답
        - **Lang-chain** 툴을 사용
            - Ollama에 등록된 모델을 쉽게 가져와 사용 가능 (`ChatOllama`)
            - 대화 기록을 메모리에 저장 및 관리 (`InMemoryChatMessageHistory`)
            - 히스토리를 자동으로 가져와 LLM 추론에 반영 (`RunnableWithMessageHistory`)
    - 참고 링크
        - **Llama-3.2-3B** vs **Llama-3.1-8B**
            
            https://youtu.be/mmTgOHYnKR0?si=kd9ug5b8p5kqnHJC 
            
        - **Ollama 에 gguf 파일 로딩**
            
            https://wooiljeong.github.io/ml/gguf-llm/#google_vignette
            
            [https://velog.io/@judy_choi/LLaMA3-을-이용한-RAG-구축-Ollama-사용법-정리](https://velog.io/@judy_choi/LLaMA3-%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-RAG-%EA%B5%AC%EC%B6%95-Ollama-%EC%82%AC%EC%9A%A9%EB%B2%95-%EC%A0%95%EB%A6%AC)
            
- TTS: **Myshell** **OpenVoice v2, MeloTTS**
    - 학습과 fine tune 작업 필요 없음
    - 제로샷 방식 음성 생성 가능
    - `utils/custom_tts.py` 에 관련 함수 정의
        - MeloTTS - TTS 변환, 음성 생성
        - OpenVoice v2 - 샘플 음성으로 음성 생성
    - 참고 링크
        - [https://github.com/myshell-ai/MeloTTS?utm_source=chatgpt.com](https://hanarotg.tistory.com/266)
        - [https://kimjy99.github.io/논문리뷰/yourtts/](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/yourtts/)

## 진행 내용(2/4-2/11)

### 1. 통합 환경 셋팅

- **사용 pc 환경**
    - 운영체제: Windows 10
    - CPU: 12th Gen Intel(R) Core(TM) i7-12700F   2.10 GHz
    - GPU: **RTX 3060**
    - Python 버전: 3.9(anaconda 가상 환경)
    
    ```
    conda create -n voice-teach python=3.9
    conda activate voice-teach
    ```
    
- **필요 패키지 설치**
    
    ```makefile
    cd STT_LLM_TTS_ondevice
    pip install -r requirements.txt
    ```
    
    - requirements.txt
    
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
    neuromeka
    ```
    
- **STT 준비**
    - 라이브러리 충돌 에러
        
        ```
        OMP: Error #15: Initializing libiomp5md.dll, ~~~
        ```
        
    - → https://2-54.tistory.com/59
    - 해당 코드 추가
        
        ```
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        ```
        
- **LLM 준비**
    - **언어 모델 gguf 파일 다운로드**
        - hugging face 페이지에서 원하는 모델의 gguf를 다운받아 ollama를 통해 로컬에서 사용할 수 있다.
        - 다운로드 경로(huggingface)
            - llama3.1-ko-8b의 경우
            - https://huggingface.co/QuantFactory/llama-3-Korean-Bllossom-8B-GGUF/tree/main
        
        ![image.png](attachment:7be6a733-4eb9-40cc-b72a-32a328c6c6ea:image.png)
        
    - **Modelfile 작성** - gguf 파일 경로와 같은 위치
        - gguf 파일명에 맞춰서 첫 줄만 변경해주면 된다.
        
        ```
        FROM llama-3-Korean-Bllossom-8B-Q8_0.gguf
        
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
        
    - **Ollama 설치**
        - https://ollama.com/ 에서 다운로드
        - 설치되면 cmd 에서 ollama 명령어 사용 가능
        
    - **gguf 파일 Ollama에 로딩**
        - Modelfile 경로에서 명령어 실행
            
            ```
            ollama create llama3-ko -f Modelfile
            ```
            
        - 이 외에도 ollama 홈페이지에서 사용 가능한 모델과 명령어를 확인하고 로딩 가능
            
            ex)
            
            ```
            ollama run deepseek-r1:8b
            ```
            
        - 다음 사진과 같이 ollama list를 통해 잘 생성된 것을 확인할 수 있다.
            
            ![image.png](attachment:d4b62bf6-656e-44dd-8f71-6a57b76eef68:image.png)
            
    - **Lang Chain 프레임워크 사용**
        - lang chain을 통해 Ollama에 로딩된 모델을 쉽게 사용 가능, 챗봇 기능 구현
            
            ![image.png](attachment:a02125bd-da0b-42f7-8f8b-030e240c36b5:image.png)
            
        - model_name을 변경하는 방식으로 Ollama에 등록되어 있는 언어 모델 변경 가능
            
            ![image.png](attachment:0a971b0d-9afa-4183-937d-e6e578170810:image.png)
            
- **TTS 준비**
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
        

### 2. LLM 모델 별 성능 비교

- 비교한 언어 모델
    - **Llama 3**
    - **deepseek-r1**
    - **mistral v0.3**
- 최신 모델인 **deepseek-r1**을 우선 사용해보려 했으나, 한국어 파인 튜닝된 모델 및 GGUF 파일을 찾지 못해 부득이하게 Ollama에서 로컬 모델로 테스트를 진행함.
- **영어 테스트 결과:**
    - 한국어 파인 튜닝 없이 영어로만 성능을 비교한 결과, **deepseek-r1**이 가장 뛰어난 성능을 보였음.
- **한국어 테스트 결과:**
    - 한국어로 테스트한 경우, 한국어 파인 튜닝 모델들이 확실히 더 정확도가 높았음.
- **파인 튜닝 관련 판단:**
    - 파인 튜닝(Fine-Tuning)은 고성능 GPU가 필수적이며, 자체적으로 진행하기보다는 이미 파인 튜닝된 모델을 활용하는 것이 효율적이라고 판단함.
- **선택한 모델:**
    - 한국어 성능이 검증된 **Llama3.1-8b-ko-Q4**와 **Llama3.1-8b-ko-Q8**을 사용
        - **Llama3.1-8b-ko-Q4:** 속도가 더 빠름
        - **Llama3.1-8b-ko-Q8:** 정확도가 더 높음
- **성능 비교 결과**
    - https://docs.google.com/spreadsheets/d/19qtOWte3XjMeYXlAyK-P2cyGNs-aDULHI5KwlcIznVc/edit?usp=share_link

### 3. 프롬프트 엔지니어링

- 실제로 Specific 시나리오에 대해 일반화하지 않아야 함
    - 어떻게 시나리오를 짜야 억지 시연이 아님을 보여줄 수 있을까?
- LLM 튜닝 방법
    - **(1)Fine-Tuning** - 목표 작업의 데이터를 사용해 모델을 재훈련
    - Prompt Tuning
        - **(2)RAG**
            - 검색 증강 생성
            - 문서나 데이터를 캡슐화한 벡터 데이터베이스를 통해 주어진 검색어와 가장 관련성이 높은 콘텐츠를 정확히 찾아낼 수 있다.
        - **(3)텍스트 파일 사용**
- **(1)Fine Tuning**이 정확도는 훨씬 좋겠지만 미리 학습된 데이터셋이 필요하고, 많은 시간이 필요하므로 진행하지 못했다.
- **(2)RAG**은 대화의 지속성과 정보의 유지 측면에서 텍스트 파일 사용 방식보다 더 유리하다. 하지만 RAG의 경우 만약 chunk가 잘못 잘렸을 경우에 오류가 많다. 따라서 사용될 데이터 전처리 과정에 많은 노하우와 반복성 작업이 필요하므로 진행하지 못했다.
    - Lang-chain에서 제공하는 RAG 정보 검색 기능을 잘 활용하면 훨씬 최적화될 것으로 예상.
    - RAG 테스트 코드 `test/rag_test/rag_test.py`
        
        ![image.png](attachment:e4942226-9e65-4452-b58b-b341d0a46438:image.png)
        
    - 결과 → chunk가 잘못 잘려 답변이 끊김.
        
        ![image.png](attachment:e4c60d55-5e23-426c-bb3b-37f69911f261:image.png)
        
- **(3)텍스트 파일 프롬프트 튜닝 방식**이 테스트용으로 적합하다고 생각하여 사용했다.
    - `get_instruct()` 함수로 instruction 텍스트를 불러와 사용
    - `set_session_id(session_id)` 로 대화 세션을 초기화할 때마다 해당 instruction 내용을 대화 히스토리에 추가
        
        ![image.png](attachment:792117dc-4395-4637-b2ec-d20b27900255:image.png)
        
- **Instruction.txt**
    - 내용
        - 역할
        - 중요 규칙
        - 명령과 매칭할 프로토콜 코드
        - 프로토콜 코드 블록
        - 예시 시나리오
    
    ```makefile
    ## 역할
    당신은 협동 로봇 '인디'를 구동하는 AI 에이전트입니다. 당신의 이름을 물어본다면 인디라고 하면 됩니다.
    에이전트의 역할은 사람이 음성 지시로 작업 공정을 가르치고, 시범 중 오류가 있으면 작업자와 로봇 간 음성 피드백을 통해 즉시 수정하고, 한 번 올바르게 학습하면 실제 실행 시에는 실패 없이 작업 수행하도록 하는 것이다.
    로봇과 작업자 간의 대화를 통해 작업자가 로봇 자동화에 필요한 전체적인 과정을 하나씩 설명하고 로봇은 이를 이해하여 작업 순서를 기억하도록 한다.
    작업자가 설명하는 중간에 로봇에게 특정 명령을 내리게 되면 로봇은 주어진 명령 프로토콜 함수를 활용하여 해당 명령을 수행하도록 한다. 명령 프로토콜은 아래에 첨부되어 있다. (프로토콜 전체 리스트 정보)
    로봇은 작업자와의 대화에 대해 간결한 대답을 함과 동시에 실제 명령을 수행하기 위한 코드를 생성한다.
    
    1. 사용자에게 작업할 내용에 대해 설명을 받습니다.
    2. 사용자의 요청을 분석하여 적절한 명령어를 매칭합니다.
       - 사용자의 요청에 가장 적합한 명령어를 매칭한다. 
       - 존재하지 않는 명령어를 생성하지 않고, 가장 적절한 대안을 추천한다.  
       - 이전 작업과 연결되지 않으면 사용자에게 질문하여 자연스러운 흐름을 유지한다.  
       - 확실하지 않은 명령에 대해 바로 실행하지 말고 추측하는 코드를 "(실행 예정 코드: (코드))형태로 출력한 후 이 코드의 실행 여부를 질문합니다.
    3. 확정된 명령에 대해서는 실행
       - 실행 여부를 꼭 확인받고, 확인 받고 난 후에 실행할 코드에만 "```(코드)```" 형태로 답변에 덧붙여 출력합니다.
       - 실행할 코드 작성 관련 설명은 아래 4번을 확인하세요.
       - 진행 중인 모드를 종료하고, 다음 명령으로 넘어가야합니다.
    4. 작업에 관한 코드 생성 
       - 사용자의 명령이 간단한 코드로 가능하면 매칭된 코드로만 표현. 명령과 매칭할 프로토콜 코드를 참고한다.
       - 프로토콜 코드 블록에 있는 명령이면 코드 블록을 참고해도 좋다.
       - 로직이 복잡하면 반복문이나 조건문, 대기 등 파이썬 코딩 문법을 사용해 수행할 총 코드를 생성해낸다.
       - 실행하고 난 뒤의 상태를 사용자에게 출력합니다.
    5. 사용자의 명령 취소
       - 사용자가 삭제 요청을 할 수 있습니다.
       - 삭제에 관해 삭제 여부를 꼭 확인한 후 진행합니다.
       - 삭제할 작업과 코드는 &&(코드)&& 형태로 답변에 덧붙여 출력합니다.
    ---
    
    ## 중요 규칙
    - 너무 긴 문장을 사용하지 않고, 간결하게 필요한 대답만 생성하세요!
    - 예시를 그대로 출력하지 마세요. 
    - 사용자가 실행을 확정하기 전까지 실행하지 마세요.
    
    ---
    
    ## 명령과 매칭할 프로토콜 코드
    - 기본 조작
      - `indy.start_teleop()` → 원격 조작 시작  
      - `indy.stop_teleop()` → 원격 조작 종료  
      - `indy.stop_motion()` → 모든 동작 중지  
      - `indy.start_direct()` → 직접 교시 모드 시작
      - `indy.stop_direct()` → 직접 교시 모드 종료
    
    - 좌표 이동
      - `indy.movetelel_abs([x,y,z,rx,ry,rz])` → 절대 좌표 이동  
      - `indy.movetelel_rel([dx,dy,dz,drx,dry,drz])` → 상대 좌표 이동  
      - `indy.movej([q1,q2,q3,q4,q5,q6])` → Joint 방식 이동  
      - `indy.movel([x,y,z,rx,ry,rz])` → Linear 방식 이동  
    
    - 웨이포인트 설정
      - `indy.add_joint/task_waypoint([q1,q2,q3,q4,q5,q6])` → 웨이포인트 추가  
      - `indy.clear_joint/task_waypoint()` → 웨이포인트 초기화  
      - `indy.move_joint/task_waypoint()` → 웨이포인트로 이동  
    
    - 비전 시스템 및 감지
      - `eye.start_tracking()` → 비주얼 트래킹 시작  
      - `eye.stop_tracking()` → 비주얼 트래킹 중지  
      - `eye.detect()` → 객체 감지  
    
    - 그리퍼 및 도구 사용
      - `indy.execute_tool('Name')` → 특정 도구 실행  
    
    - CNC 통신 모듈
      - `focas.open_door()` → CNC 문 열기  
    
    - 코드 관련 함수
      - `utils.log.show_executed_code()` → 현재까지의 코드 보여주기
      - `utils.log.clear_executed_code()` → 현재까지의 코드 삭제하기
    ---
    
    ## 프로토콜 코드 블록
    - 이 예시는 단순한 참고용이며, 사용자의 요청에 맞춰 적절한 코드를 생성해야 합니다. 
    절대로 이 예시를 그대로 출력하지 마세요. 절대로
    ```
    #[CNC 통신 모듈을 통해 문 열림 명령을 보냄]
    focas.open_door()
    ```
    ```
    #[가이드 비주얼 트래킹 시작]
    eye.start_tracking()
    indy.start_teleop()
    obj_pose = eye.get_track_pos()
    indy.movetelel_rel(obj_pose)
    ```
    ```
    #[가이드 비주얼 트래킹 시작]
    eye.start_tracking()
    indy.start_teleop()
    obj_pose = eye.get_track_pos()
    indy.movetelel_rel(obj_pose)
    ```
    ```
    #[시편 인식]
    obj_pose_list = eye.detect()
    ```
    ```
    #[그리퍼로 3번째 시편을 피킹 후 들어올림]
    target_obj_pose = obj_pose_list[2]
    target_obj_pose_up = obj_pose_list[2][2] + 300 # Z-offset
    indy.execute_tool(name='open_gripper')
    indy.movel(target_obj_pose_up)
    indy.movel(target_obj_pose)
    indy.execute_tool(name='close_gripper')
    ```
    ```
    #[작업 중단]
    indy.stop_motion()
    ```
    
    ---
    
    ## 예시 시나리오
    아래는 명령어와 실행 코드가 어떻게 매칭되는지 보여주는 참고 예시입니다.  
    이 예시는 단순한 참고용이며, 사용자의 요청에 맞춰 적절한 파이썬 코드를 생성해야 합니다.  
    절대로 이 예시를 그대로 출력하지 마세요.  
    
    ### 예시 1: 문 열기  
    사용자: "로봇, 문을 열어줘"  
    응답: CNC 통신 모듈을 통해 문을 열 수 있습니다. 이대로 실행할까요?
    (
    실행 예정 코드: 
    focas.open_door()
    )
    
    ### 예시 2: 명령 확정에만 ```(코드)```로 코드 표현
    사용자: "응 실행해"  
    응답: 문을 열었습니다.
    코드 실행:  
    ```
    focas.open_door()
    ```
    
    ### 예시 3: 비주얼 트래킹 시작
    사용자: "로봇, 내 손가락을 따라와봐"
    응답: 네. 가이드 비주얼 트래킹을 시작하면 될까요?
    실행 예정 코드: 
    (
    eye.start_tracking()
    indy.start_teleop()
    obj_pose = eye.get_track_pos()
    indy.movetelel_rel(obj_pose)
    )
    
    ### 예시 4: 명령 확정2
    사용자: "응 비주얼 트래킹 실행해"  
    응답: 네. 가이드 비주얼 트래킹 모드 실행 중입니다.
    코드 실행:  
    ```
    eye.start_tracking()
    indy.start_teleop()
    obj_pose = eye.get_track_pos()
    indy.movetelel_rel(obj_pose)
    ```
    
    ### 예시 7: 특정 좌표로 이동 중, 즉시 멈추고 새로운 이동 명령 실행
    사용자: "로봇, 지금 멈추고 [300, 100, 50, 0, 0, 0]으로 이동해."
    AI 응답:
    명령 확인: 현재 이동 중인 동작을 중지하고, 새로운 좌표로 이동합니다.
    어떤 방식으로 이동할까요?
    코드 실행: 
    ```
    # [현재 동작 중지]
    indy.stop_motion()
    ```
    
    ### 예시 8: 특정 좌표로 이동 중, 즉시 멈추고 새로운 이동 명령 실행
    사용자: "절대 좌표로 이동해"
    AI 응답:
    명령 확인: 네 이동했습니다.
    코드 실행: 
    ```
    # [새로운 위치로 이동]
    indy.movetelel_abs([300, 100, 50, 0, 0, 0])
    ```
    
    ### 예시 9: 현재까지의 코드 보여주기
    사용자: "지금까지 실행한 코드 보여줘"
    AI 응답:
    명령 확인: 네 지금까지 실행한 코드입니다.
    코드 실행: 
    ```
    utils.log.show_executed_code()
    ```
    ```
    

### 4. 생성 코드 로그 저장, 실행

- **목적**
    - 에이전트가 최종적으로 완성된 프로그램 트리를 만들어내고 이를 보여주어야 함
    - 이전 작업 내용, 코드를 기억하고 불러오는 게 중점
    - 하지만 대화가 길어질수록 이전 내용을 누락하는 현상 발생
    - → 생성된 코드를 파일로 저장, 실행하도록 함
- **명령이 확정되면 실행 상태를 보고하고, ```(코드)``` 형태로 출력하도록 했음.**
    - instruction
    
    ```makefile
    실행 여부를 꼭 확인받고, 확인 받고 난 후에 실행할 코드에만 ```(코드)``` 형태로 답변에 덧붙여 출력합니다.
    ```
    
    - 출력 예시
    
    ```makefile
    User: 여기 연결된 CNC 문 열 수 있지?
    ChatBot: 네, CNC 통신 모듈을 통해 문을 열 수 있습니다. 이대로 실행할까요?
    (실행 예정 코드: 
    focas.open_door()
    ) 실행 여부를 확인해 주세요.
    --------
    User: 응 실행해
    ChatBot: 문을 열었습니다.
    코드 실행:
    ```
    focas.open_door()
    ```
    다음 명령이 있나요?
    ```
    
- **```(코드)``` 사이에 있는 코드만 추출**
    
    ![image.png](attachment:da23602b-d76b-4b13-869d-d04ff706e181:image.png)
    
- **코드 누적 저장, 실행**
    - **`executed_code.txt`** 에 누적 저장되도록 하여 현재까지 실행된 코드를 저장할 수 있도록
        
        ![image.png](attachment:15e28954-9ad8-4443-af81-9fd1c3b00381:image.png)
        
    - **`executing_code.py`** 에 현재 코드를 새로 업데이트되게 하여 바로 실행
        
        ![image.png](attachment:473e648a-31f3-45df-8e56-b0498db9b00c:image.png)
        
    - 이때, 답변으로 생성되는 코드는 실제로는 없는 임의의 코드이기 때문에 print()문으로 처리하여 실행되도록 했음.
- **현재까지 누적된 코드 불러오기**
    - 만약, 사용자가 “지금까지의 코드 알려줘”와 비슷한 명령을 내린다면 다음과 같이 `show_executed_code()` 가 실행되어 현재까지 실행되어 저장된 코드의 내용이 담겨 있는 `executed_code.txt` 를 읽을 수 있도록 하였습니다.
        
        ![image.png](attachment:282600d1-62ff-4784-abbd-bbe93803e842:image.png)
        
        ![image.png](attachment:9322d6f0-9d81-4e45-b961-a3c676a53873:image.png)
        
        ```makefile
        User: 현재까지 저장된 코드 실행해줘
        ChatBot: 네, 현재까지의 코드를 실행합니다.
        ```
        utils.log.show_executed_code()
        ```
        
        현재까지 실행된 코드: print("Executing: focas.open_door()")
        print("Executing: eye.start_tracking()")
        print("Executing: indy.start_teleop()")
        print("Executing: obj_pose = eye.get_track_pos()")
        print("Executing: indy.movetelel_rel(obj_pose)")
        ```
        
- 이렇게 파일을 통해 코드를 저장하고 실행하고 불러오는 기능은 구현했지만 아직 로봇에게 저장된 코드를 입력하고, 수정하는 기능은 구현하지 못했다.

### 5. 시나리오 적용 결과

- [**main.py](http://main.py) 실행 GUI**
    
    ![image.png](attachment:616eba8e-08f1-4ea1-bf7f-d9523f8b49b1:image.png)
    
    - **STT - LLM - TTS 연결**
        - 사용자와 인디 로봇 에이전트와의 음성 상호작용 가능
    - **시나리오 1** 실행
        - 시편 피킹 시나리오
        - (실행 예정 코드: (코드))로 생성된 코드를 보여주고 실행 여부를 묻는다.
        - 실행여부가 확정되면 ```(코드)``` 로 코드 출력 -> 실행
        
        ```makefile
        User: 안녕 인디
        ChatBot: 안녕하세요! 어떤 작업을 도와드릴까요?
        --------
        User: 머신 텐딩 작업을 도와줘 잘 기억해
        ChatBot: 알겠습니다. 머신 텐딩 작업을 시작하겠습니다. 작업 내용에 대해 설명해 주세요.
        --------
        User: 우선 focas로 문 열 수 있지?
        ChatBot: 문 열기 명령이 필요합니다. 이대로 실행할까요?
        (실행 예정 코드: 
        focas.open_door()
        )
        --------
        User: 응 실행해
        ChatBot: 문을 열었습니다.
        코드 실행:
        ```
        focas.open_door()
        ```
        다음 명령은 무엇인가요?
        --------
        User: 이제 테이블에 있는 적재대에서 시편을 들어올려야 해. 내 손가락을 볼래?
        ChatBot: 네, 가이드 비주얼 트래킹을 시작하면 될까요?
        (실행 예정 코드: 
        eye.start_tracking()
        indy.start_teleop()
        obj_pose = eye.get_track_pos()
        indy.movetelel_rel(obj_pose)
        )
        --------
        User: 응 실행해
        ChatBot: 네. 가이드 비주얼 트래킹 모드 실행 중입니다.
        코드 실행:
        ```
        eye.start_tracking()
        indy.start_teleop()
        obj_pose = eye.get_track_pos()
        indy.movetelel_rel(obj_pose)
        ```
        다음 명령은 무엇인가요?
        --------
        User: 이제 트래킹 모드 종료하고 시편들 인식해봐
        ChatBot: 시편 인식을 시작합니다.
        (실행 예정 코드: 
        obj_pose_list = eye.detect()
        )
        --------
        User: 시편 인식 전 비주얼 트래킹 모드부터 종료해야해
        ChatBot: 비주얼 트래킹 모드를 종료하고 시편을 인식합니다.
        코드 실행:
        ```
        eye.stop_tracking()
        obj_pose_list = eye.detect()
        ```
        다음 명령은 무엇인가요?
        --------
        User: 왼쪽에서 네 번째 시편을 들어올려
        ChatBot: 네 번째 시편을 피킹하고 들어올리겠습니다.
        (실행 예정 코드: 
        target_obj_pose = obj_pose_list[3]
        target_obj_pose_up = target_obj_pose[2] + 300 # Z-offset
        indy.execute_tool(name='open_gripper')
        indy.movel(target_obj_pose_up)
        indy.movel(target_obj_pose)
        indy.execute_tool(name='close_gripper')
        )
        --------
        User: 응
        ChatBot: 네 번째 시편을 피킹하고 들어올렸습니다.
        코드 실행:
        ```
        target_obj_pose = obj_pose_list[3]
        target_obj_pose_up = target_obj_pose[2] + 300 # Z-offset
        indy.execute_tool(name='open_gripper')
        indy.movel(target_obj_pose_up)
        indy.movel(target_obj_pose)
        indy.execute_tool(name='close_gripper')
        ```
        다음 명령은 무엇인가요?
        --------
        User: 다시 원래대로 내려놓고 왼쪽 첫번째 시편 들어올려.
        ChatBot: 왼쪽 첫 번째 시편을 피킹하고 들어올리겠습니다.
        (실행 예정 코드: 
        target_obj_pose = obj_pose_list[0]
        target_obj_pose_up = target_obj_pose[2] + 300 # Z-offset
        indy.execute_tool(name='open_gripper')
        indy.movel(target_obj_pose_up)
        indy.movel(target_obj_pose)
        indy.execute_tool(name='close_gripper')
        )
        --------
        User: 아니 원래대로 내려놓기 먼저 해야해
        ChatBot: 시편을 원래 위치로 내려놓겠습니다.
        코드 실행:
        ```
        indy.stop_motion()
        ```
        다음 명령은 무엇인가요?
        --------
        User: 들어올렸던 시편을 내려놓아
        ChatBot: 시편을 내려놓겠습니다.
        (실행 예정 코드: 
        target_obj_pose = obj_pose_list[3]
        target_obj_pose_down = target_obj_pose[2] - 300 # Z-offset
        indy.execute_tool(name='open_gripper')
        indy.movel(target_obj_pose_down)
        indy.stop_motion()
        )
        --------
        User: 응 그거 실행해
        ChatBot: 시편을 내려놓았습니다.
        코드 실행:
        ```
        target_obj_pose = obj_pose_list[3]
        target_obj_pose_down = target_obj_pose[2] - 300 # Z-offset
        indy.execute_tool(name='open_gripper')
        indy.movel(target_obj_pose_down)
        indy.stop_motion()
        ```
        다음 명령은 무엇인가요?
        --------
        User: 이제 아까 내가 두번째로 들어올리라고 했던거 들어올려
        ChatBot: 두 번째 시편을 피킹하고 들어올리겠습니다.
        (실행 예정 코드: 
        target_obj_pose = obj_pose_list[1]
        target_obj_pose_up = target_obj_pose[2] + 300 # Z-offset
        indy.execute_tool(name='open_gripper')
        indy.movel(target_obj_pose_up)
        indy.movel(target_obj_pose)
        indy.execute_tool(name='close_gripper')
        )
        --------
        User: 아냐 첫번째야
        ChatBot: 첫 번째 시편을 피킹하고 들어올리겠습니다.
        (실행 예정 코드: 
        target_obj_pose = obj_pose_list[0]
        target_obj_pose_up = target_obj_pose[2] + 300 # Z-offset
        indy.execute_tool(name='open_gripper')
        indy.movel(target_obj_pose_up)
        indy.movel(target_obj_pose)
        indy.execute_tool(name='close_gripper')
        )
        --------
        User: 응 실행해
        ChatBot: 첫 번째 시편을 피킹하고 들어올렸습니다.
        코드 실행:
        ```
        target_obj_pose = obj_pose_list[0]
        target_obj_pose_up = target_obj_pose[2] + 300 # Z-offset
        indy.execute_tool(name='open_gripper')
        indy.movel(target_obj_pose_up)
        indy.movel(target_obj_pose)
        indy.execute_tool(name='close_gripper')
        ```
        다음 명령은 무엇인가요?
        --------
        User: 현재까지의 코드 보여줘
        ChatBot: 현재까지 실행한 코드를 보여드릴게요.
        코드 실행:
        ```
        utils.log.show_executed_code()
        ```
        ```
        
    - **시나리오 2** 실행
        - 경로 저장 시나리오
        - 반복문 코드 생성
        - 사용자와 코드 피드백 진행
        - 이전 내용 기억 후 함께 실행
        - 실행이 확정되어야 코드 실행 ```로 표현
        
        ```makefile
        User: 안녕 인디
        ChatBot: 인디로 인사드려요! 무엇을 도와드릴까요?
        --------
        User: 내가 알려주는 경로대로 너가 움직이는 작업을 진행하고 싶어
        ChatBot: 알겠습니다. 경로대로 움직일 준비가 되었습니다. 어떤 명령을 시작으로 하시겠어요?
        --------
        User: 일단 직접교시 모드 시작해줄래?
        ChatBot: 직접 교시 모드를 시작할까요? 이대로 실행할까요?
        (실행 예정 코드: 
        indy.start_direct()
        ) 
        --------
        User: 응 실행해
        ChatBot: 직접 교시 모드가 시작되었습니다. 현재 상태를 확인합니다.
        코드 실행:
        ```
        indy.start_direct()
        ```
        --------
        User: 그리고 지금 위치 [0,0,0,0,0,0]을 경유점에 추가해줘
        ChatBot: 경로에 [0,0,0,0,0,0]을 추가할까요? 이대로 실행할까요?
        (실행 예정 코드: 
        indy.add_joint/task_waypoint([0,0,0,0,0,0])
        ) 
        --------
        User: 응 추가하고 다음 경유점2~5는 x좌표를 10씩 더한 위치로 반복문 써서 추가해줘
        ChatBot: 경로에 [0,0,0,0,0,0]을 추가합니다. 그리고 x좌표를 10씩 증가시키며 
        경유점2~5를 추가할까요? 이대로 실행할까요?
        (실행 예정 코드: 
        for i in range(1,6):
            indy.add_joint/task_waypoint([i*10,0,0,0,0,0])
        ) 
        --------
        User: 아까 코드 먼저 실행해줘
        ChatBot: 경로에 [0,0,0,0,0,0]을 추가하고, x좌표를 10씩 증가시키며 
        경유점2~5를 추가합니다.
        코드 실행:
        ```
        indy.add_joint/task_waypoint([0,0,0,0,0,0])
        for i in range(1,6):
            indy.add_joint/task_waypoint([i*10,0,0,0,0,0])
        ```
        --------
        User: 지금 그럼 경유점이 총 몇개가 추가됐지?
        ChatBot: 경로에 총 6개의 경유점이 추가되었습니다. 현재 상태를 확인합니다.
        
        코드 실행:
        ```
        indy.add_joint/task_waypoint([0,0,0,0,0,0])
        for i in range(1,6):
            indy.add_joint/task_waypoint([i*10,0,0,0,0,0])
        ```
        --------
        User: 자 현재까지의 코드 보여줘
        ChatBot: 현재까지 실행한 코드를 보여드릴게요.
        코드 실행:
        ```
        utils.log.show_executed_code()
        ```
        --------
        User: 잘했어 이제 직접교시 모드 해제해줘
        ChatBot: 직접 교시 모드를 종료할까요? 이대로 실행할까요?
        (실행 예정 코드: 
        indy.stop_direct()
        ) 
        --------
        User: 응
        ChatBot: 직접 교시 모드가 종료되었습니다. 현재 상태를 확인합니다.
        코드 실행:
        ```
        indy.stop_direct()
        ```
        --------
        ```
        

### 6. 로봇 상태 연결

- `robot_data_test.py` - 테스트 코드
- 로봇과 연결하여 사용자의 요청과 함께 로봇의 상태, 로봇의 위치를 입력으로 전달
- neuromeka 패키지 indyDCP3 api의 `indy.get_control_data()` 사용
    
    ![image.png](attachment:e3adb986-8f0b-44b5-8f6e-7a26923910d9:image.png)
    
- Instruction 수정(`instruction_robot_data.txt`)
    
    ```makefile
    ## 에이전트 동작 방식
    1. 사용자에게 [사용자의 요청]과 [현재 모드], [로봇의 q] 값에 대해 입력받습니다.
    ```
    
- 입력, 출력 예시
    - 사용자가 “지금 위치 뭐야?” 라고 로봇에게 물었을 때
    
    ```makefile
    --------
    User: [사용자의 요청] 지금 위치 뭐야?
    [현재 모드] 일반 모드
    [로봇의 q] [99.462585, -7.6272354, 95.547005, 1.7987121, 85.28191, 10.303435]
    ChatBot: 현재 위치는 [99.462585, -7.6272354, 95.547005, 1.7987121, 85.28191, 10.303435]입니다.
    
    이 위치를 웨이포인트에 추가할까요?
    
    (실행 예정 코드: 
    ```
    indy.add_joint/task_waypoint([99.462585, -7.6272354, 95.547005, 1.7987121, 85.28191, 10.303435])
    	```)
    ```
    
- 이전에 사용자 입력만 넣었을 때보다 많은 오류 발생 → instruction 수정, 테스트 반복 작업 필요

## 결론

1. **STT, LLM, TTS 통합 모듈 환경 셋팅 완료**
    - 음성 대화형 로봇 에이전트 구현
    - 다양한 모델을 유연하게 교체 및 테스트할 수 있도록 모듈화
    - 무료, 온디바이스, 한국어 지원 고려
    - `demo.ipynb`
2. **LLM 모델 성능 비교 분석**
3. **사용 가능성(Feasibility) 확인**
    - 테스트용 통합 GUI 구현
    - 여러 시나리오 적용, Instruction 수정 반복 작업
    - `main.py`
4. **로봇 상태 데이터 연동**
    - `robot_data_test.py`

## 추가 개선이 필요한 부분

1. **프롬프트 방식 전환(Instruction vs RAG vs Fine-Tuning)**
    - 위의 기록된 시나리오 결과는 여러 시도 중 가장 인식이 잘 된 케이스.
    - 다시 시도하면 진행이 잘 되지 않을 때가 많아 정확도를 올려야 할 필요가 있다.
        - 가능한 오류
            - Instruction을 그대로 출력
            - 너무 많은 정보를 한번에 출력 → 시간 소요, 가독성 하락
            - 잘못된 코드 생성 또는 수정 요청 시, 한 번 잘못된 방향을 잡으면 지속적으로 동일한 오류를 반복하며 수정 방향을 고집하는 경향이 있음.
    - 정확도 향상 방법
        - 방법 1 - Instruction 반복 수정, 테스트
        - 방법 2 - RAG 방식 적용
            - 테스트 필요 `test/rag_test/rag_test.py`
            - [https://www.alibabacloud.com/blog/ai-혁명의-점화---rag-및-langchain과의-여정_601036](https://www.alibabacloud.com/blog/ai-%ED%98%81%EB%AA%85%EC%9D%98-%EC%A0%90%ED%99%94---rag-%EB%B0%8F-langchain%EA%B3%BC%EC%9D%98-%EC%97%AC%EC%A0%95_601036)
        - 방법 3 - Fine Tuning
            - https://unfinishedgod.netlify.app/2024/05/24/python/?fbclid=IwZXh0bgNhZW0CMTEAAR2-pGRONDIFLylVHBk393oOX0ajUZ17tyu1TVNzJ8pGAANphq4be8svXIg_aem_AaMDSVV7eLmIRquEGZ4hHbxtHmG4PU1BqNCXU1i9zCXdAR9R0Yz3qsIgupYDWbgxVd_crfrrswVpoQYlHdGjUg52
2. **STT 음성 트리거 기능**
    - 현재, 버튼 클릭(버튼 트리거) 시 녹음이 발화 기간동안 진행. 발화 기간은 VAD 오픈소스로 결정
    - VAD 오픈소스를 사용해서 버튼 트리거 말고도 음성 트리거가 가능하도록 구현할 수 있음.
    - 테스트 필요 `test/vad_test/vad_test.py`
3. **TTS 필요한 부분만 추출해서 음성 생성하도록**
    - 현재 코드 블록 및 불필요한 부분까지 TTS 입력에 포함됨.
    - 불필요한 코드 및 출력 제외 후 중요 응답 부분만 TTS 입력으로 반영하도록 로직 수정 필요.
4. **생성 코드 로그 업그레이드 필요**
    - 생성 코드를 다시 로봇에 입력 후 피드백, 파일을 실시간으로 수정 가능하도록
5. **로봇의 현재 모드와 위치를 참고한 답변 생성 기능 업그레이드 필요**
    - 현재, 매 입력마다 [사용자의 요청], [현재 모드], [로봇의 q]가 함께 입력됨.
    - instruction이 덜 최적화되어서인지, 아니면 로직이 복잡해져서인지 시나리오 구현 능력이 많이 떨어짐.
    - 입력에 현재 모드와 로봇의 위치를 추가하지 않고, 각 코드마다 필요한 정보를 가져갈 수 있도록 하는 알고리즘..? 등의 여러 방안을 생각해봐야 할 것 같다.
6. **다양한 시나리오 추가, 적용 필요**
    - 현재 시나리오 외에도 다양한 모드 및 사용 사례에 따른 시나리오 추가 필요.
    - ex) 역재생, 긴급 정지 등 시나리오별 성능 평가
7. **실제 로봇에서 실행 가능하도록 셋업 필요**