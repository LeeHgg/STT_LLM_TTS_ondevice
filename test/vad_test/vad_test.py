import webrtcvad
import pyaudio
import wave
import time
import queue
import threading
import speech_recognition as sr

# STT 변환기
recognizer = sr.Recognizer()

# VAD 초기화 (0~3, 숫자가 클수록 민감)
vad = webrtcvad.Vad(2)

# 오디오 설정
RATE = 16000  # 샘플링 레이트 (16kHz)
FRAME_DURATION = 30  # 프레임 길이 (ms, 10 / 20 / 30 선택 가능)
CHUNK = int(RATE * FRAME_DURATION / 1000)  # 한 번 읽을 샘플 수
FORMAT = pyaudio.paInt16
CHANNELS = 1

# PyAudio 스트림 초기화
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# 녹음 중인지 상태 플래그
recording = False
audio_frames = []
last_speech_time = None
queue_audio = queue.Queue()

# 무음 지속 시 녹음 종료 (초)
SILENCE_TIMEOUT = 2.0

def save_and_transcribe(audio_data):
    """녹음된 오디오 저장 및 STT 변환"""
    file_name = f"recorded_{int(time.time())}.wav"
    
    # WAV 파일로 저장
    with wave.open(file_name, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(audio_data))
    
    print(f"[LOG] 녹음 종료: {file_name}")
    
    # STT 변환
    with sr.AudioFile(file_name) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language="ko-KR")
            print(f"[STT] 변환된 텍스트: {text}")
        except sr.UnknownValueError:
            print("[STT] 인식 실패")
        except sr.RequestError:
            print("[STT] API 요청 실패")

def vad_thread():
    """VAD 감지 및 녹음 관리"""
    global recording, audio_frames, last_speech_time
    
    print("[LOG] VAD 감지 시작...")
    while True:
        frame = stream.read(CHUNK, exception_on_overflow=False)
        is_speech = vad.is_speech(frame, RATE)

        if is_speech:
            if not recording:
                recording = True
                audio_frames = []
                print("[LOG] 녹음 시작")
            last_speech_time = time.time()
            audio_frames.append(frame)
        
        else:
            if recording and last_speech_time and time.time() - last_speech_time > SILENCE_TIMEOUT:
                recording = False
                queue_audio.put(audio_frames)  # STT 처리를 위해 큐에 저장

def stt_thread():
    """녹음된 음성을 STT 변환"""
    while True:
        audio_data = queue_audio.get()
        save_and_transcribe(audio_data)

# Thread 실행
threading.Thread(target=vad_thread, daemon=True).start()
threading.Thread(target=stt_thread, daemon=True).start()

# 프로그램 실행 유지
while True:
    time.sleep(1)
