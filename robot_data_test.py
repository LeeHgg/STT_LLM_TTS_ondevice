import sys
import os
import time
import threading
import torch
import subprocess
import re
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QTextEdit, QVBoxLayout, QLabel, QLineEdit
from utils.audio import Audio_record, Custom_faster_whisper
from utils.sLLM import Ollama_sLLM
from utils.custom_tts import Custom_TTS
from neuromeka import IndyDCP3
import utils.log

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 환경 변수 설정
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class RobotAssistant(QWidget):
    def __init__(self):
        super().__init__()

        # STT, LLM, TTS 초기화
        self.audio_record = Audio_record()
        self.stt_model = Custom_faster_whisper()
        self.stt_model.set_model('base')
        
        self.llm = Ollama_sLLM(model_name='llama3-ko-8b-Q8', file_name="instruction_robot_data.txt")
        self.session_id = 'robot_chat_123245'
        self.session_count = 0
        self.llm.set_session_id(self.session_id)
        
        self.tts_module = Custom_TTS()
        self.tts_module.set_model()
        self.tts_module.get_reference_speaker(speaker_path='tts_model/sample_sunhi.mp3')  # 여자 목소리

        # GUI 요소 설정
        self.initUI()

    def initUI(self):
        self.setWindowTitle("인디 로봇 AI 어시스턴트")
        self.setGeometry(100, 100, 500, 400)

        layout = QVBoxLayout()

        self.label = QLabel("음성 입력 또는 텍스트 입력 후 전송 버튼을 누르세요")
        layout.addWidget(self.label)

        self.text_input = QLineEdit()
        layout.addWidget(self.text_input)

        self.send_button = QPushButton("전송")
        self.send_button.clicked.connect(self.process_input)
        layout.addWidget(self.send_button)

        self.stt_button = QPushButton("음성 입력")
        self.stt_button.clicked.connect(self.run_stt)
        layout.addWidget(self.stt_button)

        self.response_display = QTextEdit()
        self.response_display.setReadOnly(True)
        layout.addWidget(self.response_display)

        self.tts_button = QPushButton("음성 출력")
        self.tts_button.clicked.connect(self.run_tts)
        layout.addWidget(self.tts_button)

        self.setLayout(layout)

    def run_stt(self):
        """STT 실행"""
        self.label.setText("음성 인식 중...")
        self.audio_record.record_start()
        while self.audio_record.recording:
            time.sleep(0.1)

        audio_dic = self.audio_record.record_stop(0.1)
        _, result_denoise, time_denoise = self.stt_model.run(audio_dic['audio_denoise'])
        txt_denoise = f'{result_denoise} ({time_denoise}s)'
        print(txt_denoise)
        self.text_input.setText(result_denoise)
        self.label.setText("음성 인식 완료")

    def get_robot_info(self):
        """로봇 상태 및 현재 위치 정보 가져오기"""
        indy = IndyDCP3('192.168.10.210') 
        control_data = indy.get_control_data() 
        print(control_data)
        op_state = control_data['op_state']
        print(f"op_state: {op_state}")
        if op_state == 5: 
            robot_state = "일반 모드"
        elif op_state == 17: 
            robot_state = "원격 조작 모드"
        else:
            robot_state = f"알 수 없는 상태 ({op_state})"
        robot_position = control_data['q']

        return robot_state, robot_position

    def process_input(self):
        """LLM 실행"""
        user_input = self.text_input.text().strip()
        if not user_input:
            return
        robot_state, robot_position = self.get_robot_info()
        formatted_input = (
            f"[사용자의 요청] {user_input}\n"
            f"[현재 모드] {robot_state}\n"
            f"[로봇의 q] {robot_position}"
        )
        print(formatted_input) 

        self.label.setText("AI 응답 중...")
        response = self.llm.invoke(formatted_input, self.session_id)
        self.response_display.setText(response)

        # 코드 저장 및 실행
        self.save_and_run_code(formatted_input, response)

        self.label.setText("응답 완료")

    def save_and_run_code(self, user_input, response):
        # 로그 저장
        log_entry = f"User: {user_input}\nChatBot: {response}\n--------\n"
        with open("./llm_log/chat_log.txt", "a", encoding="utf-8") as log_file:
            log_file.write(log_entry)
        
        extracted_code = self.extract_code_blocks(response)
        # 코드 저장
        if extracted_code: 
            extracted_code = "\n" + extracted_code
            with open("./llm_log/executed_code.txt", "a", encoding="utf-8") as code_file:
                code_file.write(extracted_code)  # 누적 저장
            with open("./llm_log/executing_code.py", "w", encoding="utf-8") as code_file:
                code_file.write(extracted_code)  # 실행할 코드
            # 현재 코드 실행
            self.run_executing_code()

    # Function to extract code blocks from chatbot response and wrap each line in print()
    def extract_code_blocks(self, response):
        code_blocks = re.findall(r'```(?:python)?\n(.*?)```', response, re.DOTALL)
        if not code_blocks:
            return None
        wrapped_code = []
        for block in code_blocks:
            lines = block.strip().split("\n")  # 코드 줄 단위로 분리
            if any("utils.log.show_executed_code()" in line for line in lines):
                wrapped_lines = [f'import sys\nimport os\nsys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))\nimport utils.log\n{line}' for line in lines]
            else:
                wrapped_lines = [f'print("{line}")' for line in lines]  # 각 줄을 print()로 감싸기
            wrapped_code.append("\n".join(wrapped_lines))

        return "\n\n".join(wrapped_code)

    # Function to execute the generated Python script
    def run_executing_code(self):
        try:
            result = subprocess.run(["python", "./llm_log/executing_code.py"], capture_output=True, text=True)
            print("[코드 실행 결과]\n", result.stdout)
            if result.stderr:
                print("\n실행 중 오류 발생\n", result.stderr)
        except Exception as e:
            print(f"실행 실패: {e}")

    def run_tts(self):
        """TTS 실행"""
        response_text = self.response_display.toPlainText().strip()
        if response_text:
            self.tts_module.make_speech(response_text)

            audio_file = "./tts_output/tmp.wav"

            # 파일이 생성될 때까지 최대 5초 동안 기다림
            timeout = 20
            elapsed_time = 0.0

            while not os.path.exists(audio_file) and elapsed_time < timeout:
                time.sleep(0.2) 
                elapsed_time += 0.2

            if not os.path.exists(audio_file):
                print(f"오류: {audio_file} 파일이 생성되지 않았습니다.")
                return

            print(f"TTS 파일 생성 완료: {audio_file}")

            os.system(f"start {audio_file}")  

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = RobotAssistant()
    gui.show()
    sys.exit(app.exec_())
