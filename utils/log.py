import os

def show_executed_code():
   file_path = "./llm_log/executed_code.txt"

   # 파일이 존재하지 않으면 빈 문자열 반환
   if not os.path.exists(file_path):
      print("실행된 코드가 없습니다.")
      return

   # 파일 내용을 읽어서 반환
   with open(file_path, "r", encoding="utf-8") as file:
      code = file.read()
      print(f"현재까지 실행된 코드: {code}")

def clear_executed_code():
    file_path = "./llm_log/executed_code.txt"
    
    # 파일이 존재하면 비우기
    if os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as file:
            file.write("")
        print("실행된 코드 기록이 초기화되었습니다.")
    else:
        print("실행된 코드 기록이 없습니다.")