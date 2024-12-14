# controller.py
# 다른 가상환경에서 실행되며 requests 라이브러리 사용 가정
import concurrent.futures

import requests

# 특정 모델 목적지 정보와 request 정보를 담을 수 있는 함수로 변경 필요
def call_main_task(data):
    response = requests.post("http://localhost:8000/execute", json={"data": data})
    if response.status_code == 200:
        return response.json()["result"]
    else:
        raise Exception(f"Error from server: {response.text}")

if __name__ == "__main__":
    # ThreadPoolExecutor를 사용하여 동시에 요청 보내기
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # 1부터 10까지의 숫자를 병렬로 call_main_task에 전달
        futures = {executor.submit(call_main_task, i): i for i in range(1, 20)}
        # 결과를 출력
        for future in concurrent.futures.as_completed(futures):
            input_data = futures[future]  # 제출된 데이터 (1부터 10까지의 숫자)
            try:
                result = future.result()  # call_main_task의 반환값
                print(f"Called main_task({input_data}) -> {result}")
            except Exception as e:
                print(f"Task {input_data} raised an exception: {e}")