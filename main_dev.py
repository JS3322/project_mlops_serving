# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Any
import onnxruntime as ort
import numpy as np
import os
import threading
import ast
from contextlib import asynccontextmanager
import uvicorn

# -------------------------------
# 파라미터 설정
# -------------------------------
MODEL_DIR = "models"
session_cache = {}
session_lock = threading.Lock()

# -------------------------------
# 입력 데이터 모델
# -------------------------------
class InferenceRequest(BaseModel):
    model_path: str = Field(..., description="ONNX 모델 경로")
    input_data: str = Field(
        default="[[0.0, 1.0, 1.0]]",
        description="2차원 float 배열 (Python-style string)"
    )

    def to_ndarray(self) -> np.ndarray:
        try:
            parsed = ast.literal_eval(self.input_data)
            return np.array(parsed, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"입력 데이터를 ndarray로 변환 실패: {e}")

# -------------------------------
# lifespan 이벤트 처리
# -------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("📦 startup: preload ONNX models from folder")
    with session_lock:
        for filename in os.listdir(MODEL_DIR):
            model_path = os.path.join(MODEL_DIR, filename)
            if not (filename.endswith(".onnx") and os.path.isfile(model_path)):
                continue
            try:
                session = ort.InferenceSession(model_path)
                session_cache[model_path] = session
                print(f"✅ Loaded: {model_path}")
            except Exception as e:
                print(f"❌ Failed to load {model_path}: {e}")
    yield
    print("🧹 shutdown complete")

# -------------------------------
# FastAPI 인스턴스 생성
# -------------------------------
app = FastAPI(lifespan=lifespan)

# -------------------------------
# 캐시된 세션으로 추론
# -------------------------------
@app.post("/infer/cached")
def infer_cached(req: InferenceRequest):
    try:
        with session_lock:
            session = session_cache.get(req.model_path)
        if session is None:
            raise HTTPException(status_code=404, detail="모델이 메모리에 로드되어 있지 않습니다.")

        input_array = req.to_ndarray()
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        output = session.run([output_name], {input_name: input_array})[0]
        return {"cached": True, "result": output.tolist()}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

# -------------------------------
# 호출 시마다 세션 생성
# -------------------------------
@app.post("/infer/uncached")
def infer_uncached(req: InferenceRequest):
    try:
        if not os.path.exists(req.model_path):
            raise HTTPException(status_code=404, detail="모델 파일을 찾을 수 없습니다.")

        session = ort.InferenceSession(req.model_path)
        input_array = req.to_ndarray()
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        output = session.run([output_name], {input_name: input_array})[0]
        return {"cached": False, "result": output.tolist()}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

# -------------------------------
# 현재 메모리에 올라간 모델 목록
# -------------------------------
@app.get("/models", response_model=List[str])
def list_models():
    with session_lock:
        return list(session_cache.keys())

# -------------------------------
# main 함수
# -------------------------------
if __name__ == "__main__":
    uvicorn.run("main_dev:app", host="0.0.0.0", port=8000, reload=True)
