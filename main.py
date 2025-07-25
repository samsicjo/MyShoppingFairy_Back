from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.personal_router import router as personal_router
from api.user_router import router as user_router
from api.crawling_router import router as crawling_router
from api.gemini_router import router as gemini_router
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="퍼스널 컬러 분석 API", description="얼굴 이미지로 퍼스널 컬러를 분석합니다")

origins = [
    "http://localhost",
    "http://localhost:8000", # 백엔드 포트 (FastAPI 자체)
    "http://localhost:3000", # 프론트엔드 포트 (Next.js/React 등)
    "https://restapi--myshoppingfairy.netlify.app"
    # 여기에 프론트엔드가 실행되는 정확한 URL을 추가하세요.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # 또는 ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allow_headers=["*"], # 또는 필요한 헤더 목록
)
# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

# 라우터 등록
app.include_router(personal_router)
app.include_router(user_router)
app.include_router(crawling_router)
app.include_router(gemini_router)

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')