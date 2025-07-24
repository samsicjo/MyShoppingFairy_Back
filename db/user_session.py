from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from core.config import settings

# .env에서 읽어온 DB URL로 DB 엔진 생성
SQLALCHEMY_DATABASE_URL = settings.database_url

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_recycle=3600,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

# DB 세션을 만드는 팩토리
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)