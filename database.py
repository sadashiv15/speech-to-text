from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Change these according to your PostgreSQL setup
DATABASE_URL = "postgresql://postgres:pune%40123@localhost:5432/speech_db"

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()