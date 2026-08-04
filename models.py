from sqlalchemy import Column, Integer, Text, DateTime
from sqlalchemy.sql import func

from database import Base

class Transcription(Base):
    __tablename__ = "transcriptions"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())