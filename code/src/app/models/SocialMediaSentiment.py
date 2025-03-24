from pydantic import BaseModel


class SocialMediaSentiment(BaseModel):
    platform: str
    post_id: str
    content: str
    timestamp: str
    sentiment_score: float
    intent: str
