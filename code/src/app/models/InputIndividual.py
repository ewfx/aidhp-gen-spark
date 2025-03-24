from pydantic import BaseModel
from typing import List, Dict
from .Transactions import Transactions
from .SocialMediaSentiment import SocialMediaSentiment


class InputIndividual(BaseModel):
    customer_id: str
    name: str
    age: int
    gender: str
    education: str
    occupation: str
    income: int
    location: str
    timestamp: str
    device_type: str
    interests: List[str]
    preferences: List[str]
    financial_needs: List[str]
    transaction_history: List[Transactions]
    recent_searches: List[str]
    pages_visited: List[str]
    time_spent_on_page: Dict[str, int]
    number_of_visits: int
    ads_clicked: List[str]
    current_location: str
    weather: str
    active_session_duration: int
    app_open_frequency: str
    social_media_sentiment: List[SocialMediaSentiment]
    notification_interaction: Dict[str, float]
    feedback: dict
