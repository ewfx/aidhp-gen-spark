from pydantic import BaseModel
from typing import List


class OffersDataIndividual(BaseModel):
    merchant_offers: List[str]
    bank_offers: List[str]
    products_services: List[str]
    content_recommendations: List[str]
    user_segments: List[str]
