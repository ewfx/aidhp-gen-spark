from pydantic import BaseModel
from .OffersDataIndividual import OffersDataIndividual
from .InputIndividual import InputIndividual


class RequestData(BaseModel):
    input_individual: InputIndividual
    offers_data_individual: OffersDataIndividual
