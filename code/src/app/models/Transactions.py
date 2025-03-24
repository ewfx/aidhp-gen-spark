from pydantic import BaseModel


class Transactions(BaseModel):
    product_id: str
    category: str
    amount: float
    purchase_date: str
    payment_mode: str
    transaction_type: str
