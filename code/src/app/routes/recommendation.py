from flask import Blueprint, jsonify, request  # type: ignore
from pydantic import ValidationError
from ..services.RecommendOffers import RecommendOffers
from ..models.RequestData import RequestData

# Create a Blueprint for the recommendation route
recommendation_bp: dict = Blueprint("recommendation", __name__)


@recommendation_bp.route("/api/recommendation/merchant_offer", methods=["POST"])
def merchant_offer():
    # Get JSON data from the request
    try:
        req_data = RequestData(**request.get_json())
    except ValidationError as e:
        return (jsonify({"error": "Invalid request data", "details": e.errors()}), 400)

    # Extract the data
    input_individual = req_data.input_individual
    offers_data_individual = req_data.offers_data_individual

    # Process the data
    r = RecommendOffers(input_individual, offers_data_individual)
    response = r.generate_merchant_offer()

    # Return the response
    response = {"message": "Predicted Successfully", "data": response}

    return jsonify(response), 200


@recommendation_bp.route("/api/recommendation/bank_offer", methods=["POST"])
def bank_offer():
    # Get JSON data from the request
    try:
        req_data = RequestData(**request.get_json())
    except ValidationError as e:
        return (jsonify({"error": "Invalid request data", "details": e.errors()}), 400)

    # Extract the data
    input_individual = req_data.input_individual
    offers_data_individual = req_data.offers_data_individual

    # Process the data
    r = RecommendOffers(input_individual, offers_data_individual)
    response = r.generate_bank_offer()

    # Return the response
    response = {"message": "Predicted Successfully", "data": response}

    return jsonify(response), 200
