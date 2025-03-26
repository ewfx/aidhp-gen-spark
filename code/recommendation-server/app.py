# app.py

from flask import Flask, request, jsonify
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import json
from flask_cors import CORS

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

app = Flask(__name__)
CORS(app)
# -----------------------------
# Model Architecture Definitions
# -----------------------------
class DeepModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeepModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class WideDeepModel(nn.Module):
    def __init__(self, wide_dim, deep_input_dim, hidden_dim):
        super(WideDeepModel, self).__init__()
        self.wide = nn.Linear(wide_dim, 1)
        self.deep = DeepModel(deep_input_dim, hidden_dim, 1)
        
    def forward(self, wide_input, deep_input):
        wide_out = self.wide(wide_input)
        deep_out = self.deep(deep_input)
        combined = wide_out + deep_out  # simple summation
        return torch.sigmoid(combined)  # output between 0 and 1

def load_trained_model(wide_dim=10, deep_input_dim=800, hidden_dim=100):
    model = WideDeepModel(wide_dim, deep_input_dim, hidden_dim)
    # Adjust the path to your model file if needed
    model.load_state_dict(torch.load("./models/wide.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# -----------------------------
# Initialize BERT Components
# -----------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # use [CLS] token
    return cls_embedding.numpy().flatten()

def get_user_bert_embedding(texts):
    embeddings = [get_text_embedding(text) for text in texts if text.strip() != ""]
    if not embeddings:
        return np.zeros(768, dtype=np.float32)
    return np.mean(embeddings, axis=0)

# -----------------------------
# Helper Functions for Feature Extraction (Inference)
# -----------------------------
def get_wide_features(user):
    # Extract wide features from user structured data: age, income, and number_of_visits.
    age = user.get("age", 0)
    income = user.get("income", 0)
    visits = user.get("number_of_visits", 1)  # default to 1 if missing
    features = [age / 100.0, income / 200000.0, visits / 10.0]
    features += [0.0] * (10 - len(features))
    return np.array(features, dtype=np.float32)

def get_deep_features(user):
    # Combine declared interests and preferences into one text for BERT.
    interests = user.get("interests", [])
    preferences = user.get("preferences", [])
    combined_text = " ".join(interests) + " " + " ".join(preferences)
    bert_emb = get_user_bert_embedding([combined_text])  # 768-dim vector
    return bert_emb

def get_dummy_product_embedding(product):
    # Generate a dummy 32-dim product embedding from product_id for consistency.
    seed_str = ''.join(filter(str.isdigit, product["product_id"]))
    seed = int(seed_str) if seed_str else 42
    np.random.seed(seed)
    return np.random.rand(32).astype(np.float32)

def predict_score(model, wide_feature, deep_feature):
    wide_tensor = torch.tensor(wide_feature, dtype=torch.float).unsqueeze(0)
    deep_tensor = torch.tensor(deep_feature, dtype=torch.float).unsqueeze(0)
    with torch.no_grad():
        score = model(wide_tensor, deep_tensor)
    return score.item()

# -----------------------------
# Regenerated Bank Products Catalog
# -----------------------------
bank_products = [
    {
        "product_id": "DC001",
        "type": "Debit Card",
        "productName": "FlexiFlow Debit",
        "category": "Everyday",
        "features": ["No Maintenance Fee", "High Cashback", "Instant Alerts"],
        "offers": [
            {"offerName": "Zero Annual Fee", "offerType": "Fee Waiver", "description": "No annual fee for the first year."},
            {"offerName": "Cashback Bonus", "offerType": "Cashback", "description": "Earn extra cashback on purchases."}
        ]
    },
    {
        "product_id": "DC002",
        "type": "Debit Card",
        "productName": "NetSavvy Debit",
        "category": "Online",
        "features": ["Global ATM Access", "Low Transaction Fees", "Real-time Notifications"],
        "offers": [
            {"offerName": "No Transaction Fee", "offerType": "Fee Waiver", "description": "No fees on foreign transactions."}
        ]
    },
    {
        "product_id": "DC003",
        "type": "Debit Card",
        "productName": "Everyday Debit",
        "category": "Everyday",
        "features": ["Everyday Discounts", "Reward Points"],
        "offers": [
            {"offerName": "Reward Points Multiplier", "offerType": "Rewards", "description": "Double points on everyday purchases."}
        ]
    },
    {
        "product_id": "DC004",
        "type": "Debit Card",
        "productName": "CashBoost Debit",
        "category": "Rewards",
        "features": ["Cashback Rewards", "Budget Management Tools"],
        "offers": [
            {"offerName": "Instant Cashback", "offerType": "Cashback", "description": "Immediate cashback on every transaction."}
        ]
    },
    {
        "product_id": "DC005",
        "type": "Debit Card",
        "productName": "SmartSpend Debit",
        "category": "Smart",
        "features": ["Real-time Spending Tracker", "Zero Maintenance Fee"],
        "offers": [
            {"offerName": "Smart Saver", "offerType": "Savings", "description": "Earn extra savings on monthly spending."}
        ]
    },
    {
        "product_id": "CC001",
        "type": "Credit Card",
        "productName": "JetSet Credit",
        "category": "Travel",
        "features": ["Reward Miles", "Travel Perks", "Airport Lounge Access"],
        "offers": [
            {"offerName": "Travel Perks", "offerType": "Travel", "description": "Enjoy exclusive travel benefits."}
        ]
    },
    {
        "product_id": "CC002",
        "type": "Credit Card",
        "productName": "Platinum Luxe",
        "category": "Luxury",
        "features": ["Luxury Rewards", "Concierge Services", "Premium Support"],
        "offers": [
            {"offerName": "Zero Annual Fee", "offerType": "Fee Waiver", "description": "First-year fee waiver."},
            {"offerName": "Exclusive Rewards", "offerType": "Rewards", "description": "Bonus rewards on luxury purchases."}
        ]
    },
    {
        "product_id": "CC003",
        "type": "Credit Card",
        "productName": "RewardVibe Credit",
        "category": "Everyday",
        "features": ["Everyday Rewards", "Cashback Offers", "Flexible Payment Options"],
        "offers": [
            {"offerName": "Instant Cashback", "offerType": "Cashback", "description": "Immediate cashback on all purchases."}
        ]
    },
    {
        "product_id": "CC004",
        "type": "Credit Card",
        "productName": "Cashback Plus Credit",
        "category": "Cashback",
        "features": ["High Cashback", "Bonus Points", "Discount Offers"],
        "offers": [
            {"offerName": "Double Cashback", "offerType": "Cashback", "description": "Double cashback on select categories."}
        ]
    },
    {
        "product_id": "CC005",
        "type": "Credit Card",
        "productName": "Travel Elite Credit",
        "category": "Travel",
        "features": ["Bonus Miles", "Travel Insurance", "Global Acceptance"],
        "offers": [
            {"offerName": "Premium Travel Rewards", "offerType": "Travel", "description": "Extra miles and travel insurance."}
        ]
    },
    {
        "product_id": "LN001",
        "type": "Loan",
        "productName": "Personal Loan",
        "category": "Personal",
        "features": ["Low Interest Rates", "Flexible EMI Options", "Quick Approval"],
        "offers": [
            {"offerName": "Low Processing Fee", "offerType": "Fee Reduction", "description": "Reduced processing fee on your loan."}
        ]
    },
    {
        "product_id": "LN002",
        "type": "Loan",
        "productName": "Home Loan",
        "category": "Housing",
        "features": ["Competitive Interest Rates", "Long-term Financing", "Expert Consultation"],
        "offers": [
            {"offerName": "Reduced Processing Fee", "offerType": "Fee Reduction", "description": "Benefit from reduced processing fees."}
        ]
    },
    {
        "product_id": "LN003",
        "type": "Loan",
        "productName": "Auto Loan",
        "category": "Automotive",
        "features": ["Low Interest Rates", "Fast Disbursement", "Flexible EMIs"],
        "offers": [
            {"offerName": "Quick Approval", "offerType": "Service", "description": "Fast approval for your auto loan."}
        ]
    },
    {
        "product_id": "LN004",
        "type": "Loan",
        "productName": "Education Loan",
        "category": "Education",
        "features": ["Affordable Rates", "Flexible Repayment", "No Collateral Required"],
        "offers": [
            {"offerName": "Zero Processing Fee", "offerType": "Fee Waiver", "description": "No processing fee on education loans."}
        ]
    },
    {
        "product_id": "LN005",
        "type": "Loan",
        "productName": "Business Loan",
        "category": "Business",
        "features": ["Competitive Interest Rates", "Flexible Repayment Options", "Dedicated Business Support"],
        "offers": [
            {"offerName": "Business Growth Offer", "offerType": "Incentive", "description": "Special rates for business expansion."}
        ]
    }
]

merchant_products = [
    # 5 Coffee / Food & Beverage offers
    {
        "product_id": "MB001",
        "type": "Beverage",
        "productName": "Starbucks Coffee",
        "category": "Coffee",
        "features": ["Fresh Brew", "Loyalty Points", "Mobile Order"],
        "offers": [
            {"offerName": "Buy 1 Get 1 Free", "offerType": "BOGO", "description": "Buy one coffee and get one free."},
            {"offerName": "Free Pastry", "offerType": "Freebie", "description": "Free pastry with purchase over $5."}
        ]
    },
    {
        "product_id": "MB002",
        "type": "Beverage",
        "productName": "Dunkin' Donuts",
        "category": "Coffee",
        "features": ["Discounts", "Loyalty Rewards"],
        "offers": [
            {"offerName": "20% Off", "offerType": "Discount", "description": "20% off your next purchase."}
        ]
    },
    {
        "product_id": "MB003",
        "type": "Beverage",
        "productName": "Local Brew Café",
        "category": "Coffee",
        "features": ["Organic", "Artisan Roasts"],
        "offers": [
            {"offerName": "Free Upgrade", "offerType": "Upgrade", "description": "Free size upgrade on your coffee."}
        ]
    },
    {
        "product_id": "MB004",
        "type": "Food",
        "productName": "Gourmet Bites",
        "category": "Food",
        "features": ["Fresh Ingredients", "Exclusive Recipes"],
        "offers": [
            {"offerName": "Lunch Combo Deal", "offerType": "Combo", "description": "Special combo price for lunch."}
        ]
    },
    {
        "product_id": "MB005",
        "type": "Food",
        "productName": "Healthy Eats",
        "category": "Food",
        "features": ["Organic", "Low-Calorie"],
        "offers": [
            {"offerName": "Discount Meal", "offerType": "Discount", "description": "10% off on healthy meal options."}
        ]
    },
    # 5 Electronics / Watches offers
    {
        "product_id": "MB006",
        "type": "Electronics",
        "productName": "TechTime Smartwatch",
        "category": "Watches",
        "features": ["Fitness Tracking", "Notifications", "Long Battery"],
        "offers": [
            {"offerName": "10% Off", "offerType": "Discount", "description": "Get 10% off your smartwatch purchase."}
        ]
    },
    {
        "product_id": "MB007",
        "type": "Electronics",
        "productName": "Luxury Timepiece",
        "category": "Watches",
        "features": ["Premium Design", "Water Resistant"],
        "offers": [
            {"offerName": "Free Strap", "offerType": "Freebie", "description": "Free premium strap with every purchase."}
        ]
    },
    {
        "product_id": "MB008",
        "type": "Electronics",
        "productName": "SmartStyle Watch",
        "category": "Watches",
        "features": ["Multiple Watch Faces", "Activity Tracking"],
        "offers": [
            {"offerName": "Bundle Discount", "offerType": "Bundle", "description": "Bundle with phone for extra discount."}
        ]
    },
    {
        "product_id": "MB009",
        "type": "Electronics",
        "productName": "Modern Chrono",
        "category": "Watches",
        "features": ["Sleek Design", "Touchscreen"],
        "offers": [
            {"offerName": "Cashback", "offerType": "Cashback", "description": "Earn cashback on purchase."}
        ]
    },
    {
        "product_id": "MB010",
        "type": "Electronics",
        "productName": "Classic Elegance",
        "category": "Watches",
        "features": ["Timeless Style", "Quality Craftsmanship"],
        "offers": [
            {"offerName": "Free Service", "offerType": "Service", "description": "Free servicing for one year."}
        ]
    },
    # 5 Travel / Flights offers
    {
        "product_id": "MB011",
        "type": "Travel",
        "productName": "SkyHigh Flights",
        "category": "Flights",
        "features": ["Discounted Tickets", "Flexible Booking"],
        "offers": [
            {"offerName": "Flight Discount", "offerType": "Discount", "description": "Up to 20% off on flight bookings."}
        ]
    },
    {
        "product_id": "MB012",
        "type": "Travel",
        "productName": "Global Explorer",
        "category": "Flights",
        "features": ["Frequent Flyer Miles", "Reward Miles"],
        "offers": [
            {"offerName": "Bonus Miles", "offerType": "Rewards", "description": "Earn extra miles on every booking."}
        ]
    },
    {
        "product_id": "MB013",
        "type": "Travel",
        "productName": "Budget Air",
        "category": "Flights",
        "features": ["Low Cost", "No-Frills"],
        "offers": [
            {"offerName": "Special Fare", "offerType": "Discount", "description": "Exclusive low fare offer."}
        ]
    },
    {
        "product_id": "MB014",
        "type": "Travel",
        "productName": "Elite Airways",
        "category": "Flights",
        "features": ["Business Class", "Lounge Access"],
        "offers": [
            {"offerName": "Lounge Pass", "offerType": "Travel", "description": "Complimentary lounge access on international flights."}
        ]
    },
    {
        "product_id": "MB015",
        "type": "Travel",
        "productName": "FlySmart Deals",
        "category": "Flights",
        "features": ["Last Minute Deals", "Flexible Cancellation"],
        "offers": [
            {"offerName": "Last Minute Discount", "offerType": "Discount", "description": "Extra discount on last-minute bookings."}
        ]
    },
    # 5 Investments offers
    {
        "product_id": "MB016",
        "type": "Investments",
        "productName": "Smart Invest Plan",
        "category": "Investments",
        "features": ["Low Brokerage Fees", "Diversified Portfolio"],
        "offers": [
            {"offerName": "Free Advisory", "offerType": "Service", "description": "Free investment advisory for new customers."}
        ]
    },
    {
        "product_id": "MB017",
        "type": "Investments",
        "productName": "Growth Portfolio",
        "category": "Investments",
        "features": ["Long-term Growth", "Risk Management"],
        "offers": [
            {"offerName": "Bonus Points", "offerType": "Rewards", "description": "Earn bonus reward points on investments."}
        ]
    },
    {
        "product_id": "MB018",
        "type": "Investments",
        "productName": "Wealth Builder",
        "category": "Investments",
        "features": ["Expert Management", "Personalized Advice"],
        "offers": [
            {"offerName": "Waived Fees", "offerType": "Fee Waiver", "description": "No management fee for the first year."}
        ]
    },
    {
        "product_id": "MB019",
        "type": "Investments",
        "productName": "Smart Retirement Plan",
        "category": "Investments",
        "features": ["Tax Benefits", "Long-term Savings"],
        "offers": [
            {"offerName": "Bonus Advisory", "offerType": "Service", "description": "Free retirement planning consultation."}
        ]
    },
    {
        "product_id": "MB020",
        "type": "Investments",
        "productName": "Equity Growth Fund",
        "category": "Investments",
        "features": ["High Returns", "Diversification"],
        "offers": [
            {"offerName": "Loyalty Bonus", "offerType": "Rewards", "description": "Earn extra rewards for long-term investments."}
        ]
    }
]
model = load_trained_model(wide_dim=10, deep_input_dim=800, hidden_dim=100)
# -----------------------------
# Flask Server Endpoints
# -----------------------------
@app.route('/bank-recommend', methods=['POST'])
def bank_recommend():
    # Expect a JSON payload with user data
    user = request.get_json(force=True)
    
    # Extract wide features from the user dictionary
    wide_feat = get_wide_features(user)  # shape: (10,)
    
    # For deep features, use declared interests and preferences
    interests = user.get("interests", [])
    preferences = user.get("preferences", [])
    combined_text = " ".join(interests) + " " + " ".join(preferences)
    bert_emb = get_user_bert_embedding([combined_text])  # 768-dim
    
    recommendations = []
    # Evaluate each bank product
    for product in bank_products:
        product_emb = get_dummy_product_embedding(product)  # 32-dim
        deep_feat = np.concatenate([bert_emb, product_emb])  # 800-dim
        score = predict_score(model, wide_feat, deep_feat)
        recommendations.append({
            "productName": product["productName"],
            "score": score,
            "type": product["type"],
            "offers": product["offers"]
        })
    
    # Sort by descending score
    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return jsonify(recommendations)

@app.route('/merchant-recommend', methods=['POST'])
def merchant_recommend():
    # Expect a JSON payload with user data
    user = request.get_json(force=True)
    
    # Extract wide features from the user dictionary
    wide_feat = get_wide_features(user)  # shape: (10,)
    
    # For deep features, use declared interests and preferences
    interests = user.get("interests", [])
    preferences = user.get("preferences", [])
    combined_text = " ".join(interests) + " " + " ".join(preferences)
    bert_emb = get_user_bert_embedding([combined_text])  # 768-dim
    
    recommendations = []
    # Evaluate each merchant product
    for product in merchant_products:
        product_emb = get_dummy_product_embedding(product)  # 32-dim
        deep_feat = np.concatenate([bert_emb, product_emb])  # 800-dim
        score = predict_score(model, wide_feat, deep_feat)
        recommendations.append({
            "productName": product["productName"],
            "score": score,
            "type": product["type"],
            "offers": product["offers"]
        })
    
    # Sort by descending score
    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
