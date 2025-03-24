from flask import Flask, render_template, jsonify, request  # type: ignore
from .routes.recommendation import recommendation_bp

app = Flask(__name__, template_folder="../templates")

# Register the recommendation Blueprint
app.register_blueprint(recommendation_bp)


@app.route("/")
def home():
    return render_template("home.html")
