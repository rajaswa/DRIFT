import platform

from flask import Flask, request

from src.scripts import *


app = Flask(__name__)


@app.route("/gettimeseries", methods=["POST"])
def index():
    model_input = request.form["model_input"]
    out = evaluate(model_input)
    return out


app.run(debug=False, port=4990)
