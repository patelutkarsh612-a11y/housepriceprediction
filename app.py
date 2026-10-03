from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        area = float(request.form["area"])
        bedrooms = float(request.form["bedrooms"])
        bathrooms = float(request.form["bathrooms"])
        parking = float(request.form["parking"])
        age = float(request.form["age"])

        result = model.predict([[
            area,
            bedrooms,
            bathrooms,
            parking,
            age
        ]])

        prediction = round(result[0], 2)

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)