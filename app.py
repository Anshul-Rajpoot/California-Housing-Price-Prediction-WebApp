from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load preprocessing pipeline and trained model
pipeline = joblib.load("pipeline.pkl")
model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_price = None
    error_message = None

    if request.method == "POST":
        try:
            # Read form data
            data = {
                "longitude": float(request.form["longitude"]),
                "latitude": float(request.form["latitude"]),
                "housing_median_age": float(request.form["housing_median_age"]),
                "total_rooms": float(request.form["total_rooms"]),
                "total_bedrooms": float(request.form["total_bedrooms"]),
                "population": float(request.form["population"]),
                "households": float(request.form["households"]),
                "median_income": float(request.form["median_income"]),
                "ocean_proximity": request.form["ocean_proximity"],  # categorical
            }

            # Convert to DataFrame
            df = pd.DataFrame([data])

            # Apply same preprocessing as training
            X_processed = pipeline.transform(df)

            # Predict
            y_pred = model.predict(X_processed)[0]

            predicted_price = round(float(y_pred), 2)

        except Exception as e:
            error_message = f"Error: {e}"

    return render_template(
        "index.html",
        predicted_price=predicted_price,
        error_message=error_message,
    )

if __name__ == "__main__":
    app.run(debug=True)
