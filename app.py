from flask import Flask
from flask import render_template, request
from flask_bootstrap import Bootstrap
import joblib
import numpy as np

app = Flask(__name__)
bootstrap = Bootstrap(app)

model_filename = "model/model.joblib"
scaler_filename = "model/scaler.joblib"
target_names = ["setosa", "versicolor", "virginica"]

model = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)

@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        sepal_length = request.form["sepalLength"]
        sepal_width = request.form["sepalWidth"]
        petal_length = request.form["petalLength"]
        petal_width = request.form["petalWidth"]

        if sepal_width == "" or sepal_length == "" or petal_width == "" or petal_length == "":
            return render_template("home.html", input_validated=False, result=False)

        sample = np.array([float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)])
        sample = sample.reshape(1, -1)

        sample_scaled = scaler.transform(sample)

        index = model.predict(sample_scaled)[0]
        result = target_names[index]

        return render_template("home.html", input_validated=True, result=result)
        

    return render_template("home.html", input_validated=True, result=False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)