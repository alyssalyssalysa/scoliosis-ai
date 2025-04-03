from flask import Flask, request, render_template, url_for
import numpy as np
import joblib
from PIL import Image
import os
from werkzeug.utils import secure_filename

print("app.py is running...")

app = Flask(__name__)

# Load the trained model
model = joblib.load("knn_model_0726.task")

# Folder to store uploaded files
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ✅ Now this is the homepage
@app.route("/")
def home():
    return render_template("home.html")


# ✅ Diagnosis page moved to /diagnosis
@app.route("/diagnosis", methods=["GET", "POST"])
def diagnosis():
    prediction = None
    image_url = None

    if request.method == "POST":
        if "image" in request.files:
            image_file = request.files["image"]
            filename = secure_filename(image_file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            image_file.save(filepath)

            try:
                features = preprocess_image(filepath)
                pred = model.predict(features)
                prediction = "✅ No Scoliosis Detected ✅" if pred[0] == 0 else "🚨 Scoliosis Detected 🚨"
                image_url = url_for("static", filename=f"uploads/{filename}")
            except Exception as e:
                prediction = f"Error: {str(e)}"

    return render_template("diagnosis.html", prediction=prediction, image_url=image_url)

# Image preprocessing
def preprocess_image(image_path):
    if not os.path.isfile(image_path):
        raise ValueError(f"Path is not a file: {image_path}")

    try:
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image = image.convert("RGB")
        image_array = np.array(image).flatten()

        if image_array.shape[0] != 150528:
            raise ValueError(f"Expected 150528 features, got {image_array.shape[0]}")

        return image_array.reshape(1, -1)
    except Exception as e:
        raise ValueError(f"Failed to process image: {e}")

# Launch the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))  # use Render's assigned port
    app.run(host="0.0.0.0", port=port)














