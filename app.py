from flask import Flask, request, render_template, url_for
import os
from werkzeug.utils import secure_filename
from PIL import Image

print("app.py is running...")

app = Flask(__name__)

# === Upload folder setup ===
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# === Homepage ===
@app.route("/")
def home():
    return render_template("home.html")

# === Diagnosis page ===
@app.route("/diagnosis", methods=["GET", "POST"])
def diagnosis():
    prediction = None
    image_url = None

    if request.method == "POST" and "image" in request.files:
        image_file = request.files["image"]
        filename = secure_filename(image_file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image_file.save(filepath)

        try:
            image_url = url_for("static", filename=f"uploads/{filename}")
            # Simple placeholder for diagnosis, you can replace this with actual logic later
            prediction = "🚨 Scoliosis Detected 🚨"  # This can be replaced by your actual model inference

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("diagnosis.html", prediction=prediction, image_url=image_url)

# === Run the app ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)



