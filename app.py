from flask import Flask, request, render_template, url_for
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from werkzeug.utils import secure_filename
from model_file import AttentionCNN  # Replace with your actual model file if needed

print("app.py is running...") # Confirm running

app = Flask(__name__)

# === Load PyTorch Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AttentionCNN()
model.load_state_dict(torch.load("model_ver7.task", map_location=device))  # Replace with your filename
model.to(device)
model.eval()

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
            input_tensor = preprocess_image(filepath)
            with torch.no_grad():
                output = model(input_tensor)
                predicted_class = torch.argmax(output, dim=1).item()
                prediction = "✅ No Scoliosis Detected ✅" if predicted_class == 0 else "🚨 Scoliosis Detected 🚨"
                image_url = url_for("static", filename=f"uploads/{filename}")
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("diagnosis.html", prediction=prediction, image_url=image_url)

# === Image preprocessing for CNN ===
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    if not os.path.isfile(image_path):
        raise ValueError(f"Path is not a file: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)  # Shape: [1, 3, 224, 224]
        return image.to(device)
    except Exception as e:
        raise ValueError(f"Failed to process image: {e}")

# === Run the app ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)














