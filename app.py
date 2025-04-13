import torch
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import io
import numpy as np

# Assuming you have a model class like this (replace with your actual model)
class YourModelClass(torch.nn.Module):
    def __init__(self):
        super(YourModelClass, self).__init__()
        # Define your model layers here
        self.fc = torch.nn.Linear(224 * 224 * 3, 2)  # Example, adjust based on your model

    def forward(self, x):
        # Define your forward pass here
        return self.fc(x)

# Initialize FastAPI app
app = FastAPI()

# Load the model
model = YourModelClass()
model.load_state_dict(torch.load('model_ver7.task'))  # Load your trained model's weights
model.eval()  # Set the model to evaluation mode

# Pydantic model for handling image input
class ImageRequest(BaseModel):
    image: str  # The image will be base64-encoded

# Endpoint for image prediction
@app.post("/predict")
async def predict(data: ImageRequest):
    # Decode the base64 image
    image_data = io.BytesIO(base64.b64decode(data.image))
    img = Image.open(image_data)

    # Preprocess the image (resize to match the input size of your model)
    img = img.resize((224, 224))  # Adjust size if needed based on your model
    img = np.array(img)  # Convert to numpy array

    # Normalize the image if needed (example: normalize to range [0, 1])
    img = img / 255.0  # Normalize if needed

    # Convert the image to a tensor
    img_tensor = torch.tensor(img).float()

    # Add batch dimension (model expects batch, even if it's just one image)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    # Make a prediction
    with torch.no_grad():  # No need to compute gradients for inference
        output = model(img_tensor)

    # Assuming output is a logits tensor, convert it to predicted class
    _, predicted_class = torch.max(output, 1)

    return {"prediction": predicted_class.item()}  # Return the predicted class








