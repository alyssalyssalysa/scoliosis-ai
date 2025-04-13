from PIL import Image, ImageFilter

# Load the image from the static folder
image = Image.open("static/sample.jpg")

# Apply Gaussian blur
blurred_image = image.filter(ImageFilter.GaussianBlur(radius=45))

# Save the blurred image back into static/
blurred_image.save("static/sample_blurred.jpg")

print("Done! Blurred image saved as static/sample_blurred.jpg")

