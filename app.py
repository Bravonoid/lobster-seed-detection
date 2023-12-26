from roboflow import Roboflow
import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np

# Model
rf = Roboflow(api_key="bSFRHLXDjZUN9FuQw9qt")
project = rf.workspace().project("lobster-seed-detection")
model = project.version(1).model

CONFIDENCE = 30
OVERLAP = 55


def draw(image, prediction):
    # Get the coordinates
    x_center = prediction["x"]
    y_center = prediction["y"]
    w = prediction["width"]
    h = prediction["height"]

    # Draw the rectangle
    draw = ImageDraw.Draw(image)

    draw.rectangle(
        [
            (x_center - w / 2, y_center - h / 2),
            (x_center + w / 2, y_center + h / 2),
        ],
        width=3,
        outline=(255, 0, 0),
    )

    return image


def main():
    st.title("Lobster Seed Detection")

    # Choose the mode
    mode = st.sidebar.selectbox("Choose the mode", ("Upload", "Camera"))
    uploaded_file = None

    if mode == "Upload":
        st.subheader("Upload an image")
        uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
    elif mode == "Camera":
        st.subheader("Take a photo")
        uploaded_file = st.camera_input("Take a photo")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert image to string
        image = np.array(image.convert("RGB"))

        # Predict
        try:
            predictions = model.predict(image, confidence=CONFIDENCE, overlap=OVERLAP)

            # Draw
            image = Image.open(uploaded_file)
            for prediction in predictions:
                image = draw(image, prediction)

            # Show
            st.image(image, caption="Result", use_column_width=True)

            # Display total number of lobsters
            st.header(f"Total number of lobsters: {len(predictions)}")

        # Handle payload too large error
        except Exception as e:
            if "Payload Too Large" in str(e):
                st.error("Image too large. Please upload an image less than 3MB.")


if __name__ == "__main__":
    main()
