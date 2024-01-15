from roboflow import Roboflow
import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from streamlit_drawable_canvas import st_canvas
import pandas as pd

# Model
rf = Roboflow(api_key=st.secrets["roboflow"]["apikey"])
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

        # Convert image to string
        image = np.array(image.convert("RGB"))

        with st.spinner("Predicting..."):
            # Predict
            try:
                predictions = model.predict(
                    image, confidence=CONFIDENCE, overlap=OVERLAP
                )

                # Draw
                image = Image.open(uploaded_file)
                for prediction in predictions:
                    image = draw(image, prediction)

            # Handle payload too large error
            except Exception as e:
                if "Payload Too Large" in str(e):
                    st.error("Image too large. Please upload an image less than 3MB.")

        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="",
            stroke_width=2,
            stroke_color="#ffffff",
            background_image=image,
            update_streamlit=False,
            drawing_mode="rect",
            point_display_radius=1,
            key="canvas",
        )

        total_manual = 0
        total_pred = len(predictions)

        if canvas_result.json_data is not None:
            objects = pd.json_normalize(
                canvas_result.json_data["objects"]
            )  # need to convert obj to str because PyArrow
            # for col in objects.select_dtypes(include=['object']).columns:
            #     objects[col] = objects[col].astype("str")
            # st.dataframe(objects)

            total_manual = len(objects)

        st.write(f"Total number of lobsters: {total_pred}")
        st.write(f"Total number of manual count: {total_manual}")

        # Display total number of lobsters
        st.header(f"Total number of lobsters: {total_pred + total_manual}")


if __name__ == "__main__":
    main()
