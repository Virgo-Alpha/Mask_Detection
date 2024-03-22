import streamlit as st
import time
from fastai.learner import load_learner

# Load the model
model_path = 'study_nbs/mask_classifier_model.pkl'
# learn = load_learner(model_path)

learn = load_learner(model_path, cpu=True)

def predict_mask(image):
    dl = learn.dls.test_dl([image])
    test_pred, _ = learn.get_preds(dl=dl)
    predicted_class_idx = test_pred.numpy()[0].argmax()
    return "masked" if predicted_class_idx == 1 else "not masked"


if "photo" not in st.session_state:
    st.session_state["photo"] = "not done"

col1, col2, col3 = st.columns([1,2,1])

col1.markdown("# Welcome to the Mask detector app")
col1.markdown("Upload a picture and the app will tell you if the person is wearing a mask or not")

def change_photo_state():
    st.session_state["photo"] = "done"

uploaded_photo = col2.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"], on_change=change_photo_state)
camera_photo = col2.camera_input("Take a photo", on_change=change_photo_state)

if st.session_state["photo"] == "done":
    progress_bar = col2.progress(0)

    for perc_complete in range(100):
        time.sleep(0.05)
        progress_bar.progress(perc_complete + 1)

    col2.success("Photo uploaded successfully")

    col3.metric(label="Temperature", value="36.5 °C", delta="3 °C")

    with st.expander("Show photo"):
        if uploaded_photo is None:
            st.image(camera_photo, use_column_width=True)
        else:
            st.image(uploaded_photo, use_column_width=True)

prediction = predict_mask(uploaded_photo) if uploaded_photo is not None else predict_mask(camera_photo)

col2.markdown(f"## Prediction: {prediction}")
