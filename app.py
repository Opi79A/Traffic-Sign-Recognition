import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="centered"
)

# Traffic sign information
signs = {
    0: ("Speed Limit: 20 km/h", "Maximum speed allowed is 20 km/h."),
    1: ("Speed Limit: 30 km/h", "Maximum speed allowed is 30 km/h."),
    2: ("Speed Limit: 50 km/h", "Maximum speed allowed is 50 km/h."),
    3: ("Speed Limit: 60 km/h", "Maximum speed allowed is 60 km/h."),
    4: ("Speed Limit: 70 km/h", "Maximum speed allowed is 70 km/h."),
    5: ("Speed Limit: 80 km/h", "Maximum speed allowed is 80 km/h."),
    6: ("End of Speed Limit: 80 km/h", "The 80 km/h speed limit ends."),
    7: ("Speed Limit: 100 km/h", "Maximum speed allowed is 100 km/h."),
    8: ("Speed Limit: 120 km/h", "Maximum speed allowed is 120 km/h."),
    9: ("No Passing", "Overtaking other vehicles is prohibited."),
    10: ("No Passing for Heavy Vehicles", "Heavy vehicles over 3.5 tons must not overtake."),
    11: ("Priority at Next Intersection", "You have priority at the next intersection, subject to local rules."),
    12: ("Priority Road", "You are traveling on a priority road."),
    13: ("Yield", "Slow down and give way to traffic with priority."),
    14: ("Stop", "Stop completely and check for other traffic before proceeding."),
    15: ("No Vehicles", "Vehicles are prohibited from entering."),
    16: ("Heavy Vehicles Prohibited", "Vehicles over 3.5 tons are prohibited."),
    17: ("No Entry", "Do not enter this road from this direction."),
    18: ("General Caution", "Be alert and watch for possible hazards."),
    19: ("Dangerous Curve to the Left", "Slow down before the left curve."),
    20: ("Dangerous Curve to the Right", "Slow down before the right curve."),
    21: ("Double Curve", "Slow down and prepare for successive curves."),
    22: ("Bumpy Road", "Reduce speed because the road surface may be uneven."),
    23: ("Slippery Road", "Avoid sudden braking and sharp turns."),
    24: ("Road Narrows on the Right", "Be careful because the road narrows on the right."),
    25: ("Road Work", "Slow down and watch for road workers."),
    26: ("Traffic Signals Ahead", "Be prepared to obey traffic signals ahead."),
    27: ("Pedestrians", "Watch for pedestrians crossing the road."),
    28: ("Children Crossing", "Slow down and watch for children."),
    29: ("Bicycles Crossing", "Watch for cyclists crossing the road."),
    30: ("Beware of Ice or Snow", "The road may be icy or covered with snow."),
    31: ("Wild Animals Crossing", "Watch for wild animals crossing the road."),
    32: ("End of Speed and Passing Limits", "The indicated restrictions end."),
    33: ("Turn Right Ahead", "Turn right as indicated by the sign."),
    34: ("Turn Left Ahead", "Turn left as indicated by the sign."),
    35: ("Ahead Only", "Continue straight ahead."),
    36: ("Go Straight or Right", "Continue straight or turn right."),
    37: ("Go Straight or Left", "Continue straight or turn left."),
    38: ("Keep Right", "Keep to the right side as indicated."),
    39: ("Keep Left", "Keep to the left side as indicated."),
    40: ("Roundabout", "Follow the roundabout direction and give way as required."),
    41: ("End of No Passing", "The no-passing restriction ends."),
    42: ("End of No Passing for Heavy Vehicles", "The restriction for heavy vehicles ends.")
}


@st.cache_resource
def load_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(43, activation="softmax")
    ])

    model.load_weights("traffic_sign_weights.weights.h5")
    return model


model = load_model()

# Custom styling
st.markdown("""
<style>
    .block-container {
        max-width: 760px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 16px;
        margin-bottom: 28px;
    }

    .section-title {
        font-size: 20px;
        font-weight: 650;
        margin-top: 18px;
        margin-bottom: 10px;
    }

    .result-card {
        background-color: #f0f7ff;
        border: 1px solid #cfe3ff;
        border-radius: 14px;
        padding: 22px;
        margin-top: 16px;
        margin-bottom: 16px;
    }

    .sign-name {
        color: #123d73;
        font-size: 27px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .sign-description {
        color: #334155;
        font-size: 16px;
        line-height: 1.6;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 9px;
        font-weight: 600;
        min-height: 45px;
    }

    [data-testid="stFileUploader"] {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="main-title">🚦 Traffic Sign Recognition</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'An AI-powered application for traffic sign classification'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# Image input
st.markdown("### 1. Choose an Image")

input_method = st.radio(
    "Image source",
    ["Upload Image", "Use Camera"],
    horizontal=True
)

image = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload a traffic sign image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

else:
    camera_file = st.camera_input("Take a picture of a traffic sign")

    if camera_file:
        image = Image.open(camera_file).convert("RGB")

# Preview and prediction
if image is not None:
    st.markdown("### 2. Image Preview")

    st.image(
        image,
        caption="Selected Traffic Sign",
        width=320
    )

    if st.button("Recognize Traffic Sign", type="primary"):

        with st.spinner("Analyzing the image..."):

            resized = image.resize((32, 32))

            image_array = np.array(
                resized,
                dtype=np.float32
            ) / 255.0

            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(image_array, verbose=0)[0]

            class_id = int(np.argmax(predictions))
            confidence = float(predictions[class_id] * 100)

        sign_name, description = signs[class_id]

        st.divider()
        st.markdown("### 3. Recognition Results")

        st.markdown(
            f"""
            <div class="result-card">
                <div class="sign-name">{sign_name}</div>
                <div class="sign-description">{description}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Class ID", class_id)

        with col2:
            st.metric("Confidence", f"{confidence:.2f}%")

        st.progress(min(confidence / 100, 1.0))

        if confidence < 60:
            st.warning(
                "The prediction may be unreliable. "
                "Try a clearer image."
            )

        st.caption(
            "This application is an educational project. "
            "Predictions may be incorrect and should not be used "
            "as the sole basis for real driving decisions."
        )

else:
    st.info(
        "Upload an image or use your camera to get started."
    )

st.divider()

st.caption("Deep Learning Project | Traffic Sign Recognition")
