import streamlit as st
import numpy as np
from audio_recorder_streamlit import audio_recorder
from helper import extract_feature
import soundfile
import pickle as pk
import io
import os

SAMPLE_RATE = 48000

with open("model.pk", "rb") as f:
    model = pk.load(f)

st.title('Speech Emotion Classifier')

# Center-align the elements
st.markdown(
    """
    <style>
        .center {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

audio_bytes = audio_recorder(text="Click to record audio",recording_color="#e8b62c",neutral_color="#6aa36f",icon_size="3x")

if audio_bytes:
    st.empty()

    st.audio(audio_bytes, format="audio/wav")

    data, samplerate = soundfile.read(
        io.BytesIO(audio_bytes),
        format="RAW",
        channels=1,
        samplerate=SAMPLE_RATE,
        subtype="FLOAT",
    )
    data = np.nan_to_num(data)

    features = extract_feature(data, SAMPLE_RATE, True, True, True)
    f_features = np.expand_dims(features, 0)
    print(f_features.shape)

    st.empty()
    st.empty()
    st.header('The predicted emotion of the speech is:', divider='rainbow', anchor='center')
    body = ["", model.predict(f_features)[0]]
    for b in body:
        print(b)
        st.header(b, anchor='center')
