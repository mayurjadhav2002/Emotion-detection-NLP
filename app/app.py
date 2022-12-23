import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

pipe_lr = joblib.load(open("models/emotion_classifier_pipe_lr_03_june_2021.pkl", 'rb'))
emoji = {'anger':'😠', 'disgust':'😤', 'fear':"😨", 'joy':"😂", 'neutral':"😐", 'sadness':"😔", 'shame':"😓",
       'surprise':"😯"}
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results
def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results
def main():
    mednu = ("Home", "Monitor", "About")
    choice = st.sidebar.selectbox("Menu", mednu)
    if choice == 'Home':
        st.subheader("Home - Emotion in Space")
        with st.form(key='emotion_detection'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1,col2 = st.columns(2)

            p = predict_emotions(raw_text)
            proba = get_prediction_proba(raw_text)
            with col1:
                st.success("Original Text")
                st.write(raw_text)
                st.success("Prediction")
                st.write(p[0], emoji[p[0]])

                
        
            with col2:
                st.success("Prediction Probability")
                df = pd.DataFrame(proba, columns=pipe_lr.classes_)
                df2 = df.T.reset_index()
                df2.columns = ["emotions", "Probability"]
                fig = alt.Chart(df2).mark_bar().encode(x='emotions', y='Probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)


    elif choice == 'Monitor':
        st.subheader('Monitor')
    else:
        st.subheader('About')

if __name__ == '__main__':
    main()
