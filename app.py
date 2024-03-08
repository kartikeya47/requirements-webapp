import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def vectorize(text):
    
    vectorizer = pickle.load(open('models/VectorizerFinal.sav', 'rb'))
    X = vectorizer.transform([text])
    print(X)
    return X

def predict(vt):
    
    model = pickle.load(open('models/LogisticRegressionModelFinal.sav', 'rb'))
    pred = model.predict(vt)
    return pred[0]

def green_tick():
    return """
        <div class="tick-container">
            <div class="tick">&#10003;</div>
        </div>
        <style>
            .tick-container {
                width: 70px;
                height: 70px;
                overflow: hidden;
            }

            .tick {
                font-size: 50px;
                color: #28a745;
                line-height: 60px;
                text-align: center;
                transform-origin: 50% 50%;
                animation: tick-animation 0.5s ease-in-out;
                border: 4px solid #28a745;
                border-radius: 50%;
            }

            @keyframes tick-animation {
                0% {
                    opacity: 0;
                    transform: scale(0.5) rotate(-45deg);
                }
                100% {
                    opacity: 1;
                    transform: scale(1) rotate(0deg);
                }
            }
        </style>
    """


def main():
    st.title("Requirement Ambiguity Checker")
    
    input = st.text_area("Enter the requirement here:")

    if st.button("Click to Check"):
        if input:
            vectorized_text = vectorize(input)
            pred = predict(vectorized_text)
            col1, col2 = st.columns(2)
            with col1:
                st.write("## NOCUOUS")
                if pred == "NOCUOUS":
                    st.markdown(green_tick(), unsafe_allow_html=True) 
            with col2:
                st.write("## INNOCUOUS")
                if pred == "INNOCUOUS":
                    st.markdown(green_tick(), unsafe_allow_html=True)
        else:
            st.warning("Enter a Requirement to Proceed!")

if __name__ == "__main__":
    main()