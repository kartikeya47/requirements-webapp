import streamlit as st
import pickle

def vectorize(text):
    
    vectorizer = pickle.load(open('models/VectorizerFinal.sav', 'rb'))
    X = vectorizer.transform([text])
    print(X)
    return X

def predict(vt):
    
    model = pickle.load(open('models/LogisticRegressionModelFinal.sav', 'rb'))
    pred = model.predict(vt)
    return pred[0]


def main():
    st.title("Requirement Ambiguity Prediction")
    
    input = st.text_area("Enter the requirement here:")

    if st.button("Click to Predict"):
        if input:
            vectorized_text = vectorize(input)
            pred = predict(vectorized_text)
            st.success(f"Prediction is: {pred} Requirement")
        else:
            st.warning("Please enter some text before predicting.")

if __name__ == "__main__":
    main()