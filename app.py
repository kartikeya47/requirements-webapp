import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def vectorize(text):
    
    vectorizer = pickle.load(open('models/VectorizerFinal.sav', 'rb'))
    X = vectorizer.transform([text])
    return X

def predict(vt):
    
    model = pickle.load(open('models/LogisticRegressionModelFinal.sav', 'rb'))
    pred = model.predict(vt)
    return pred[0]

def extract_noun_and_pronoun_from_dict(sentence_dict, cluster_name):
    more_sentences_bool = 0
    sentence_1, sentence_2, *more_sentences = list(sentence_dict[cluster_name])
    words_noun = word_tokenize(str(sentence_1))
    tagged_words_noun = pos_tag(words_noun)
    noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
    noun = [word for word, tag in tagged_words_noun if tag in noun_tags]
    if len(noun) > 1:
      noun = noun[0] + " " + noun[1]
    else:
      noun = noun[0]
    words_pronoun = word_tokenize(str(sentence_2))
    tagged_words_pronoun = pos_tag(words_pronoun)
    pronoun_tags = ['PRP', 'PRP$', 'WP', 'WP$', 'DT']
    pronoun = [word for word, tag in tagged_words_pronoun if tag in pronoun_tags][0]
    if len(more_sentences) >= 1:
      more_sentences_bool = 1
    return noun, pronoun, more_sentences_bool

def replace_pronoun(sentence_dict, noun, pronoun, cluster_name):
    sentence_1, sentence_2, *more_sentences = list(sentence_dict[cluster_name])
    sentences = str(sentence_2).split(" ")
    replaced_sentence = []
    for sentence in sentences:
      if sentence == pronoun:
        replaced_sentence.append(noun)
      else:
        replaced_sentence.append(sentence)
    test_string = " ".join(replaced_sentence)

    return test_string

def coreference_resolution(original_sentence):

    try:

        nlp = spacy.load("en_coreference_web_trf")
        doc = nlp(original_sentence)
        sentence_dict = doc.spans.data
        dict_length = len(sentence_dict)
        res = original_sentence

        for i in range(1, dict_length + 1):
            cluster_name = f'coref_clusters_{i}'
            second_sentence = sentence_dict[cluster_name][1]
            noun, pronoun, more_sentences_bool = extract_noun_and_pronoun_from_dict(sentence_dict, cluster_name)
            new_sentence = replace_pronoun(sentence_dict, noun, pronoun, cluster_name)
            res = res.replace(str(second_sentence), str(new_sentence))
            if(more_sentences_bool == 1):
                sentences = original_sentence.split(" ")
                replaced_sentence = []
                for sentence in sentences:
                    if sentence == pronoun:
                        replaced_sentence.append(noun)
                    else:
                        replaced_sentence.append(sentence)
                res = " ".join(replaced_sentence)

        return res

    except:
    
        return original_sentence

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
                margin-bottom: 18px;
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
            if pred == "INNOCUOUS":
                st.write("###### Resolved Requirement: " + coreference_resolution(input))
        else:
            st.warning("Enter a Requirement to Proceed!")

if __name__ == "__main__":
    main()