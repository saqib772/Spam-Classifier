import streamlit as st
import pickle
from nltk.corpus import stopwords
import string
import nltk
 
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text=y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
   
    return " ".join(y)



# tfidf= pickle.load('vectorizer.pkl','rb')
# model= pickle.load('model.pkl','rb')
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f1:
    model = pickle.load(f1)

st.title(" Email / Sms Spam Claasifier")
input_sms=st.text_input("Enter The Message")

if st.button('predict'):
    #preprocess
    transform_sms=transform_text(input_sms) 

    #vectorize
    vector_input=tfidf.transform([transform_sms])
    #Predict 
    result=model.predict(vector_input)[0]

    #display
    if result==1:
        st.header("Spam")
    else:
            st.header("Not spam")

