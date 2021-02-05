# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 21:34:37 2021

@author: Ryan
"""


import streamlit as st
import pickle
#import sklearn
#import pandas
#from sklearn.feature_extraction.text import CountVectorizer

def main():
    
    #countvect = joblib.load(open('count_vect.pkl', 'rb'))
    countvect  = pickle.load(open('count_vect.pkl', 'rb'))
    model = pickle.load(open('naive_bayes_model.pkl', 'rb'))
    st.title('Spam Classifier')
    text = st.text_area('Enter verification text')
    if st.button('Predict'):
        
        text = [text]
        text = countvect.transform(text)
        result = model.predict(text)
        
        if result == [0]:
            st.write('This message is NOT SPAM')
        else:
            st.write('This message is SPAM!!')
    
    
if __name__ == '__main__':
    main()