# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 21:34:37 2021

@author: Ryan
"""


import streamlit as st
import joblib
import pandas

st.set_option('deprecation.showfileUploaderEncoding', False)

st.set_option('deprecation.showPyplotGlobalUse', False)


def main():
    
    countvect = joblib.load(open('count_vect.pkl', 'rb'))
    model = joblib.load(open('naive_bayes_model.pkl', 'rb'))
    st.title('Spam Classifier')
    text = st.text_area('Enter verification text')
    if st.button('Predict'):
        text = countvect.transform(text)
        result = model.predict(text)
        result= str(result)
        st.write('The text is : ', result)
    
    
    
if __name__ == '__main__':
    main()