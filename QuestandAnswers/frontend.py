# import streamlit as st
# import requests

# st.title("Q&A Model")
# st.write("Enter your question below:")

# question = st.text_input("Question:")

# if st.button("Submit"):
#     if question:
#         response = requests.post("http://localhost:8000/predict", json={"text": question}, timeout=10)
#         if response.status_code == 200:
#             answer = response.json()["answer"]
#             st.write(f"Answer: {answer}")
#         else:
#             st.write("Error: Unable to get a response from the API.")
#     else:
#         st.write("Please enter a question.")

import streamlit as st
import requests

st.title("Q&A Model")
st.write("Enter your question and passage below:")

question = st.text_input("Question:")
passage = st.text_area("Passage:")

if st.button("Submit"):
    if question and passage:
        response = requests.post("http://localhost:8000/predict", json={"question": question, "passage": passage}, timeout=10)
        if response.status_code == 200:
            answer = response.json()["answer"]
            st.write(f"Answer: {answer}")
        else:
            st.write("Error: Unable to get a response from the API.")
    else:
        st.write("Please enter both a question and a passage.")
