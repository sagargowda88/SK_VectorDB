# streamlit_app.py
import streamlit as st
from vector_store import VectorStore
import pandas as pd

st.title("CSV File Query App")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, encoding_errors="ignore")
    st.write("Data Preview:")
    st.write(data.head())

    cols_to_index = data.columns
    
    vector_store = VectorStore()
    vector_store.index_data(data, cols_to_index)
    st.success("Data indexed successfully!")

    query = st.text_input("Enter Query", "")

    if query:
        try:
            st.write("Top 5:")
            result_ids = [uid for uid, _ in vector_store.find_similar_vectors(query, 5)]
            result_df = data.loc[result_ids].reset_index(drop=True)
            st.write(result_df)
        except Exception as e:
            st.error(f"An error occurred: {e}")
