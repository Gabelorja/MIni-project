import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle

st.set_page_config(page_title="Text Classification", layout="wide")

st.title("Text Classification System")
st.write("Upload your TSV file to get predictions")

# File uploader
uploaded_file = st.file_uploader("Choose a TSV file", type=['tsv', 'txt'])

if uploaded_file is not None:
    try:
        # Read the file
        df = pd.read_csv(uploaded_file, sep='\t', header=None, names=['text', 'label'])

        st.success(f"File loaded successfully! {len(df)} rows found.")

        # Show preview
        with st.expander("Preview data (first 5 rows)"):
            st.dataframe(df.head())

        # Load model and tokenizer
        with st.spinner("Loading model..."):
            model = keras.models.load_model('text_classification_model.h5')
            with open('tokenizer.pkl', 'rb') as f:
                tokenizer = pickle.load(f)

        st.success("Model loaded!")

        # Make predictions
        if st.button("Analyze Text", type="primary"):
            with st.spinner("Processing..."):
                # Preprocess
                sequences = tokenizer.texts_to_sequences(df['text'])
                max_length = 100
                padded_sequences = keras.preprocessing.sequence.pad_sequences(
                    sequences,
                    maxlen=max_length,
                    padding='post'
                )

                # Predict
                predictions = model.predict(padded_sequences)
                predicted_classes = (predictions > 0.5).astype(int).flatten()

                # Calculate results
                total = len(df)
                correct = np.sum(predicted_classes == df['label'].values)
                incorrect = total - correct
                accuracy = (correct / total) * 100

                # Display results
                st.header("Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Samples", total)

                with col2:
                    st.metric("Correct Predictions", f"{correct} ({accuracy:.2f}%)")

                with col3:
                    st.metric("Incorrect Predictions", f"{incorrect} ({100-accuracy:.2f}%)")

                # Show detailed results
                with st.expander("View detailed predictions"):
                    results_df = pd.DataFrame({
                        'Text': df['text'],
                        'True Label': df['label'],
                        'Predicted Label': predicted_classes,
                        'Confidence': predictions.flatten(),
                        'Correct': predicted_classes == df['label'].values
                    })
                    st.dataframe(results_df, use_container_width=True)

                # Show some examples
                st.subheader("Sample Predictions")

                col_left, col_right = st.columns(2)

                with col_left:
                    st.write("✅ **Correct Predictions (5 samples)**")
                    correct_samples = results_df[results_df['Correct'] == True].head(5)
                    for idx, row in correct_samples.iterrows():
                        st.text(f"Text: {row['Text'][:100]}...")
                        st.text(f"Label: {row['True Label']} | Confidence: {row['Confidence']:.4f}")
                        st.divider()

                with col_right:
                    st.write("❌ **Incorrect Predictions (5 samples)**")
                    incorrect_samples = results_df[results_df['Correct'] == False].head(5)
                    if len(incorrect_samples) > 0:
                        for idx, row in incorrect_samples.iterrows():
                            st.text(f"Text: {row['Text'][:100]}...")
                            st.text(f"True: {row['True Label']} | Predicted: {row['Predicted Label']} | Conf: {row['Confidence']:.4f}")
                            st.divider()
                    else:
                        st.write("No incorrect predictions!")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Make sure your file is in the correct format (TSV with text and label columns)")
else:
    st.info("👆 Please upload a TSV file to begin")
    st.write("**Expected format:**")
    st.code("text\\tlabel\\nSample text here\\t0\\nAnother sample\\t1")
