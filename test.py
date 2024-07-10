import streamlit as st
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import sacrebleu

# Load the model and tokenizer
model_name = "facebook/m2m100_418M"
model = M2M100ForConditionalGeneration.from_pretrained(model_name)
tokenizer = M2M100Tokenizer.from_pretrained(model_name)

# List of available language codes
available_languages = {
    'Arabic': 'ar',
    'Chinese': 'zh',
    'French': 'fr',
    'German': 'de',
    'Hindi': 'hi',
    'Italian': 'it',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Portuguese': 'pt',
    'Russian': 'ru',
    'Spanish': 'es',
    'Turkish': 'tr',
    'English': 'en'
}

def translate(text, src_lang, tgt_lang):
    # Set the source language
    tokenizer.src_lang = src_lang
    # Encode the input text
    encoded_input = tokenizer(text, return_tensors="pt")
    # Generate the translated tokens
    generated_tokens = model.generate(**encoded_input, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
    # Decode the tokens to get the translated text
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated_text

def calculate_bleu_score(reference, translation):
    bleu = sacrebleu.corpus_bleu([translation], [[reference]])
    return bleu.score

def main():
    st.title("Translation App with BLEU Score Calculation")
    
    st.sidebar.header("Translation Settings")
    source_lang = st.sidebar.selectbox("Select source language", list(available_languages.keys()))
    target_lang = st.sidebar.selectbox("Select target language", list(available_languages.keys()))

    source_code = available_languages[source_lang]
    target_code = available_languages[target_lang]

    text_to_translate = st.text_area("Enter the text to translate:")
    reference_translation = st.text_area("Enter the reference translation (optional):")

    if st.button("Translate"):
        if text_to_translate:
            translated_text = translate(text_to_translate, source_code, target_code)
            st.write("Translated text:")
            st.write(translated_text)
            
            if reference_translation:
                bleu_score = calculate_bleu_score(reference_translation, translated_text)
                st.write(f"BLEU score: {bleu_score:.2f}")
        else:
            st.write("Please enter text to translate.")

if __name__ == "__main__":
    main()
