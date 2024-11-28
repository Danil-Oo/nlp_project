import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
model_name_or_path = "models/folder"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(DEVICE)

st.title('Генерация текста GPT-моделью по пользовательскому prompt')
st.divider()

st.subheader('Информация о модели')
st.write('Модель: sberbank-ai/rugpt3small_based_on_gpt2')
st.write('Число эпох: 200')
st.write('Время обучения: 54 мин.')
st.write('Датасет: какой-то красноречивый рассказ журналиста о своей деятельности')
st.divider()

st.subheader('Блок генерации')
text = st.text_area('Введите текст для генерации',
                    'Я сделал модель, которая криво работает')
input_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
num_beams = st.slider('Выберите num_beams', 1, 10, 3, 1)
temperature = st.slider('Выберите temperature', 0.8, 2.0, 1.0, 0.2)
top_k = st.slider('Выберите top_k', 2, 10, 3, 1)
top_p = st.slider('Выберите top_p', 0.5, 0.9, 0.7, 0.1)
max_length = st.slider('Выберите максимальную длину', 5, 100, 15, 5)

model.eval()
with torch.no_grad():
    out = model.generate(input_ids,
                         do_sample=True,
                         num_beams=num_beams,
                         temperature=temperature,
                         top_k=top_k,
                         top_p=top_p,
                         max_length=max_length,
                         no_repeat_ngram_size=2
                         )

generated_text = list(map(tokenizer.decode, out))[0]

if st.button('Сгенерировать текст'):
    st.write(generated_text)
