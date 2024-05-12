import streamlit as st
from pyaspeller import YandexSpeller
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from func import *


#модели для проверки энциклопедичности
tokenizer_enciclopedic = AutoTokenizer.from_pretrained("bert-base-uncased")
#декоратор для загрузки модели в кэш
@st.cache_resource
def load_model_enciclopedic():
    model_enciclopedic = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model_enciclopedic.load_state_dict(torch.load('weights_bert_with_val', map_location=torch.device(device)))
    if torch.cuda.is_available():
        model_enciclopedic.cuda()
    return model_enciclopedic

model_enciclopedic = load_model_enciclopedic()

prompt = "Учитывая следующий текст, определите, написан ли он в энциклопедическом стиле:\n\nТекст:"


#модель для проверки соответствия Wiki-разметеке
tokenizer_wiki = AutoTokenizer.from_pretrained('MediaWiki_classifier')
@st.cache_resource
def load_model_wiki():
    model_wiki = AutoModelForSequenceClassification.from_pretrained('MediaWiki_classifier')
    if torch.cuda.is_available():
        model_wiki.cuda()
    return model_wiki

model_wiki = load_model_wiki()




#модели для проверки нейтральности
tokenizer_neutrality = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny-sentiment-balanced')
@st.cache_resource
def load_model_neutrality():
    model_neutrality = AutoModelForSequenceClassification.from_pretrained('cointegrated/rubert-tiny-sentiment-balanced')
    if torch.cuda.is_available():
        model_neutrality.cuda()
    return model_neutrality

model_neutrality =  load_model_neutrality()


#Загрузка и очистка данных

data = st.text_area('Введите текcт', 'text input')
clean_data = clean_up(str(data))

option = st.selectbox('Выберите проверку',
    ('', 'Энциклопедичность', 'Соответствие Wiki разметке', 'Исправление ошибок', 'Нейтральность', 'Плагиат'))

if option == 'Энциклопедичность':
    enciclopedic=enciclopedic(data, tokenizer_enciclopedic, model_enciclopedic, prompt)
    if enciclopedic =='Текст написан в энциклопедическом стиле':
        st.write(f':green[{enciclopedic}]')
    else:
        st.write(f':red[{enciclopedic}]')

if option == 'Соответствие Wiki разметке':
    wiki = is_mediawiki_markup(data)
    if wiki == True:
        st.write(f':green[{wiki}]')
    else:
        st.write(f':red[{wiki}]')

elif option == 'Исправление ошибок':
    speller = YandexSpeller()
    data_correct = []
    for i in range(0, len(clean_data), 500):
       data_correct.append(speller.spelled(clean_data[i:i+500]))
    data_correct = ' '.join(data_correct)
    if clean_data == data_correct:
        st.write(':green[Нет ошибок]')
    else:
        st.markdown(':red[Статья содержит ошибки]')
        st.write(f'Исправлены слова: :blue[{set(clean_data.split()) - set(data_correct.split())}]')
        st.text_area('Исправленный текст:', data_correct)



elif option == 'Нейтральность':
    label = get_sentiment(clean_data, tokenizer_neutrality, model_neutrality, 'label')
    score = get_sentiment(clean_data, tokenizer_neutrality, model_neutrality, 'score')
    proba = get_sentiment(clean_data, tokenizer_neutrality, model_neutrality, 'proba')
    st.write(label, ' на ', score)


elif option == 'Плагиат':
    plagiarism = plagiarism(clean_data)
    st.write(plagiarism)

else:
    st.write('Не выбрана ни одна проверка!!')



