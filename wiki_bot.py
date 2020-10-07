import bs4 as bs
import urllib.request
import re
import nltk
import random
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
import spacy
from flask import Flask, request, jsonify
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

TOKEN_API = config['movidesk']['TOKEN_API']

dados = urllib.request.urlopen('https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial')
dados = dados.read()
dados_html = bs.BeautifulSoup(dados, 'lxml')
paragrafos = dados_html.find_all('p')

conteudo = ''
for p in paragrafos:
    conteudo += p.text

conteudo = conteudo.lower()
lista_sentencas = nltk.sent_tokenize(conteudo)

pln = spacy.load('pt_core_news_sm')
stop_words = spacy.lang.pt.stop_words.STOP_WORDS

def preprocessamento(texto):
    # URLS
    texto = re.sub(r'https?://[A-za-z0-9./]+', ' ', texto)

    # Espacos em branco
    texto = re.sub(r' +', ' ', texto)
    documento = pln(texto)
    lista = []

    for token in documento:
        lista.append(token.lemma_)

    lista = [palavra for palavra in lista if palavra not in stop_words and palavra not in string.punctuation]
    lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()])

    return lista

lista_sentencas_preprocessada = []
for i in range(len(lista_sentencas)):
    lista_sentencas_preprocessada.append(preprocessamento(lista_sentencas[i]))

"""for _ in range(5):
    i = random.randint(0, len(lista_sentencas) - 1)
    print(lista_sentencas[i])
    print(lista_sentencas_preprocessada[i])
    print('-------------')
"""
"""Etapa 3 Frases de boas-vindas"""

texto_boas_vindas = ('hey', 'olá', 'opa', 'oi', 'eae')
texto_boas_vindas_respostas = ('Oi tudo bem?', 'olá', 'bem-vindo', 'oi', 'como você está?')


def responder_saudacao(texto):
    for palavra in texto.split():
        if palavra.lower() in texto_boas_vindas:
            return random.choice(texto_boas_vindas_respostas)


responder_saudacao('oi')

"""# 4 Entendimento TF-IDF(Team frequency - inverse documento frequency)"""



def responder(texto_usuario):
    resposta_chatbot = ''
    lista_sentencas_preprocessada.append(texto_usuario)

    tfidf = TfidfVectorizer()
    palavras_vetorizadas = tfidf.fit_transform(lista_sentencas_preprocessada)

    similaridade = cosine_similarity(palavras_vetorizadas[-1], palavras_vetorizadas)

    indice_sentenca = similaridade.argsort()[0][-2]
    vetor_similar = similaridade.flatten()
    vetor_similar.sort()
    vetor_encontrado = vetor_similar[-2]

    if (vetor_encontrado == 0):
        resposta_chatbot = resposta_chatbot + 'Desculpe, mas não entendi!'
        return resposta_chatbot
    else:
        resposta_chatbot = resposta_chatbot + lista_sentencas[indice_sentenca]
        return resposta_chatbot

app = Flask(__name__)
@app.route("/<string:txt>", methods=["POST"])
def conversar(txt):
    resposta = ''
    texto_usuario = txt
    texto_usuario = texto_usuario.lower()
    if (responder_saudacao(texto_usuario) != None):
        resposta = responder_saudacao(texto_usuario)
    else:
        resposta = responder(preprocessamento(texto_usuario))
        lista_sentencas_preprocessada.remove(preprocessamento(texto_usuario))
    return jsonify({"texto_respondido": resposta})

app.run(port=5000, debug=False)
