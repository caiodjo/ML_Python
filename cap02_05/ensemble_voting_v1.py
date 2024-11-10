#versão 1 - Voting
import numpy as np
from nltk.corpus import stopwords
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

#Carregando os dados
#load_files já entende a organizacao em pastas.
noticias = load_files('dados', encoding = 'utf-8', decode_error='replace')

#Separando variáveis de entrada e saída
X = noticias.data
y = noticias.target

#lista de stopwords
my_stop_words = set(stopwords.words('english'))

#Divisão em treino e teste (70/30)
X_treino, X_teste, y_treino, y_teste = train_test_split(X,y, test_size=0.3, random_state=93)

#Vetorização
vectorizer = TfidfVectorizer(norm= None, stop_words=list(my_stop_words), max_features=1000, decode_error="ignore")

#Aplicamos a vetorização.
#Observe que treinamos e aplicamos em treino e apenas aplicas em teste
X_treino_vectors = vectorizer.fit_transform(X_treino) #Treina e extrai os vetores de palavras
X_teste_vectors = vectorizer.transform(X_teste)

#Criando 3 modelos com 3 algoritimos diferentes
modelo1 = LogisticRegression(multi_class= 'multinomial', solver='lbfgs', random_state=30, max_iter=1000)

modelo2 = RandomForestClassifier(n_estimators=1000, max_depth=100, random_state=1)

modelo3 = MultinomialNB()

#Lista para o resultado
resultado = list()

#iniciando o modelo de votação
voting_model = VotingClassifier(estimators=[('lg', modelo1), ('rf', modelo2), ('nb',modelo3)], voting='soft')
print("\nModelo de votacao:\n")
print(voting_model)

#Treinamento
voting_model = voting_model.fit(X_treino_vectors, y_treino)

#Previsoes com dados de teste
previsoes = voting_model.predict(X_teste_vectors)

#Grava o resultado
resultado.append(accuracy_score(y_teste, previsoes))

#Print
print("\nAcurácia do Modelo:", accuracy_score(y_teste, previsoes), "\n")
print("\n")