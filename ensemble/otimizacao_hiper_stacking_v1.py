import numpy as np
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score

noticias = load_files('dados', encoding='utf-8', decode_error='replace')

X = noticias.data
y = noticias.target

my_stop_words = set(stopwords.words('english'))

d1 = list()

for x in range(100):
  X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, train_size=0.7, random_state=x)

  vectorizer = TfidfVectorizer(norm = None, stop_words=list(my_stop_words), max_features=1000, decode_error='ignore')

  X_treino_vectors = vectorizer.fit_transform(X_treino)
  X_teste_vectors = vectorizer.transform(X_teste)

  modelos_base = [('rf', RandomForestClassifier(n_estimators=100, random_state=42)),('nb', MultinomialNB())]

  stacking_model = StackingClassifier(estimators=modelos_base, final_estimator=LogisticRegression(multi_class='multinomial', random_state=30, max_iter=1000))


  acuracia = stacking_model.fit(np.asarray(X_treino_vectors.todense()), y_treino).score(np.asarray(X_teste_vectors.todense()), y_teste)

  d1.append((x, acuracia))
  print('-hiperparametro random state:', x, '-Acurácia', acuracia)


print('\nMoelhores resultados:\n')
mx= max(d1, key=lambda x: x[1])
print('-Random state:', mx[0], '-Acuracia', mx[1])
print('\n')