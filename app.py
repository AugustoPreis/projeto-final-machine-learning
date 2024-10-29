import pandas as pd
from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# Carrega o dataset
dataset = pd.read_csv('./dataset.csv', encoding='latin-1')

# Altera a classificação (spam/ham) para 0/1
dataset['category'] = dataset['Category'].map({ 'ham': 1, 'spam': 0 })
dataset['message'] = dataset['Message']

# Renomeia as colunas
dataset = dataset[['category', 'message']]

# Define features e target
X = dataset['message']
Y = dataset['category']

# Divide em conjunto de treino e teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Modelos utilizados
models = {
  'LogisticRegression': (LogisticRegression(), { 'classifier__C': [0.1, 1.0, 10.0] }),
  'RandomForest': (RandomForestClassifier(), { 'classifier__n_estimators': [5, 10, 20] }),
  'GradientBoosting': (GradientBoostingClassifier(), { 'classifier__n_estimators': [5, 10, 20] })
}

#Variáveis que salvam informações do melhor modelo
best_model = None
best_model_name = ''
best_model_accuracy = 0

# Loop para treinar e buscar os melhores parâmetros de cada modelo
for model_name, (model, params) in models.items():
  print(f'Treinando modelos de {model_name}...\n')

  # Cria pipeline com o modelo
  pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)),
    ('classifier', model)
  ])

  # Grade de parâmetros do grid search
  param_grid = {
    'vectorizer__min_df': [1, 2],
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    **params
  }

  # Cria grid search com o pipeline e a grade de parâmetros
  grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=0, n_jobs=-1)
  grid_search.fit(X_train, Y_train)

  # Melhor modelo encontrado
  best_model_grid = grid_search.best_estimator_
  Y_pred = best_model_grid.predict(X_test)
  accuracy = accuracy_score(Y_test, Y_pred)

  # Caso o modelo atual seja o melhor já encontrado, atualiza as variáveis
  if (accuracy > best_model_accuracy):
    best_model_name = model_name
    best_model = best_model_grid
    best_model_accuracy = accuracy

  print(f'Melhor modelo {model_name} possui precisão de: {accuracy:.4f}\n')

print(f'Melhor modelo encontrado foi {best_model_name} com precisão de: {best_model_accuracy:.4f}\n')

app = Flask(__name__)

# A rota recebe um objeto json com a propriedade 'conteudo'
# e retorna se aquele conteúdo é spam ou não
@app.route('/predict', methods=['POST'])
def predict():
  try:
    # Conteúdo enviado na requisição
    conteudo = request.json['conteudo']

    # Predição: 0 para spam e 1 para não spam
    prediction = best_model.predict([conteudo])

    # Formata o retorno
    spam = 'Sim' if prediction[0] == 0 else 'Não'

    return jsonify({ 'spam': spam }), 200
  except Exception as error:
    return jsonify({ 'error': str(error) }), 500

if __name__ == '__main__':
  app.run(debug=True, use_reloader=False)