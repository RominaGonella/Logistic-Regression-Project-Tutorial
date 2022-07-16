# librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
import pickle

# se carga archivo de la web
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv', delimiter=';')

# se guarda en la carpeta data/raw
df_raw.to_csv('../data/raw/dataset_bank.csv')

# se carga desde la carpeta
df_bank = pd.read_csv('../data/raw/dataset_bank.csv', index_col = 0)

### sustitución de categorías "unknown" en categóricas

# Tipo de trabajo
print(df_bank['job'].value_counts())
print(f'Cantidad de categorías: {df_bank["job"].nunique()}')

# sustituyo categoría "unknown" por la más frecuente
mask = df_bank['job'] == 'unknown'
df_bank.loc[mask, 'job'] = 'admin.'
print(df_bank['job'].value_counts())

# Estado civil
print(df_bank['marital'].value_counts())
print(f'Cantidad de categorías: {df_bank["marital"].nunique()}')

# sustituyo categoría "unknown" por la más frecuente
mask = df_bank['marital'] == 'unknown'
df_bank.loc[mask, 'marital'] = 'married'
print(df_bank['marital'].value_counts())

# Nivel educativo
print(df_bank['education'].value_counts())
print(f'Cantidad de categorías: {df_bank["education"].nunique()}')

# sustituyo categoría "unknown" por la más frecuente
mask = df_bank['education'] == 'unknown'
df_bank.loc[mask, 'education'] = 'university.degree'
print(df_bank['education'].value_counts())

# Tiene créditos en default?
print(df_bank['default'].value_counts())
print(f'Cantidad de categorías: {df_bank["default"].nunique()}')

# sustituyo categoría "unknown" por la más frecuente
mask = df_bank['default'] == 'unknown'
df_bank.loc[mask, 'default'] = 'no'
print(df_bank['default'].value_counts())

# Tiene préstamo de vivienda?
print(df_bank['housing'].value_counts())
print(f'Cantidad de categorías: {df_bank["housing"].nunique()}')

# sustituyo categoría "unknown" por la más frecuente
mask = df_bank['housing'] == 'unknown'
df_bank.loc[mask, 'housing'] = 'yes'
print(df_bank['housing'].value_counts())

# Tiene algún préstamo?
print(df_bank['loan'].value_counts())
print(f'Cantidad de categorías: {df_bank["loan"].nunique()}')

# sustituyo categoría "unknown" por la más frecuente
mask = df_bank['loan'] == 'unknown'
df_bank.loc[mask, 'loan'] = 'no'
print(df_bank['loan'].value_counts())

# convierto a categóricas
cols = df_bank.select_dtypes(include = ['object']).columns
for i in cols:
    df_bank[i] = pd.Categorical(df_bank[i])
df_bank.info()

# dropeo algunas variables
df_bank = df_bank.drop(columns= ['pdays', 'emp.var.rate', 'nr.employed'], axis = 1)

# convierto target en binario
df_bank['y']=df_bank['y'].cat.codes
df_bank['y'].value_counts()

# creo dummies para las variables categóricas
df_bank = pd.get_dummies(df_bank, columns=['job',	'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'range_age'], drop_first = True)

# escalo los datos, uso MinMaxScaler
scaler = MinMaxScaler()
train_scaler = scaler.fit(df_bank[['age', 'duration', 'campaign', 'previous', 'cons.price.idx', 'cons.conf.idx', 'euribor3m']])
df_bank[['age', 'duration', 'campaign', 'previous', 'cons.price.idx', 'cons.conf.idx', 'euribor3m']] = train_scaler.transform(df_bank[['age', 'duration', 'campaign', 'previous', 'cons.price.idx', 'cons.conf.idx', 'euribor3m']])

# guardo data frame pronto para modelar
df_bank.to_csv('../data/processed/dataset_bank_proccesed.csv')

#### Estimación del modelo

# separo en X e y, elimino rangos de edad de X (mantengo la numérica)
X = df_bank.drop(columns = ['y', 'range_age_(20, 30]', 'range_age_(30, 40]', 'range_age_(40, 50]', 'range_age_(50, 60]', 'range_age_(60, 70]', 'range_age_(70, 80]', 'range_age_(80, 90]', 'range_age_(90, 100]'])
y = df_bank['y']

# estimo modelo de regresión logística (parámetros por defecto, sólo aumenté max_iter no eran suficientes las que vienen por defecto)
model = LogisticRegression(max_iter = 200)
model.fit(X, y)

# cross validation para evaluar el modelo

# Adaptación de una función del material de clase
def cross_validation(model, _X, _y, _scoring, _cv=5):

    results = cross_validate(estimator=model,
                                X=_X,
                                y=_y,
                                cv=_cv,
                                scoring=_scoring,
                                return_train_score=True)
                                
    return {"Train score": results['train_score'],
              "Mean Train score": results['train_score'].mean(),
              "Validation score": results['test_score'],
              "Mean Validation score": results['test_score'].mean()
              }         
    
cross_validation(model, X, y, 'recall')

# grid search para buscar mejores hiperparámetros

# defino parametros
solvers = ['newton-cg', 'lbfgs', 'liblinear']
c_values = [100, 10, 1.0]
max_iter = [200, 250, 300]

# defino grid search
grid = dict(solver=solvers, C=c_values, max_iter = max_iter)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='recall',error_score=0)
grid_result = grid_search.fit(X, y)

# resumo resultados
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# usando la mejor combinación de hiperparámetros, estimo modelo final
best_param = grid_result.best_params_
best_model = LogisticRegression(**best_param)
best_model.fit(X, y)

# guardo modelo
filename = '../models/finalized_model.sav'
pickle.dump(best_model, open(filename, 'wb'))