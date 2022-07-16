# Resumen del proceso

1. Se trabaja en archivo "explore.ipynb" con una base de datos de entrenamiento, por lo que se realiza el EDA sobre esa base sin separar en train y test.
2. Se estima modelo de regresión logística, se prueba cuales variables incluir
3. Se evalua modelo inicial mediante cross-validation, como el objetivo es identificar los clientes que podrían suscribir un depósito a plazo (y=1), se utiliza como métrica el recall
4. se aplica grid search para buscar la mejor combinación de parámetros, una vez que se obtiene se vuelve a estimar el modelo y se guarda
5. se pasa al archivo "app.py" solamente el código imprescindible