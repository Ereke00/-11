import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor

np.random.seed(0)
floor = np.random.randint(1, 11, 100)  # Этаж
area = np.random.randint(20, 160, 100)  # Ауданы
price = np.random.randint(50000, 70000, 100)  # Квартира жалдау куны(тенгеде)

X = np.column_stack((floor, area))
y = price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

base_models = [
    ('линейная регрессия', LinearRegression()),
    ('случайный лес', RandomForestRegressor(random_state=1))
]

stacking_model = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())

for name, model in base_models:
    model.fit(X_train, y_train)

stacking_model.fit(X_train, y_train)

base_models_predictions = {name: model.predict(X_test) for name, model in base_models}
stacking_model_prediction = stacking_model.predict(X_test)

for name, predictions in base_models_predictions.items():
    mse = np.mean((y_test - predictions)**2)
    print(f"MSE модели '{name}': {mse}")

stacking_mse = np.mean((y_test - stacking_model_prediction)**2)
print(f"Stacking MSE: {stacking_mse}")

plt.figure(figsize=(8, 6))

for name, predictions in base_models_predictions.items():
    plt.scatter(y_test, predictions, label=name, alpha=0.7)

plt.scatter(y_test, stacking_model_prediction, label='Стекинг', alpha=0.7)
plt.plot(y_test, y_test, color='red', linestyle='--', label='Идеальное предсказание')

plt.xlabel('Фактическая цена аренды (тенге)')
plt.ylabel('Предсказанная цена аренды (тенге)')
plt.title('Сравнение фактических и предсказанных цен аренды')
plt.legend()
plt.show()
