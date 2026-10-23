
import numpy as np

# Примеры для обучения: вход и правильный ответ
X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([[2.0], [4.0], [6.0], [8.0]])

# Начальные параметры сети
np.random.seed(42)
W = np.random.randn(1, 1) * 0.1
b = np.zeros((1, 1))

learning_rate = 0.01

# Обучение
for epoch in range(5000):
    prediction = X @ W + b
    error = prediction - y

    # Вычисляем, как нужно изменить параметры
    dW = (2 / len(X)) * (X.T @ error)
    db = (2 / len(X)) * np.sum(error, axis=0, keepdims=True)

    # Обновляем параметры
    W -= learning_rate * dW
    b -= learning_rate * db

    if (epoch + 1) % 1000 == 0:
        prediction_check = X @ W + b
        loss = np.mean((prediction_check - y) ** 2)
        print(f"Эпоха {epoch + 1}: MSE = {loss:.6f}")

# Проверяем, чему научилась сеть
print("\nПроверка:")
for value in [1.0, 2.0, 3.0, 4.0, 5.0]:
    pred = np.array([[value]]) @ W + b
    print(f"{value:.0f} -> {pred[0, 0]:.3f}")

print(f"\nВыученный вес W: {W[0, 0]:.4f}")
print(f"Выученное смещение b: {b[0, 0]:.4f}")
