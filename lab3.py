import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 0.1, 0.2, 0.5, 0.9, 1.5, 2.9, 3, 3.3])
y = np.array([4.706, 4.332, 4.8769, 5.5462, 6.797,
              9.3235, 13.3948, 13.3171, 14.8502])

true_a = 4
true_b = 3


def least_squares(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)

    b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    a = (sum_y - b * sum_x) / n

    return a, b


def compute_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_standard_errors(x, y, a, b):
    n = len(x)
    y_pred = a + b * x
    S = np.sum((y - y_pred) ** 2)

    sum_x = np.sum(x)
    sum_x2 = np.sum(x**2)

    denominator = (n - 2) * (sum_x2 - (sum_x**2) / n)
    delta_b = np.sqrt(S / denominator)
    delta_a = delta_b * np.sqrt(sum_x2 / n)

    return delta_a, delta_b


a_manual, b_manual = least_squares(x, y)
y_pred_manual = a_manual + b_manual * x
error_manual = compute_error(y, y_pred_manual)
delta_a_manual, delta_b_manual = compute_standard_errors(
    x, y, a_manual, b_manual)

A = np.vstack([np.ones_like(x), x]).T
a_lstsq, b_lstsq = np.linalg.lstsq(A, y, rcond=None)[0]
y_pred_lstsq = a_lstsq + b_lstsq * x
error_lstsq = compute_error(y, y_pred_lstsq)
delta_a_lstsq, delta_b_lstsq = compute_standard_errors(x, y, a_lstsq, b_lstsq)

manual_a_error = abs(a_manual - true_a)
manual_b_error = abs(b_manual - true_b)
lstsq_a_error = abs(a_lstsq - true_a)
lstsq_b_error = abs(b_lstsq - true_b)

y_true = true_a + true_b * x
rmse_manual = compute_error(y_true, y_pred_manual)
rmse_lstsq = compute_error(y_true, y_pred_lstsq)

print(
    f"Ручний метод:   a = {a_manual:.4f} ± {delta_a_manual:.4f}, b = {b_manual:.4f} ± {delta_b_manual:.4f}, RMSE = {error_manual:.4f}")
print(
    f"np.linalg.lstsq: a = {a_lstsq:.4f} ± {delta_a_lstsq:.4f}, b = {b_lstsq:.4f} ± {delta_b_lstsq:.4f}, RMSE = {error_lstsq:.4f}")

print("\nПорівняння з істинними значеннями (a=4, b=3):")
print(
    f"Ручний метод - Різниця a: {manual_a_error:.6f}, Різниця b: {manual_b_error:.6f}, RMSE = {rmse_manual:.6f}")
print(
    f"np.linalg.lstsq - Різниця a: {lstsq_a_error:.6f}, Різниця b: {lstsq_b_error:.6f}, RMSE = {rmse_lstsq:.6f}")

plt.scatter(x, y, color='blue', label='Вхідні дані')

plt.plot(x, y_pred_manual, color='red',
         label=f'Ручний метод: y={a_manual:.2f} + {b_manual:.2f}x')

plt.plot(x, y_pred_lstsq, color='green', linestyle='--',
         label=f'lstsq: y={a_lstsq:.2f} + {b_lstsq:.2f}x')

plt.plot(x, y_true, color='orange', linestyle='-.',
         label=f'Істинна лінія: y=4 + 3x')

plt.xlabel("x")
plt.ylabel("y")
plt.title("Порівняння методів апроксимації")
plt.legend()
plt.grid(True)

plt.savefig("comparison_plot.png", bbox_inches='tight')
print("Графік збережено у файлі 'comparison_plot.png'")
