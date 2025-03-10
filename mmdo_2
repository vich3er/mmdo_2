import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog


def get_input_vector(prompt, size=None):
    """Функція для введення числового вектора"""
    while True:
        try:
            vector = list(
                map(float, input(f"{prompt} (через пробіл): ").split()))
            if size and len(vector) != size:
                print(f"Помилка: Очікується {size} значень!")
                continue
            return vector
        except ValueError:
            print("Помилка: Введіть числові значення!")


def parse_constraints():
    """Функція для введення обмежень та їх коректного розбору"""
    A_ub, b_ub = [], []
    A_eq, b_eq = [], []

    m = int(input("Введіть кількість обмежень: "))

    for i in range(m):
        while True:
            try:
                constraint = input(
                    f"Введіть {i+1}-е обмеження (коефіцієнти, знак <=, >=, =, права частина): ")
                parts = constraint.split()

                *lhs, sign, rhs = parts
                lhs = np.array(list(map(float, lhs)), dtype=float)
                rhs = float(rhs)

                if sign == "<=":
                    A_ub.append(lhs)
                    b_ub.append(rhs)
                elif sign == ">=":
                    A_ub.append(-lhs)
                    b_ub.append(-rhs)
                elif sign == "=":
                    A_eq.append(lhs)
                    b_eq.append(rhs)
                else:
                    print("Помилка: Використовуйте тільки <=, >= або =")
                    continue

                break
            except ValueError:
                print(
                    "Помилка: Некоректний ввід! Переконайтесь, що всі значення числові.")

    return np.array(A_ub), np.array(b_ub), np.array(A_eq), np.array(b_eq)


while True:
    mode = input("Виберіть тип оптимізації (max/min): ").strip().lower()
    if mode in ["max", "min"]:
        break
    print("Помилка: Введіть 'max' або 'min'!")

c = get_input_vector("Введіть коефіцієнти цільової функції")

A_ub, b_ub, A_eq, b_eq = parse_constraints()

c = np.array(c)
if mode == "max":
    c = -c

res = linprog(c, A_ub=A_ub if A_ub.size else None, b_ub=b_ub if b_ub.size else None,
              A_eq=A_eq if A_eq.size else None, b_eq=b_eq if b_eq.size else None, method='highs')

if res.success:
    x_opt = res.x
    f_opt = res.fun if mode == "min" else -res.fun
    print("\n Оптимальний розв’язок:", x_opt)
    print(" Значення цільової функції:", f_opt)
    print(" Кількість ітерацій:", res.nit)
else:
    print("\n Розв’язок не знайдено!", res.message)

if len(c) == 2:
    x1 = np.linspace(0, max(b_ub) if b_ub.size else 10, 400)
    x2 = np.linspace(0, max(b_ub) if b_ub.size else 10, 400)
    X1, X2 = np.meshgrid(x1, x2)
    fig, ax = plt.subplots()

    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#17becf']

    for i in range(len(A_ub)):
        a1, a2 = A_ub[i]
        b = b_ub[i]
        color = colors[i % len(colors)]

        if a2 != 0:
            x2_vals = (b - a1 * x1) / a2

            if a2 > 0:
                ax.fill_between(x1, -1000, x2_vals,
                                alpha=0.3, color=color, label=f"{a1}x1 + {a2}x2 ≤ {b}")
            else:
                ax.fill_between(x1, x2_vals, 1000,
                                alpha=0.3, color=color, label=f"{a1}x1 + {a2}x2 ≥ {b}")

            ax.plot(x1, x2_vals, color=color)

        else:
            x_fixed = b / a1
            ax.axvline(x=x_fixed, linestyle='--', color=color)
            if a1 > 0:
                ax.fill_betweenx(x2, -1000, x_fixed,
                                 alpha=0.3, color=color, label=f"{a1}x1 ≤ {b}")
            else:
                ax.fill_betweenx(x2, x_fixed, 1000,
                                 alpha=0.3, color=color, label=f"{a1}x1 ≥ {b}")

    if res.success:
        x_opt = res.x
        ax.scatter(x_opt[0], x_opt[1], color='red',
                   s=100, label='Оптимальна точка')

    if res.success:
        plt.xlim(min(0, x_opt[0]-5), max(x_opt[0]+5, max(b_ub)))
        plt.ylim(min(0, x_opt[1]-5), max(x_opt[1]+5, max(b_ub)))
    else:
        plt.xlim(0, max(b_ub) / 2)
        plt.ylim(0, max(b_ub) / 2)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.title("Область допустимих розв’язків та оптимальна точка")
    plt.grid()
    plt.savefig("optimal_solution.png", bbox_inches='tight')
    print("Графік збережено у файлі 'optimal_solution.png'")
