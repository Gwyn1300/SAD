# -*- coding: utf-8 -*-
"""
ЛАБОРАТОРНАЯ РАБОТА: Проверка статистических гипотез о нормальности, расчет мощности и критерии согласия

ОБЩАЯ ИНСТРУКЦИЯ ДЛЯ СТУДЕНТОВ:
1. Запустите этот скрипт в вашей среде разработки (PyCharm, VS Code, IDLE).
2. Графики будут открываться в отдельных всплывающих окнах. Чтобы программа шла дальше, просто закрывайте окно с графиком.
3. Измените значение переменной MY_VARIANT на свой номер по списку.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from tqdm import tqdm

def if_a(p_value, a):
    if(p_value>a):
        return "H0 принимаем, H1 отвергаем"
    return "H1 принимаем, H0 отвергаем"

# ==========================================
# ЧАСТЬ 2. ИНИЦИАЛИЗАЦИЯ И ВЫБОР ВАРИАНТА
# ==========================================
# УКАЖИТЕ СВОЙ НОМЕР ВАРИАНТА НИЖЕ:

print(f"--- Активирован вариант №{12} ---")

# ==========================================
# ЗАДАНИЕ 1. Визуальное сравнение плотностей
# ==========================================
print("[Задание 1] Расчет теоретических плотностей...")
x = np.linspace(-4, 4, 1000)
pdf_norm = stats.norm.pdf(x, loc=0, scale=1)
pdf_t_20 = stats.t.pdf(x, df=20)
pdf_t_2  = stats.t.pdf(x, df=2)

plt.figure(figsize=(10, 5))
plt.plot(x, pdf_norm, label='Нормальное N(0, 1)', color='black', linewidth=2.5)
plt.plot(x, pdf_t_20, label='Стьюдента (df = 20)', color='red', linestyle='--', linewidth=2)
plt.plot(x, pdf_t_2, label='Стьюдента (df = 2)', color='blue', linestyle=':', alpha=0.6)
plt.title('Сравнение теоретических плотностей распределений')
plt.xlabel('Значение (X)')
plt.ylabel('Плотность вероятности')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
print("Окно с графиком открыто. Закройте его, чтобы продолжить расчеты...")
plt.show() 

max_diff = np.max(np.abs(pdf_norm - pdf_t_20))
print(f"Максимальная разница между N(0,1) и t(df=20): {max_diff:.5f}")

# ==========================================
# ЗАДАНИЕ 2. Парадокс больших выборок
# ==========================================
print("[Задание 2] Запуск тестов на нормальность для выборок разного объема...")
df_student = 20

samples = {
    'sample_30': stats.t.rvs(df=df_student, size=30),
    'sample_300': stats.t.rvs(df=df_student, size=300),
    'sample_5000': stats.t.rvs(df=df_student, size=5000)
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
results = []

for i, (name, data) in enumerate(samples.items()):
    N = len(data)
    stat_sw, p_sw = stats.shapiro(data)
    stat_dag, p_dag = stats.normaltest(data)
    
    results.append({
        'Выборка': name,
        'Размер (N)': N,
        'p-value (Шапиро-Уилок)': {round(p_sw, 5)},
        'p-value (Д\'Агостино)': {round(p_dag, 5)}
    })
    
    sns.histplot(data, kde=True, stat="density", color="lightgreen", ax=axes[i], alpha=0.6)
    x_plot = np.linspace(data.min(), data.max(), 100)
    axes[i].plot(x_plot, stats.norm.pdf(x_plot, loc=data.mean(), scale=data.std()), 
                 color='red', linestyle='--', linewidth=2, label='Теор. нормальное')
    axes[i].set_title(f'{name} (N = {N})\np-val SW: {p_sw:.4e} => {if_a(p_sw, 0.05)} \np-val Dag: {p_dag:.4e} => {if_a(p_dag, 0.05)}')
    axes[i].legend()

plt.tight_layout()
print("Окно с сеткой графиков открыто. Закройте его...")
plt.show()

df_res = pd.DataFrame(results)
print(df_res.to_string(index=False))

# ==========================================
# ЗАДАНИЕ 3. Мощность критерия (Монте-Карло)
# ==========================================


#Задача: Эмпирически оценить мощность критерия Шапиро–Уилка для объемов выборок 𝑛=30, 𝑛=300 и 𝑛=5000 при генерации данных из распределения Стьюдента (𝑑𝑓=20).
N=10000

count_30 = 0
x = stats.t.rvs(df=df_student, size=(N,30))
for j in range(0,N):
    stat_sw, p_sw = stats.shapiro(x[j])
    if p_sw<=0.05:
        count_30+=1

print(f'count_30 = {count_30}\ndensity_30 = {count_30/10000}')

count_300 = 0
x = stats.t.rvs(df=df_student, size=(N,300))
for j in range(0,N):
    stat_sw, p_sw = stats.shapiro(x[j])
    if p_sw<=0.05:
        count_300+=1

print(f'count_300 = {count_300}\ndensity_300 = {count_300/10000}')

count_5000 = 0
x = stats.t.rvs(df=df_student, size=(N,5000))
for j in range(0,N):
    stat_sw, p_sw = stats.shapiro(x[j])
    if p_sw<=0.05:
        count_5000+=1

print(f'count_5000 = {count_5000}\ndensity_5000 = {count_5000/10000}')

#Инструкция:
    #Допишите код симуляции Монте-Карло для каждого объема выборки.
    #Проведите 10000 симуляций для каждого 𝑛, фиксируя долю случаев, когда полученный 𝑝-value≤0.05.

# ==========================================
# ЗАДАНИЕ 4. Критерий согласия Хи-квадрат
# ==========================================

'''
Задача: Проверить гипотезу о нормальности выборки sample_5000 с помощью критерия Хи-квадрат Пирсона, 
самостоятельно построив интервальный ряд и объединив малочисленные интервалы (E𝑖<5).


Инструкция:
    Разбейте выборку на интервалы.
    Рассчитайте наблюдаемые частоты (𝑂𝑖) и теоретические частоты (E𝑖) через stats.norm.cdf.
    Реализуйте алгоритм слияния интервалов, где 𝐸𝑖<5.
    Запустите stats.chisquare с аргументом ddof=2.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

n = 5000
x = stats.t.rvs(df=df_student, size=n)

counts, edges = np.histogram(x, bins=15)

F = stats.norm.cdf(edges, loc=0, scale=1)
expected = n * np.diff(F)

def merge_small_bins(observed, expected, edges, min_count=5):
    observed = list(observed)
    expected = list(expected)
    edges = list(edges)

    i = 0

    # Объединяем интервалы слева направо
    while i < len(expected) - 1:
        if expected[i] < min_count:
            observed[i + 1] += observed[i]
            expected[i + 1] += expected[i]

            observed.pop(i)
            expected.pop(i)
            edges.pop(i + 1)
        else:
            i += 1

    # Проверяем последний интервал
    while len(expected) > 1 and expected[-1] < min_count:
        observed[-2] += observed[-1]
        expected[-2] += expected[-1]

        observed.pop()
        expected.pop()
        edges.pop(-2)

    return (
        np.array(observed),
        np.array(expected),
        np.array(edges)
    )

new_counts, new_expected, new_edges = merge_small_bins(
    counts, expected, edges, min_count=5
)

print("Интервальный ряд после объединения:")
print("-" * 65)

for i in range(len(new_counts)):
    print(
        f"[{new_edges[i]:.4f}; {new_edges[i+1]:.4f}] "
        f"O_i = {new_counts[i]:4d} "
        f"E_i = {new_expected[i]:.2f}"
    )

print("-" * 65)

stat_chi2, p_chi2 = stats.chisquare(
    f_obs=new_counts,
    f_exp=new_expected,
    ddof=2
)

print(f"\nСтатистика Хи-квадрат: {stat_chi2:.4f}")
print(f"p-value: {p_chi2:.6f}")

alpha = 0.05

if p_chi2 < alpha:
    print("Гипотеза о нормальности отвергается.")
else:
    print("Нет оснований отвергать гипотезу о нормальности.")

plt.figure(figsize=(10, 6))

plt.hist(
    x,
    bins=new_edges,
    density=True,
    alpha=0.6,
    edgecolor="black",
    label="Выборка"
)

arr = np.linspace(min(x), max(x), 1000)

plt.plot(
    arr,
    stats.norm.pdf(arr, loc=0, scale=1),
    color="red",
    label="Нормальное распределение"
)

plt.xlabel("Значение")
plt.ylabel("Плотность")
plt.title("Проверка нормальности выборки")
plt.legend()
plt.grid()
plt.show()
 
'''      
КОНТРОЛЬНЫЕ ВОПРОСЫ ДЛЯ ОТЧЕТА:
1. Какое максимальное расхождение плотностей получилось между N(0,1) и t(df=20)? Сильно ли меняется график при df=2?
2. Изменилась ли внутренняя природа данных при росте n с 30 до 5000 в Задании 2? Почему тесты выдали разный ответ?
3. Какую ошибку (I или II рода) совершает тест Шапиро-Уилка на малых выборках?
4. Инструкция для аналитика: База клиентов содержит 10 000 строк. График распределения — идеальный симметричный «колокол», но тест Шапиро-Уилка выдает p-value = 0.00001. Можно ли по этой выборке из 10000 наблюдений построить доверительные интервалы для математического ожидания или дисперсии?
5. Зачем в Хи-квадрат тесте объединяют малочисленные интервалы и почему ddof=2?
Как влияет на мощность критерия Хи-квадрат наличие малочисленных интервалов и неверное задание числа степеней свободы статистики критерия?
'''