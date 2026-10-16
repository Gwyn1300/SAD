#вариант 12 (а = 3, сигма = 4) n = 105
# Правило для расчета числа интервалов группировки = Фридмана Диакониса
# Закон распределения случайной величины Y = t -распределение Стьюдента с числом степеней свободы k=4
# k = 5
# epsilon = 0.025

import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def mean(x):# среднее
    mean_value = 0 
    for i in x:
        mean_value+=i

    mean_value/=len(x)
    return mean_value

def var(x):# дисперсия
    mean_value = mean(x)
    dispersion = 0 
    for i in x:
        dispersion+=(i-mean_value)**2
    dispersion/=len(x)
    return dispersion

def s_qrt(x): # исправленная дисперсия
    return (len(x)*var(x))/(len(x)-1)

def skew(x): # коэф. асимметрии
    a = 0
    mean_value = mean(x)
    for i in x:
        a+=(i-mean_value)**3
    a/=len(x)
    a/=np.sqrt(var(x))**3 
    return a

def kurtosis(x):# эксцесс
    e=0
    mean_value = mean(x)
    for i in x:
        e+=(i-mean_value)**4
    e/=len(x)
    e/=np.sqrt(var(x))**4 
    return e-3

def median(x):
    n = len(x)
    x_sort = sorted(x)

    if n%2==1:
        return x_sort[(n+1)//2] # медиана
    else:
        return (x_sort[(n+1)//2]+x_sort[((n+1)//2)+1])/2
    
def mode(x):# мода
    if isinstance(x, np.ndarray):
        arr = sorted(x.tolist())
    else:
        arr = sorted(list(x))
        
    if not arr:
        return []
    
    max_count = 0
    modes = [] 

    current_element = arr[0]
    current_count = 0

    for item in arr + [None]:
        if item == current_element:
            current_count += 1
        else:
            if current_count > max_count:
                max_count = current_count
                modes = [current_element]

            current_element = item
            current_count = 1
    return modes



a = 3
sigma = 4
n = 105

print('='*60)
print("Лабораторная работа №1. Вариант 12.")
print('='*60)


x = st.norm.rvs(loc = a, scale = sigma, size = n) # генерация выборки
plt.title("Случайная величина X(a = 3, sigma = 4, n = 105)")
plt.plot(x)
plt.savefig("./images/my_plot.png", dpi=300, bbox_inches='tight')
plt.show()
print()


abs_freq = np.histogram(x, bins='fd', density=0) # интервальный ряд абс. частот
print("Группировка данных в интервалы:")
print(abs_freq[1])
print("Интервальный ряд абсолютных частот:")
print(abs_freq[0])
print()

sum_abs_freq = 0
for i in abs_freq[0]:
    sum_abs_freq+=i
print(f"Сумма абсолютных частот: {sum_abs_freq}")
print()


relative_freq = ( # интервальный ряд относ. частот
    abs_freq[0] / n,
    abs_freq[1]
)

print("Группировка данных в интервалы:")
print(relative_freq[1])
print("Интервальный ряд абсолютных частот:")
print(relative_freq[0])
print()

sum_relative_freq = 0
for i in relative_freq[0]:
    sum_relative_freq+=i
print(f"Сумма относительных частот: {sum_relative_freq}")
print()
plt.clf()


# =========================================
#2
#==========================================

for i in range(2, 11):
    plt.hist(x, bins=i, density=1, cumulative=0) # относительные частоты
    plt.savefig(f"./images/hist_relative_freq_bins_{i}.png", dpi=300, bbox_inches='tight')
    plt.clf()

for i in range(15, 26):
    plt.hist(x, bins=i, density=1, cumulative=0)  # относительные частоты
    plt.savefig(f"./images/hist_relative_freq_bins_{i}.png", dpi=300, bbox_inches='tight')
    plt.clf()

plt.hist(x, bins='fd', density=0, cumulative=0)  # абсолютные частоты
plt.savefig("./images/hist_abs_freq_bins_fd.png", dpi=300, bbox_inches='tight')

plt.hist(x, bins='fd', density=1, cumulative=0)  # относ. частоты

xmin = min(x)
xmax = max(x)

arr = np.linspace(xmin, xmax, n)
f = st.norm.pdf(arr, loc = a, scale = sigma)
plt.plot(arr, f)
plt.title("asdsadasdasd")
plt.show()
plt.clf()

plt.hist(x, bins='fd', density=1, cumulative=1)
F = st.norm.cdf(arr, loc = a, scale = sigma)
plt.plot(arr, F)
#plt.show()
plt.clf()


b = sns.boxplot(x)
#plt.show()


#=============================
#3
#=============================





s = np.sqrt(s_qrt(x)) # исправленное стандартное отклонение

sig = np.sqrt(var(x)) # стандартное отклонение




print(f"Среднее: {mean(x)}")
print(f"Среднее: {np.mean(x)}")

print(f"Дисперсия: {var(x)}")
print(f"Дисперсия: {np.var(x)}")

print(f"Исправленная дисперсия: {s_qrt(x)}")
print(f"Исправленная дисперсия: {np.var(x, ddof = 1)}")

print(f"Исправленное стандартное отклонение: {np.sqrt(s_qrt(x))}")
print(f"Исправленное стандартное отклонение: {np.std(x, ddof = 1)}")

print(f"Стандартное отклонение: {np.sqrt(var(x))}")
print(f"Стандартное отклонение: {np.std(x)}")

print(f"Коэф. асимметрии: {skew(x)}")
print(f"Коэф. асимметрии: {st.skew(x, bias = False)}")

print(f"Эксцесс: {kurtosis(x)}")
print(f"Эксцесс: {st.kurtosis(x, bias = False)}")

print(f"Медиана: {median(x)}")
print(f"Медиана: {np.median(x)}")

print(f"Мода: {mode(x)}")
print(f"Мода: {st.mode(x)}")

print('='*50)

n_60 = n*60

v = st.norm.rvs(loc = a, scale = sigma, size = n_60)

print(f"Среднее: {mean(v)}")
print(f"Среднее: {np.mean(v)}")

print(f"Дисперсия: {var(v)}")
print(f"Дисперсия: {np.var(v)}")

print(f"Исправленная дисперсия: {s_qrt(v)}")
print(f"Исправленная дисперсия: {np.var(v, ddof = 1)}")

print(f"Исправленное стандартное отклонение: {np.sqrt(s_qrt(v))}")
print(f"Исправленное стандартное отклонение: {np.std(v, ddof = 1)}")

print(f"Стандартное отклонение: {np.sqrt(var(v))}")
print(f"Стандартное отклонение: {np.std(v)}")

print(f"Коэф. асимметрии: {skew(v)}")
print(f"Коэф. асимметрии: {st.skew(v, bias = False)}")

print(f"Эксцесс: {kurtosis(v)}")
print(f"Эксцесс: {st.kurtosis(v, bias = False)}")

print(f"Медиана: {median(v)}")
print(f"Медиана: {np.median(v)}")

print(f"Мода: {mode(v)}")
print(f"Мода: {st.mode(v)}")

print('='*80)

print("Часть 2")
print('='*80)

k = 4
y = st.t.rvs(k, size = n)

ymin = min(y)
ymax = max(y)
arr = np.linspace(ymin, ymax, 100)

plt.hist(y, bins='fd', density=1, cumulative=0)
f = st.t._pdf(arr, 4)
plt.plot(arr, f)
plt.show()
plt.clf()

plt.hist(y, bins='fd', density=1, cumulative=1)
F = st.t.cdf(arr, 4)
plt.plot(arr, F)
plt.show()

b = sns.boxplot(y)
plt.show()