import scipy.stats as st
import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


n=5
N=950
gamma = 0,965
a = 3
sigma = 2

x = st.norm.rvs(loc = a, scale = sigma, size = (N,n)) # генерация выборки

row_means = np.mean(x, axis=1)

plt.hist(row_means, bins='sturges', density=1, cumulative=0)

xmin = min(row_means)
xmax = max(row_means)

arr = np.linspace(xmin, xmax, N)

f = st.norm.pdf(arr, loc = a, scale = sigma/np.sqrt(n))
plt.plot(arr, f)
plt.show()
plt.clf()

b = sns.boxplot(row_means)
plt.show()
plt.clf()

mean_mean_x = np.mean(row_means)
var_mean_x = np.var(row_means)
s_mean_x = np.std(row_means, ddof = 1)
median_mean_x = np.median(row_means)
skew_mean_x = st.skew(row_means)
kurtosis = st.kurtosis(row_means)

count_means  = 0
for i in row_means:
    if i >a:
        count_means+=1

print(f"{(count_means/N)*100}% - процентов наблюдаемых значений случайной величины Xcр оказались больше {a}")

var_x = np.var(x, axis = 1)
plt.hist(var_x, bins='sturges', density=0, cumulative=0)


sigma_qwer = sigma**2
count_vars  = 0
for i in var_x:
    if i >sigma_qwer:
        count_vars+=1

print(f"{(count_vars/N)*100}% - процентов наблюдаемых Dв значений случайной величины оказались больше {sigma**2}")

y = var_x *n/sigma**2
print(len(y))

ymin = min(y)
ymax = max(y)

arr_y = np.linspace(ymin, ymax, len(y))

f_y = st.chi2.pdf(arr_y, n-1)
plt.hist(y, bins='sturges', density=1, cumulative=0)
plt.plot(arr_y, f_y)
plt.show()
plt.clf()

b_y = sns.boxplot(y)
plt.show()
plt.clf()


teor_mean_y = n-1
print(f"Теоритическое МО у {teor_mean_y}")
teor_var_y = 2*(n-1)
print(f"Теоритическое дисперсия у {teor_var_y}")
teor_med_y = (n-1)-2/3
print(f"Теоритическое медиана у {teor_med_y}")
teor_skew_y = np.sqrt(8/(n-1))
print(f"Теоритическое коэффициент асимметрии у {teor_skew_y }")
teor_kurtosis_y = 12/(n-1)
print(f"Теоритическое коэффициент эксцесса у {teor_med_y}")


mean_y = np.mean(y)
print(f"Среднее у {mean_y}")
var_y = np.var(y, ddof=1)
print(f"Дисперсия у {var_y}")
median_y = np.median(y)
print(f"Медиана у {median_y}")
skew_y = st.skew(y)
print(f"Коэффициент ассиметрии у {skew_y}")
kurtosis_y = st.kurtosis(y)
print(f"Коэффициент эксцесса у {kurtosis_y}")
print()

#13 пункт
x_2 = st.norm.rvs(loc = a, scale = sigma, size = n)
gamma = 0.965
t = st.norm.ppf(gamma, loc = 0, scale = 1)
print(t)
delta = t*sigma/np.sqrt(n)
print("Программый метод для нахождения интервальной оценки")
left = np.mean(x_2) - delta
right = np.mean(x_2) + delta
print(f"({left}<a<{right})")
print(f"delta = {delta}")

print()

res = scipy.stats.norm.interval((1+gamma)/2, loc=np.mean(x_2), scale=sigma/np.sqrt(n))
print(res)
