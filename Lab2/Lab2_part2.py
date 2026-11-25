import scipy.stats as st
import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


n=5
N=950
gamma = 0.965
a = 3
SIGMa = 2

def interval(x, gamma, size):
    t = st.t.ppf((1 + gamma) / 2, df=size - 1)
    s = np.std(x, ddof = 1)
    delta = t*s/np.sqrt(size)
    left = np.mean(x) - delta
    right = np.mean(x) + delta

    return left, right


x = st.norm.rvs(loc = a, scale = SIGMa, size = n)
print(np.std(x, ddof = 1))

left, right = interval(x,gamma,n)

print(f"({left}<a<{right})")
print()

s = np.std(x, ddof = 1)

res = scipy.stats.t.interval(gamma, df=n - 1, loc=np.mean(x), scale=s/np.sqrt(n))
print(res)

gamma_arr = [0.91, 0.93, 0.95, 0.97, 0.99]
len = []
for g in gamma_arr:
    inter = scipy.stats.t.interval(g, df=n - 1, loc=np.mean(x), scale=s/np.sqrt(n))
    len.append(inter[1]-inter[0])

plt.title('Зависимость длины доверительного интервала от надежности')
plt.plot(gamma_arr, len)
plt.show()


n_arr = [5, 10, 15, 20, 25]
len = []
for size in n_arr:
    x_i = x = st.norm.rvs(loc = a, scale = SIGMa, size = size)
    s = np.std(x_i, ddof = 1)
    inter = scipy.stats.t.interval(gamma, df=size - 1, loc=np.mean(x_i), scale=s/np.sqrt(size))
    len.append(inter[1]-inter[0])

plt.title('Зависимость длины доверительного интервала от объуема выборки')
plt.plot(n_arr, len)
plt.show()

x_2 = st.norm.rvs(loc = a, scale = SIGMa, size = (N,n))

lefts, rights = scipy.stats.t.interval(
                                        gamma, 
                                        df = n-1, 
                                        loc = np.mean(x_2, axis=1), 
                                        scale = np.std(x_2, ddof = 1, axis=1)/np.sqrt(n))

count = 0
for i in range(0, N):
    if(lefts[i]<a and rights[i]>a):
        count+=1

print(count/N)

y = st.norm.rvs(loc = a, scale = SIGMa, size = (N,n))

s_y = np.var(y, ddof = 1, axis=1)

plt.hist(s_y, bins='sturges', density=1, cumulative=0)
s_y_min = min(s_y)
s_y_max = max(s_y)

arr = np.linspace(s_y_min, s_y_max, N)

f = st.chi2.pdf(arr, df=n-1, scale=(SIGMa**2) / (n - 1))
plt.plot(arr, f)
plt.show()
plt.clf()

b = sns.boxplot(s_y)
plt.show()
plt.clf()

count_s_y = 0
for i in range(0,N):
    if(s_y[i]<SIGMa**2):
        count_s_y+=1

print(count_s_y/N)

z = st.norm.rvs(loc = a, scale = SIGMa, size = n)
z_left = scipy.stats.chi2.ppf((1-gamma)/2, df = n-1)
z_right = scipy.stats.chi2.ppf((1+gamma)/2, df = n-1)

ss = np.var(z, ddof=1)
left = (n-1)*ss/z_right
right = (n-1)*ss/z_left
print(f"({left}<SIGMa^2<{right})")