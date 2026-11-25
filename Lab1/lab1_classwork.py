import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import seaborn as sns
import math

a = 3
sigma = 2
n = 1000
x = st.norm.rvs(loc = a, scale = sigma, size = n)
#plt.plot(x)
#plt.show()
rez = np.histogram(x, bins = 5)
#print(rez)

'''
data = [0,0,1,1,2,2,4,4,5,5]
#plt.hist(data, bins=3, density=0, cumulative=1)# кумулята абсолютных частот (график накопленной частоты)
#plt.show()
plt.title("Кумулята относительных частот")
plt.hist(data, bins=3, density=1, cumulative= 1)
plt.show()
Z = np.histogram(data, bins = 3)
print(Z[0])
print(Z[1])

Z = np.histogram(data, bins = 3, density = 1)
print(Z[0])
print(Z[1])

shag = Z[1][1] - Z[1][0]
print(shag)

Y = np.sum(Z[0])*shag
print (Y)

plt.hist(data, bins = 3, density = 0)
plt.show()

plt.title("name")
plt.hist(x, bins = 'fd', density=1)
plt.show()

#нужно последовательный массив точек

xmin = min(x)
xmax = max(x)

arr = np.linspace(xmin, xmax, 100)
print(arr)'''
'''график должен огибать'''
'''y = st.norm.pdf(arr, loc = a, scale = sigma)
plt.plot(arr, y)
plt.hist(x, bins='scott', density=0)
plt.show()'''

#plt.hist(x, bins='fd', density=0, cumulative=1)# кумулята абсолютных частот (график накопленной частоты)
#plt.show()
#plt.title("Кумулята относительных частот")
#plt.hist(data, bins=3, density=1, cumulative= 1)
#plt.show()

#F(x) = P(X<x) = 1/2 + Ф(x-a/sigma)

'''xmin = min(x)
xmax = max(x)

arr = np.linspace(xmin, xmax, 100)
F = st.norm.cdf(arr, loc=a, scale = sigma)
plt.plot(arr, F)
#plt.show()

b = sns.boxplot(x)
plt.show()'''

#kurtosis фишер = 0 или 1

#сигма в = sqrt(Dв), S = sqrt(S^2)

sum = 0
for i in x:
    sum+= i
a = sum/n

print(a)

sum = 0
for i in x:
    sum+=(i-a)**2
var = sum/n
print(var)

s_qrt = n*var/(n-1)
print(s_qrt)
print(np.var(x), ddof = 1)

sum = 0
for i in x:
    sum+=(i-a)**3
A = sum/n
A = A/(math.sqrt(var))**3

print(A)

#ddof надо запомнить

sum = 0
for i in x:
    sum+=(i-a)**4
E = sum/n
E = (E/(math.sqrt(var))**4) - 3
print(E)
