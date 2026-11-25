import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from scipy.stats import norm
import os

print(os.getcwd())
path="D:\\study\\SAD\\SAD\\Lab4"
os.chdir(path)
os.getcwd()

doc = pd.read_excel("CHISLO_DOCTORS_2.xlsx", sheet_name='FO_2')
print(doc.head())

years = [2005, 2010, 2015, 2019, 2020, 2021]

'''
for i in years:
   plt.hist(doc[i], bins=3)
   plt.title('Диаграммы размаха (боксплоты) по годам')
   plt.xlabel('Год')
   plt.ylabel('Показатель X')
   plt.show()
'''
doc2 = pd.read_excel("CHISLO_DOCTORS (1).xlsx", sheet_name='FO_3')

res = stats.bartlett(doc[2005],doc2[2005])
print(f"res = {res}")
# пример 4 union
np.random.seed(14)
X=scipy.stats.norm.rvs(loc=40, scale = 4, size = 16)
Y=scipy.stats.norm.rvs(loc=50, scale = 3.5, size = 10)
#H0: M(X) = M(Y)
#H1: M(X)!=M(Y)

stat_sw, p_sw = stats.shapiro(X)
print(p_sw)

stat_sw, p_sw = stats.shapiro(Y)
print(p_sw)

(t_stat, p_val) = stats.ttest_ind(X,Y)
print(f"pval = {t_stat}")
print(f"pval = {p_val}")

n_x = 16
n_y = 10
X_mean = np.mean(X)
Y_mean = np.mean(Y)
S_x = np.var(X, ddof=1)
S_y = np.var(Y, ddof=1)

print(f"S_x = {S_x}")
print(f"S_Y = {S_y}")

T = (X_mean-Y_mean)/(np.sqrt((n_x-1)*S_x+(n_y-1)*S_y))*np.sqrt((n_x*n_y*(n_x+n_y-2))/(n_x+n_y))
print(T)

#H0: M(X) = M(Y)
#H2: M(X)<M(Y)(тк X_mean < Y_mean)

(t_stat, p_val) = stats.ttest_ind(X,Y, alternative="less", equal_var=False)
print(f"pval2 = {t_stat}")
print(f"pval2 = {p_val}")
#приняли H2
'''
(t_stat, p_val) = stats.ttest_ind(X,Y, alternative="greater")
print(f"pval = {t_stat}")
print(f"pval = {p_val}")
'''