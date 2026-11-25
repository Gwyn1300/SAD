import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from tqdm import tqdm

x = stats.t.rvs(df = 20, size = 25)
res = stats.shapiro(x)
print(res)