import pandas as pd
import numpy as np

database = pd.read_csv('./datasets/datasets.csv')
print('database shape:',database.shape)
data_base = database.sample(frac=1).reset_index(drop=True)
y_all = data_base.iloc[:,1]
np.random.seed(20)
data_base = data_base.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
x_all = data_base.iloc[:,2:]
print('data processing completed')