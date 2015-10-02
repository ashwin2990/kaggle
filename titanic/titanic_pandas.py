import pandas as pd
import numpy as np
df=pd.read_csv('train.csv',header=0)
for i in range(1,4):
    print i, len(df[ (df['Sex'] == 'male') & (df['Pclass'] == i) ])
