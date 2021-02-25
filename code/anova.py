import pyper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

score_name = 'SCR'

df = pd.read_excel('../log/total/CA.xlsx')
#df = df[['A', 'B', score_name]]
df = df[['a1', 'a1.1', 'a2', 'a2.1']]

print(df)

r = pyper.R(use_pandas='True')

r.assign('data', df)

r("source('anovakun_485.txt', encoding='utf-8')")
result = r('anovakun(data,"sAB", 2, 2, eta=T, holm=T)')
print(result)
