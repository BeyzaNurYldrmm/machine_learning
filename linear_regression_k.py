# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:19:57 2024

@author: 06bey
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)#verisetindeki rakamların iki basamak göster

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv("datasets/advertising.csv")
print(df.shape)
X = df[["TV"]]
y = df[["sales"]]#bgmlı değ
z=df[["radio"]]
print(X)
print(y)
print(z)

reg_model=LinearRegression().fit(X, y)
reg_model.intercept_[0]# bias icin intercept deniliyor
reg_model.coef_[0][0]#tv nin katsayısı yani w1 agırlık=coef


########################################
#tahmin

#150 birimlik tv harcaması olsa ne kadar satış olur?
print("150 birimlik tv harcamasında satış mik.=",reg_model.intercept_[0]+reg_model.coef_[0][0]*150)

print(df.describe().T)

g=sns.regplot(x=X,y=y, scatter_kws={'color':'b','s':9},
              ci=False, color="r")
g.set_title(f"Model Denklemi: Sales= {round(reg_model.intercept_[0],2)}+TV*{round(reg_model.coef_[0][0],2)}")
g.set_ylabel("Satış sayısı")
g.set_xlabel("TV harcamaları")
plt.xlim(-10,310)
plt.ylim(bottom=0)
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score


####################################
# r^2 istatistik degerlendirme

y_pred=reg_model.predict(X)#bgmsız değ. soruyorum bgmlı değişkeni bilmiyor gibi
print(mean_squared_error(y, y_pred))#MSE->10.51
y.mean()#14.02
y.std()#5.22   yani mse değeri yüksek 
np.sqrt(mean_squared_error(y, y_pred))#rmse ->3.24
mean_absolute_error(y, y_pred)# MAE ->2.54
reg_model.score(X, y)#r kare % olarak kaçta kaçını acıklamış->%61
#tv deki satışı etkilemeye bakıp acıklama istatistiğini gördük



#######################################
# çoklu doğrusal regresyon diğer bgmsız değ. ile çalışma

X=df.drop('sales', axis=1)#sales sutununu atıp diğer tüm değişkenleri aldık

print(X)

########################################
# model

X_train,y_train,X_test,y_test=train_test_split(X,y, test_size=0.20, random_state=1)
X_train.shape








 

