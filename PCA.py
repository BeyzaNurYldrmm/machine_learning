import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

file_path=r"D:\makine_ogrenmesi\datasets\hitters.csv"
df=pd.read_csv(file_path,index_col=0)

num_col=[col for col in df.columns if df[col].dtypes !="O" and "Salary" not in col]#tipi object ve salary olmayanı geç


df[num_col].head()
df=df[num_col]
df.dropna(inplace=True)#eksiklikleri sil
df.shape


df=StandardScaler().fit_transform(df)
pca=PCA()
pca_fit=pca.fit_transform(df)

 
pca.explained_variance_ratio_#varyans oran bilgisi
print(np.cumsum(pca.explained_variance_ratio_))#1.değişkenin acıklayacagı varyans, 2.değişkenin acıklayacağı varyans,....[kümülatif toplam]

################################
#optimum bileşen sayısı bulma
pca=PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("bileşen sayısı")
plt.ylabel("kümülatif varyans oranı")
plt.show()


#################################
#PCA ile 3 değişkene indirme
pca= PCA(n_components=3)
pca_fit=pca.fit_transform(df)
pca.explained_variance_ratio_

#################################
#PCA ile 5 değişkene indirme
pca= PCA(n_components=5)
pca_fit=pca.fit_transform(df)
pca.explained_variance_ratio_




