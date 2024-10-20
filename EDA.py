
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score
#from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split




"""
df=pd.read_csv("datasets\\diabetes.csv")
print(df.head,"  \n Boyutu:",df.shape)
df["Outcome"].value_counts()#çıkış sutununda kaç sınıflara ait kaç etiket olduğunu saydırdık


#görsel olarak oranları görmek için
sns.countplot(x="Outcome", data=df)
plt.show()

#outcome sutununun sınıf sayılarına 100 de oranları için
100*df["Outcome"].value_counts()/len(df)

df.describe().T

#histogram olarak görselleştirme
df["Outcome"].hist(bins=20)
plt.xlabel("Outcome")
plt.show()

def hist_olustur(dataframe, sutun):
    dataframe[sutun].hist(bins=20)
    plt.xlabel(sutun)
    plt.show(block=True)#pes pese gösterirken birbirini ezmesin grafikler
    
#her sutunun histogram grafigini olusturduk    
for col in df.columns:
    hist_olustur(df,col)
    
#sınıflandırma yapılacak sutun olmaması için
for col in df.columns:
    if col not in "Outcome":
        hist_olustur(df,col)
        
        
"""
"""
#yukarıdaki for döngüsünün farklı kullanımı
#en baştaki col ne derseniz, liste oluşturmanın en basit yoludur. Koşulu sağlayan sutun adlarını listeye ekliyoruz.
cols= [col for col in df.columns if "Outcome" not in col]

for col in cols:
    hist_olustur(df, col)

#3.farklı gösterim
for col in [col for col in df.columns if "Outcome" not in col]:
    hist_olustur(df, col)
    
"""
"""

#sınıflara göre bgmsız. değ. oranı
df.groupby("Outcome").agg({"Pregnancies":"mean"})
df.groupby("Outcome").agg({"Age":"max"})

def oran_hes(df,sutun):
    for col in [col for col in df.columns if sutun not in col]:
         print(df.groupby(sutun).agg({col :"mean"}))
        
oran_hes(df,"Outcome")
"""


#####################################
# Veri Önişleme
#####################################
df=pd.read_csv("datasets\\diabetes.csv")
df.isnull().sum()
print(df.describe().T)

def check_outlier(df, col):
    lb, ub= outlier_threshold(df,col)
    if df[(df[col]>ub) | (df[col]< lb)].any(axis=None):
        return True
    else:
        return False
  
def outlier_threshold(df,col,q1=0.05,q3=0.95):
    q_1=df[col].quantile(q1)
    q_3=df[col].quantile(q3) 
    interquantile_range=q_3-q_1
    ub=q_3+1.5*interquantile_range
    lb=q_1-1.5*interquantile_range
    return lb,ub

def replace_threshold(df, col):
    lb, ub = outlier_threshold(df, col)   
    df.loc[df[col] < lb, col] = lb # Alt sınırın(ub) altındaki aykırı değerleri alt sınıra eşitledik
    df.loc[df[col] > ub, col] = ub # Üst sınırın(lb) üstündeki aykırı değerleri üst sınıra eşitle
    print(f"{col} sütunundaki aykırı değerler güncellendi.")
    return df
    


#aykırı değer kontrol
for col in df.columns:
    print(col, check_outlier(df, col))#aykırı değer varmı kontrolunu yaptırıyoruz.
#aykırı değer threshold değeri güncelleme
replace_threshold(df,"Insulin")   

#standartlaşturma
for col in [col for col in df.columns if "Outcome" not in col]:
    df[col]=RobustScaler().fit_transform(df[[col]])
print(df.head())




###########################################
#model

y=df["Outcome"]
X=df.drop(["Outcome"], axis=1)#axis?0 satır, 1 sutunu temsil eder
log_res_ml=LogisticRegression().fit(X, y)
log_res_ml.intercept_
log_res_ml.coef_

y_pred=log_res_ml.predict(X)
print(y_pred[0:10])
print(y[0:10])

############################################
#basarı metrik

def plot_cm(y,y_pred):
    acc=round(accuracy_score(y,y_pred),2)
    cm=confusion_matrix(y,y_pred)
    sns.heatmap(cm,annot=True,fmt=".0f")
    plt.xlabel("y_pred")
    plt.ylabel("y")
    plt.title("Accuracy Score: {0}".format(acc),size=10)
    plt.show()

plot_cm(y, y_pred)
#recall f1 score ve presicier metriklerini otomatik
print(classification_report(y,y_pred))

#ROC eğrisi yani farklı thresholdlarda
y_pred=log_res_ml.predict_proba(X)[:,1]
roc_auc_score(y,y_pred)



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20, random_state=17)
log_model=LogisticRegression().fit(X_train, y_train)
y_pred=log_model.predict(X_test)
y_prob=log_model.predict_proba(X_test)[:1]#sınıf olasılığını verir


#plot_roc_curve(log_model,X_test,y_test)
#plt.title("ROC Curve")
#plt.plot([0,1], [0,1], 'r--')
#plt.show()

#AUC
roc_auc_score(y_test,y_prob)


