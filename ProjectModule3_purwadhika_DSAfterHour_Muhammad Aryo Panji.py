#!/usr/bin/env python
# coding: utf-8

# # Telco Customer Churn Analytics: to know which customer will leave the telco service
# ### Created By : Muhammad Aryo Panji
# Perusahaan B-TELL ( assumpion for study case )

# ## Business Problem Understanding

# **Context**  
# Sebuah perusahaan yang bergerak di bidang Telco Service ingin mengetahui customer yang kan tetep menggunakan layanan telco atau meninggal layanan telco.Perusahaan ingin mengetahui customer mana yang benar-benar akan meninggalkan layanan telco perusahaan setelah mencoba menggunakan layanan telco karena membantu perusahaan untuk memetakan cutomer memiliki probability besar untuk meninggalkan laynanan perusahaan sehingga dapat memperkuat dari sisi perusahaan untuk meningkat value dari layanan. Informasi terkait yang menjadi varible dalam perhitungan yaitu dari customer yang sudah menggunakan laynan telco service.
# 
# Target :
# 
# 0 : Tidak mencari layanan baru dan teteap pada layanan telco service ini
# 
# 1 : Mencari layanan telco yang baru ( akan berpindah menggunakan ISP lain)

# **Problem Statement :**
# 
# Semakin banyak layanan telekomunikasi yang bermunculan akhir-akhir ini membuat perusahaan telco semakin bersaing secara ketat dengan menawarkan paket layanan internet yang murah, cepat, dan memiliki beragam fitur yang membuat di sisi customer nyaman menggunakan product telco. Dengan ketatnya persaingan harga dan layanan di perusahaan telekomunikasi membuat B-Tell melakukan project customer churn untuk mengetahui customer yang memiliki chance besar untuk meninggal product layanan dari B-Tell or tetap bertahan menggunakan produk layanan.
# 
# Customer yang meninggalkan layanan telco perusahaan akan menjadi "oppurtunity lose" dalam kegiatan bisnis kedepan. sehinggal membuat perusahaan memetakan antara customer yang tetep menggunakan layanan dan yang tidak, dan faktor faktor apa saja yang membuat pelanggan itu beralih menggunakan layanan telco yang baru.
# 
# 
# **Goals :**
# 
# Maka berdasarkan permasalahan tersebut, perusahaan ingin memiliki kemampuan untuk memprediksi kemungkinan seorang customer akan/ingin berpindah menggunakan product layanan perusahaan telekomunikasi yang lain, sehingga dapat memfokuskan kepada customer yang akan berpindah dengan membuat paket yang semakin dekat dengan kebutuhan customer agar customer tidak beralih menggunakan produk layanan sejenis di perusahaan telco lainya.
# 
# Dan juga, perusahaan ingin mengetahui apa/faktor/variabel apa yang membuat seorang customer mau berpindah atau tidak, sehingga mereka dapat membuat rencana yang lebih baik dalam mendekati customer potensial loss (customer yang ingin berpindah menggunakan produk dari perusahaan lain.) .
# 
# **Analytic Approach :**
# 
# Jadi yang akan kita lakukan adalah menganalisis data untuk menemukan pola yang membedakan customer yang mau beralih menggunakan produk layanan atau tidak.
# 
# Kemudian kita akan membangun model klasifikasi yang akan membantu perusahaan untuk dapat memprediksi probabilitas seorang customer akan beralih atau tidak.

# **Metric Evaluation**

# maka sebisa mungkin yang akan kita lakukan adalah membuat model yang dapat meningkatkan accuracy customer yang akan churn dan actual nya juga churn

# **Important Feature**

# 
# 
# *   Dependents: Yang 
# *   Tenure: 
# 
# 

# # **Import File And Package**
# 

# In[6]:


import pandas as pd


# In[7]:


pd


# In[8]:


import pandas as pd

#Lets import the libraries first
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import warnings
warnings.filterwarnings('ignore')


# In[9]:


get_ipython().system('pip install --upgrade seaborn')


# In[10]:


data_prep = pd.read_csv (r'C:\Users\telo\Downloads\data_telco_customer_churn.csv')
print (df)


# In[249]:


df = pd.read_csv (r'C:\Users\telo\Downloads\data_telco_customer_churn.csv')


# # **Explanatory Dataset**

# In[ ]:


plt.figure(figsize=(8,7), facecolor='lightyellow')
plt.pie(data['Churn'].value_counts(), autopct='%.2f%%', pctdistance = 1.25,startangle=45, textprops={'fontsize': 15}, 
colors=['indigo','darkorange'], shadow=True)
my_circle=plt.Circle( (0,0), 0.6, color='lightyellow')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('Churn Proportion', fontsize=17, fontweight='bold')
plt.legend(['No', 'Yes'], bbox_to_anchor=(1, 1), fontsize=12)
plt.show()


# data imbalance thats why we need to catch up for creating balacing data using several method

# In[258]:


for i in df.columns:
  print(i,df[i].unique())


# In[260]:


data = df
plt.figure(figsize=(8,7), facecolor='lightyellow')
plt.pie(data['Churn'].value_counts(), autopct='%.2f%%', pctdistance = 1.25,startangle=45, textprops={'fontsize': 15}, 
colors=['indigo','darkorange'], shadow=True)
my_circle=plt.Circle( (0,0), 0.6, color='lightyellow')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('Churn Proportion', fontsize=17, fontweight='bold')
plt.legend(['No', 'Yes'], bbox_to_anchor=(1, 1), fontsize=12)
plt.show()


# # **Variable Service Having Ambiguity**

# disini kita melihat ada data "No internet service" ni bisa disimpulkan untuk tidak menggunakan layanan jadi bisa di masukan ke dalam class "no" sebelum masuk ke tahap encoding
# 
# 
# 
# OnlineSecurity ['No' 'Yes' 'No internet service']
# 
# OnlineBackup ['No' 'Yes' 'No internet service']
# 
# DeviceProtection ['Yes' 'No internet service' 'No']
# 
# TechSupport ['Yes' 'No' 'No internet service']

# In[11]:


#preparing the figure size 
fig, axarr = plt.subplots(2, 2, figsize=(15, 15))

sns.countplot('OnlineBackup',data = data_prep, ax = axarr[0][0])
sns.countplot('OnlineSecurity',data = data_prep, ax = axarr[0][1])
sns.countplot('DeviceProtection',data = data_prep, ax = axarr[1][0])
sns.countplot('TechSupport',data = data_prep, ax = axarr[1][1])


# In[12]:


data_prep['DeviceProtection'] = data_prep['DeviceProtection'].replace(['No internet service'],'No')
data_prep['OnlineBackup'] = data_prep['OnlineBackup'].replace(['No internet service'],'No')
data_prep['OnlineSecurity'] = data_prep['OnlineSecurity'].replace(['No internet service'],'No')
data_prep['TechSupport'] = data_prep['TechSupport'].replace(['No internet service'],'No')


# In[17]:


numerical = ["MonthlyCharges", "tenure"]

fig, ax = plt.subplots(1, 2, figsize=(15, 8))
for variable, subplot in zip(numerical, ax.flatten()):
    sns.boxplot(x=data_prep["Churn"], y=data_prep[variable], ax=subplot, palette = "Set2").set_title(str(variable))


# kita bisa lihat bahwa cutomer yang melakukan churn itu biasanya berada di range 80-90 dollar mothlycharges nya
# dan tenure pendek dibawah 30 score nya

# In[18]:


a=sns.PairGrid(data_prep, y_vars=["tenure"], x_vars=["MonthlyCharges"], height=4.5, hue="Churn")
ax = a.map(plt.scatter)


# demograhics

# In[21]:


def stacked_plot(data_prep, features, target):
    fig, ax = plt.subplots(figsize = (6,4))
    df = (data_prep.groupby([features, target]).size()/data_prep.groupby(features)[target].count()).reset_index().pivot(columns=target, index=features, values=0)
    df.plot(kind='bar', stacked=True, ax = ax, color = ["blue", "orange"])
    ax.xaxis.set_tick_params(rotation=0)
    ax.set_xlabel(features)
    ax.set_ylabel('Churn Percentage')


# In[23]:


stacked_plot(data_prep, "OnlineSecurity", "Churn")
stacked_plot(data_prep, "OnlineSecurity", "Churn")
stacked_plot(data_prep, "OnlineSecurity", "Churn")
stacked_plot(data_prep, "Dependents", "Churn")
stacked_plot(data_prep, "DeviceProtection", "Churn")
stacked_plot(data_prep, "TechSupport", "Churn")
stacked_plot(data_prep, "Contract", "Churn")
stacked_plot(data_prep, "PaperlessBilling", "Churn")


# monthly charger dan tenure angka numerc yang kita harus pendistribusian data nya menggunakan package normaltestdari spicy stat untuk menguji apakah data kita terdistribusi normal atau tidak baru kami lanjutkan untuk mencari korelasi 

# In[ ]:





# In[24]:


from scipy.stats import normaltest

stat1, p1 = normaltest(data_prep['tenure'])
stat2, p2 = normaltest(data_prep['MonthlyCharges'])


if p1 > 0.05:
    s1 = 'It is normally distributed.'
else:
    s1 = 'It is not normally distributed'
    
if p2 > 0.05:
    s2 = 'It is normally distributed.'
else:
    s2 = 'It is not normally distributed.'

fig = plt.figure(figsize=(20,6))

plt.subplot(121)
plt.title("Stats:"+ str(stat1)+" ,  P value:" + str(p1)+ "  , " + s1)
sns.histplot(data_prep['tenure'],kde=True, color="red")
plt.axvline(data_prep['tenure'].mean(), color='k', linestyle='dashed', linewidth=2)


plt.subplot(122)
plt.title("Stats:"+ str(stat2)+" ,  P value:" + str(p2)+ "  , " + s2)
sns.histplot(data_prep['MonthlyCharges'],kde=True, color="green")
plt.axvline(data_prep['MonthlyCharges'].mean(), color='k', linestyle='dashed', linewidth=2)


plt.show()


# karena data kita tidak terdistribusi normal maka diperlukan method sperman untuk mencari korelasi data terhadapt target churn 

# In[25]:


sns.pairplot(data_prep[['tenure','MonthlyCharges']])
plt.show()


# **insight** tenure and Monthly Charges have weak correlation

# In[34]:


Mth = sns.kdeplot(data_prep.MonthlyCharges[(data_prep["Churn"] == 0) ],
                color="Red", shade = True)
Mth = sns.kdeplot(data_prep.MonthlyCharges[(data_prep["Churn"] == 1) ],
                ax =Mth, color="Blue", shade= True)
Mth.legend(["No Churn","Churn"],loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Monthly Charges')
Mth.set_title('Monthly charges by churn')


# **Insight:** Churn is high when Monthly Charges ar high

# In[35]:


Mth = sns.kdeplot(data_prep.tenure[(data_prep["Churn"] == 0) ],
                color="Red", shade = True)
Mth = sns.kdeplot(data_prep.tenure[(data_prep["Churn"] == 1) ],
                ax =Mth, color="Blue", shade= True)
Mth.legend(["No Churn","Churn"],loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('tenure')
Mth.set_title('tenure by churn')


# In[36]:


fig, ax = plt.subplots(figsize=(8, 6))

sns.histplot(data=data_prep, x='tenure', hue='Contract', stat='percent', multiple='dodge', ax=ax, binwidth=5, alpha=0.8)
sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1), ncol=3, title='', frameon=False)

sns.despine()  
plt.show()


# In[37]:


plt.figure(figsize=(20,8))
data_prep.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')


# ## **Derived Insight: **
# 
# **HIGH** Churn seen in case of  **Month to month contracts**, **No online security**, **No Tech support**, **First year of subscription** and **Fibre Optics Internet**
# 
# **LOW** Churn is seens in case of **Long term contracts**, **Subscriptions without internet service** and **The customers engaged for 5+ years**
# 
# Factors like **Gender**, **Availability of PhoneService** and **# of multiple lines** have alomost **NO** impact on Churn
# 
# This is also evident from the **Heatmap** below

# In[252]:


plt.figure(figsize=(7,7))
sns.heatmap(data_prep.corr(), cmap="YlGnBu")


# ## Missing data

# In[253]:


missing = pd.DataFrame((df.isnull().sum())*100/df.shape[0]).reset_index()
plt.figure(figsize=(16,5))
ax = sns.pointplot('index',0,data=missing)
plt.xticks(rotation =90,fontsize =7)
plt.title("Percentage of Missing values")
plt.ylabel("PERCENTAGE")
plt.show()


# ## **Checking Outliers**

# In[250]:


num_features= ['tenure', 'MonthlyCharges']
df_num = df[num_features]
df_num.describe()

Q1 = df_num.quantile(0.25)
Q3 = df_num.quantile(0.75)
IQR = Q3 - Q1
IQR
((df_num < (Q1 - 1.5 * IQR)) | (df_num > (Q3 + 1.5 * IQR))).any()


# ## **Binning Tenure**

# berdasarkan bar plot ini kita bisa membagi tenure membagi 6 dengan memberikan tingkatan dan ini harus hati hati dalam melakukan bining dan mengubah menjadi agar tidak terlalu complex data nya
# 
# awalnya saya menggunakan bining untuk tenure nya untuk menjaga complexity namun bukanya model malah bagus membuat model jadi ambigu jadi saya memutuskan tidak menggunakan bining

# In[ ]:


#to reduce data complexity make 5 groups
#def tenure(t):
 #   if t<=12:
  #      return 1
   ##    return 2
   # elif t>24 and t<=36:
    #    return 3
    #elif t>36 and t<=48:
     #   return 4
    #elif t>48 and t<=60:
     #   return 5
    #else:
     #   return 6

#data_prep["tenure_group"]=data_prep["tenure"].apply(lambda x: tenure(x))


# # **Encode**

# kita akan meilhat ada berapa banyak feature yang dapat kami encode:

# In[30]:


for i in data_prep.columns:
  print(i,data_prep[i].unique())


# In[31]:


data_prep['DeviceProtection'] = data_prep['DeviceProtection'].replace(['No internet service'],'No')
data_prep['OnlineBackup'] = data_prep['OnlineBackup'].replace(['No internet service'],'No')
data_prep['OnlineSecurity'] = data_prep['OnlineSecurity'].replace(['No internet service'],'No')
data_prep['TechSupport'] = data_prep['TechSupport'].replace(['No internet service'],'No')


# skrng kita bisa melakukan fiture encode di column yang memiliki 2 feature manjadi numerical value

# In[ ]:


#from sklearn.preprocessing import OneHotEncoder
#ohe = OneHotEncoder(categories=[['Yes', 'No']] ,handle_unknown='ignore', sparse = False)
#cols = ['Dependents', 'PaperlessBilling', 'Churn']


# In[ ]:


#for i in cols:
 ## y=np.array(data_prep[i]).reshape(-1,1)
  #ohe.fit(y)
  #data_prep[i] = ohe.transform(y)


# ## **Encode feature menggunakan label encoder**

# In[32]:


from sklearn.preprocessing import LabelEncoder
lenc = LabelEncoder()
cols = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'Contract','InternetService','Dependents','PaperlessBilling', 'Churn']

for i in cols:
  lenc.fit(data_prep[i])
  data_prep[i] = lenc.transform(data_prep[i])


# In[33]:


## visualization of the features 
data_prep.hist(figsize=(10,20), xrot=-45)
plt.show()


# # **Handling Imbalance Data**

# In[38]:


data_prep['Churn'].value_counts()/data_prep.shape[0]*100


# In[39]:


data_prep['Churn'].value_counts()


# In[279]:


data_prep


# data yang kita punya imblance semua data nya sehingga kita menggunakan method smote untuk mem balance nya class nya

# # **Split Data**

# split data awal : 
# data_x : sebelum di tuning
# data_y : data target sebelum tuning
# 
# variable yang akan digunakan untuk masuk ke tahap ensambling : 
# 
# 
# data_x = data spiliting ebelum minmax scaller
# 
# 
# x_train = sudah di min max scaller di montly charger (sudah di scalling)
# 
# 
# x_resample = menggunakan teknik smote untuk imblance dataset
# 
# XGBClassifier = feature
# SmoteMerge = (  )
# X_xgb
# Y_xgb

# In[40]:


data_x = data_prep[[
                    'Dependents', 'tenure', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'Contract', 'PaperlessBilling', 
                    'MonthlyCharges']].copy()
                    
data_y = data_prep['Churn']


# Next, split the training data and testing data. We’ll try training:testing = 70:30

# In[318]:


from sklearn.model_selection import train_test_split
from collections import Counter

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=5)
Counter(y_train)


# In[319]:


x_train


# In[320]:


x_test


# ## Scalling Data using min_max() scaller

# Scaling Data yang memiliki range yang besar tenure dengan monthly charge

# In[44]:


x_train[[ 'tenure','MonthlyCharges']]


# In[322]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaled_train=np.array(x_train[[ 'tenure']]).reshape(-1,1)

scaler = scaler.fit(scaled_train)

x_train[[ 'tenure']] = scaler.transform(scaled_train)

scaled_test=np.array(x_test[['tenure']]).reshape(-1,1)

x_test[['tenure']] = scaler.transform(scaled_test)


# In[323]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaled_train=np.array(x_train[[ 'MonthlyCharges']]).reshape(-1,1)

scaler = scaler.fit(scaled_train)

x_train[[ 'MonthlyCharges']] = scaler.transform(scaled_train)

scaled_test=np.array(x_test[['MonthlyCharges']]).reshape(-1,1)

x_test[['MonthlyCharges']] = scaler.transform(scaled_test)


# In[324]:


x_train.describe()


# ## **SMOTE for imbalance data**
# Next, we oversample the data using SMOTE. We try oversampling the data to 55:45
# 
# **Random Oversampling: Randomly duplicate examples in the minority class.**
# 
# 
# Random Undersampling: Randomly delete examples in the majority class.
# 

# In[55]:


from imblearn.over_sampling import SMOTE


# In[56]:


from imblearn.datasets import make_imbalance
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced


# In[57]:


from imblearn.over_sampling import SMOTE


# In[58]:


from imblearn.over_sampling import SMOTE

sm = SMOTE(sampling_strategy = 0.8, k_neighbors=5, random_state=5)
x_resample, y_resample = sm.fit_resample(x_train, y_train)
Counter(y_resample)


# In[59]:


x_resample.shape


# you can see that the data can balance class 0 and class 1
# 
# And we also try oversample 60:40

# In[60]:


sm = SMOTE(sampling_strategy = 0.66, k_neighbors=5, random_state=5)

x_resample_2, y_resample_2 = sm.fit_resample(x_train, y_train)
Counter(y_resample_2)


# # Machine Learning Modelling

# In[62]:


get_ipython().system('pip install lightgbm')


# In[63]:


#using 4 model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier


# ---------------------------------------------------

# ## Random Forest Classification without oversample

# In[122]:


rf = RandomForestClassifier(random_state=5, criterion='entropy', n_estimators=18, max_depth=12)
rf.fit(x_train, y_train)  
prediction = rf.predict(x_test)
print(confusion_matrix(y_test, prediction))
print("Akurasi dari Random Forest adalah: %.2f" % (accuracy_score(y_test, prediction)*100) )

print("Recall dari Random Forest adalah:",recall_score(y_test, prediction)*100)

print("Precision dari Random Forest adalah:",precision_score(y_test, prediction)*100)


# In[123]:


from sklearn.metrics import f1_score, precision_score, recall_score #pip install sklearn
print('Recall: {}'.format(recall_score(y_test, prediction)))
print('Precision: {}'.format(precision_score(y_test, prediction)))
print('F1-Score: {}'.format(f1_score(y_test, prediction)))


# In[ ]:


fpr, tpr, _ = roc_curve(Y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    average_precision = average_precision_score(Y_test, y_scores)
    precision, recall, _ = precision_recall_curve(Y_test, y_scores)
    


# ## **Hyperparameter Tuning**
# Choose following method for hyperparameter tuning
# RandomizedSearchCV --> Fast
# GridSearchCV
# Assign hyperparameters in form of dictionery
# Fit the model
# Check best paramters and best score
# 

# In[130]:


from sklearn.model_selection import RandomizedSearchCV


# In[132]:


param_grid={
    'learning_rate':[1,0.5,0.1,0.01],
    'max_depth':[3,5,10,20],
    'n_estimators':[10,50,100,200]
}


# ## *Tidak menggunakan Data Training SMOTE*

# In[201]:


# XGBoost clasifier
from xgboost import XGBClassifier
xg = XGBClassifier(objective='binary:logistic')
modelxgb_hyperparam = xg.fit(x_train, y_train)
y_pred_modelxgb_hyperparam = modelxgb_hyperparam.predict(x_test)


# In[202]:


# checking accuracy of test dataset
print("testing accuracy is : ", modelxgb_hyperparam.score(x_test, y_test)*100)


# In[205]:


print(confusion_matrix(y_test, y_pred_modelxgb_hyperparam))
print("Akurasi dari Random Forest adalah: %.2f" % (accuracy_score(y_test, y_pred_modelxgb_hyperparam)*100) )
print("Recall dari Random Forest adalah:",recall_score(y_test, y_pred_modelxgb_hyperparam)*100)
print("Precision dari Random Forest adalah:",precision_score(y_test, y_pred_modelxgb_hyperparam)*100)
print('F1-Score: {}'.format(f1_score(y_test, y_pred_modelxgb_hyperparam)*100))


# In[212]:


test_xgb_h = pd.DataFrame({'Predicted value':y_pred_modelxgb_hyperparam, 'Actual value':y_test})
fig= plt.figure(figsize=(16,8))
test_xgb_h = test_xgb_h.reset_index()
test_xgb_h = test_xgb_h.drop(['index'],axis=1)
plt.plot(test_xgb_h[:15])
plt.legend(['Actual value','Predicted value'])


# In[209]:


y_test


# In[ ]:


# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(predictionxg.history['loss'], label='train')
pyplot.plot(predictionxg.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()


# ### Memilih data training tanpa di smote untuk dilanjukan ke step hyperparameter tuning

# RandomizedSearchCV

# In[230]:


from sklearn.model_selection import RandomizedSearchCV


# In[231]:


param_grid={
    'learning_rate':[1,0.5,0.1,0.01],
    'max_depth':[3,5,10,20],
    'n_estimators':[10,50,100,200]
}


# In[232]:


grid = RandomizedSearchCV(XGBClassifier(objective='binary:logistic'), param_grid, verbose=3)


# In[233]:


grid.fit(x_train, y_train)


# In[234]:


grid.best_params_


# In[332]:


xg_randomsearchCV = XGBClassifier(n_estimators= 50, max_depth= 3, learning_rate= 0.1)
xg_randomsearchCV = xg_randomsearchCV.fit(x_train, y_train)


# In[241]:


y_pred_xg_randomsearchCV = xg_randomsearchCV.predict(x_test)


# In[244]:


# checking accuracy of test dataset
print("testing accuracy is : ", xg_randomsearchCV.score(x_test, y_test)*100)

print(confusion_matrix(y_test, y_pred_xg_randomsearchCV))
print("Akurasi dari Random Forest adalah: %.2f" % (accuracy_score(y_test, y_pred_xg_randomsearchCV)*100) )
print("Recall dari Random Forest adalah:",recall_score(y_test, y_pred_xg_randomsearchCV)*100)
print("Precision dari Random Forest adalah:",precision_score(y_test, y_pred_xg_randomsearchCV)*100)
print('F1-Score: {}'.format(f1_score(y_test, y_pred_xg_randomsearchCV)*100))


# In[343]:


# fpr means false-positive-rate
# tpr means true-positive-rate
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_xg_randomsearchCV)

auc_score = metrics.auc(fpr, tpr)

# clear current figure
plt.clf()

plt.title('ROC Curve')
plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(auc_score))

# it's helpful to add a diagonal to indicate where chance 
# scores lie (i.e. just flipping a coin)
plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.legend(loc='lower right')
plt.show()


# TPR > FPR

# In[367]:


from sklearn.tree import xgboost


# In[370]:


get_ipython().system('pip install xgboost')


# In[368]:


get_ipython().system('pip install xgboost')


# In[245]:


plt.bar(range(len(xg_randomsearchCV.feature_importances_)), xg_randomsearchCV.feature_importances_)
plt.show()


# In[185]:


x_train.columns


# In[261]:


xg_randomsearchCV.get_booster().feature_names


# accuracy lebih besar menggunakan teknik hyperparameter tuning darirandomsearch cv dari XFBClassifier

# In[ ]:


## Performing 


# ### Model Benchmarking : K-Fold

# In[ ]:


models = [logreg,knn,dt,rf,xgb,lgbm]
score=[]
rata=[]
std=[]

for i in models:
    skfold=StratifiedKFold(n_splits=5)
    estimator=Pipeline([
        ('preprocess',transformer),
        ('model',i)])
    model_cv=cross_val_score(estimator,x_train,y_train,cv=skfold,scoring='roc_auc')
    score.append(model_cv)
    rata.append(model_cv.mean())
    std.append(model_cv.std())
    
pd.DataFrame({'model':['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'XGBoost', 'LightGBM'],'mean roc_auc':rata,'sdev':std}).set_index('model').sort_values(by='mean roc_auc',ascending=False)


# # CONCLUSION

# These are some of the quick insights from this exercise:
# 
# 1. Electronic check medium are the highest churners
# 2. Contract Type - Monthly customers are more likely to churn because of no contract terms, as they are free to go customers.
# 3. No Online security, No Tech Support category are high churners
# 4. Non senior Citizens are high churners
# 
# Note: There could be many more such insights, so take this as an assignment and try to get more insights :)

# ## *Menggunakan Data Training SMOTE*

# In[148]:


# XGBoost clasifier menggunakan SMOTE
from xgboost import XGBClassifier
xg_smote = XGBClassifier(objective='binary:logistic')
xg_smote.fit(x_resample, y_resample)


# In[158]:


# checking accuracy of test dataset
print("testing accuracy is : ", xg_smote.score(x_test, y_test)*100)
predictionxg_somte = xg.predict(x_test)


# In[159]:


from sklearn.metrics import f1_score, precision_score, recall_score #pip install sklearn
print('Recall: {}'.format(recall_score(y_test, predictionxg_somte)))
print('Precision: {}'.format(precision_score(y_test, predictionxg_somte)))
print('F1-Score: {}'.format(f1_score(y_test, predictionxg_somte)))


# In[144]:


grid = RandomizedSearchCV(XGBClassifier(objective='binary:logistic'), param_grid, verbose=3)


# In[ ]:


grid.fit(x_train, y_train)


# In[ ]:





# ## Random Forest Classification Oversample 55:45

# In[329]:


randomforest_oversample = rf.fit(x_resample, y_resample)
prediction = rf.predict(x_test)
print(confusion_matrix(y_test, prediction))
print("Akurasi dari Random Forest adalah: %.2f" % (accuracy_score(y_test, prediction)*100) )
print("Recall dari Random Forest adalah:",recall_score(y_test, prediction)*100)
print("Precision dari Random Forest adalah:",precision_score(y_test, prediction)*100)


# In[125]:


from sklearn.metrics import f1_score, precision_score, recall_score #pip install sklearn
print('Recall: {}'.format(recall_score(y_test, prediction)))
print('Precision: {}'.format(precision_score(y_test, prediction)))
print('F1-Score: {}'.format(f1_score(y_test, prediction)))


# jadi bener model naik 76.27 ke 76.47 jika tenure ggrup nya tanpa di gruping menjadi 6 group
# jadi kami better membiarkan tenure dengan range data asli dan di scaling menggunakan min_max() agar data nya standar tidak ada di luar range yang jauh me efesienkan model agar tidak terlalu complex

# ---------------------------------------------------

# ## Random Forest Classification Oversample 60:40

# In[128]:


rf.fit(x_resample_2, y_resample_2)

prediction = rf.predict(x_test)
print(confusion_matrix(y_test, prediction))

print("Akurasi dari Random Forest adalah: %.2f" % (accuracy_score(y_test, prediction)*100) )

print("Recall dari Random Forest adalah:",recall_score(y_test, prediction)*100)

print("Precision dari Random Forest adalah:",precision_score(y_test, prediction)*100)


# In[129]:


from sklearn.metrics import f1_score, precision_score, recall_score #pip install sklearn
print('Recall: {}'.format(recall_score(y_test, prediction)))
print('Precision: {}'.format(precision_score(y_test, prediction)))
print('F1-Score: {}'.format(f1_score(y_test, prediction)))


# ---------------------------------------------------

# # *Testing Model* kfold

# In[297]:


data['DeviceProtection'] = data['DeviceProtection'].replace(['No internet service'],'No')
data['OnlineBackup'] = data['OnlineBackup'].replace(['No internet service'],'No')
data['OnlineSecurity'] = data['OnlineSecurity'].replace(['No internet service'],'No')
data['TechSupport'] = data['TechSupport'].replace(['No internet service'],'No')


# In[288]:


transformer = ColumnTransformer([
    ('onehot', OneHotEncoder(drop='first'), ['Dependents','OnlineSecurity', 'OnlineBackup', 'InternetService', 'DeviceProtection', 'TechSupport', 'Contract','PaperlessBilling']),
], remainder='passthrough')


# In[300]:


x = data.drop(columns=['Churn'])
y = data['Churn']


# In[301]:


x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,test_size=0.2,random_state=2021)


# In[302]:


data


# In[303]:


testing = pd.DataFrame(transformer.fit_transform(x_train),columns=transformer.get_feature_names())
testing.head()


# In[326]:


model_cv=cross_val_score(estimator,x_train,y_train,cv=skfold,scoring='roc_auc')


# In[336]:


transformer = data_prep


# ---------------------------------------------------

# In[263]:


# Model Selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV,StratifiedKFold,train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import roc_curve, roc_auc_score, plot_roc_curve


# In[272]:


# Imbalance Dataset
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

# Ignore Warning
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Set max columns
pd.set_option('display.max_columns', None)


# In[270]:


# Feature Engineering
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# In[304]:


logreg = LogisticRegression()
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
xgb = XGBClassifier()
lgbm = lgb.LGBMClassifier()


# In[305]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer


# In[306]:


models = [logreg,knn,dt,rf,xgb,lgbm]
score=[]
rata=[]
std=[]


# In[314]:


x_train


# ---------------------------------------------------

# ## Gradient Boosting Classification without oversampling

# In[71]:


gb_clf = GradientBoostingClassifier(random_state=5, learning_rate= 1, loss= 'exponential', max_depth= 1, max_features= 1)

gb_clf.fit(x_train, y_train)
predictiongnb = gb_clf.predict(x_test)

print(confusion_matrix(y_test, predictiongnb))

print("Akurasi dari Gradient Boost adalah: %.2f" % (accuracy_score(y_test, predictiongnb)*100) )

print("Recall dari Gradient Boost adalah:",recall_score(y_test, predictiongnb)*100)

print("Precision dari Gradient Boost adalah:",precision_score(y_test, predictiongnb)*100)


# ## Gradient Boosting Classification oversampling 55:45

# In[72]:


gb_clf.fit(x_resample_2, y_resample_2)
predictiongnb = gb_clf.predict(x_test)
print(confusion_matrix(y_test, predictiongnb))
print("Akurasi dari Gradient Boost adalah: %.2f" % (accuracy_score(y_test, predictiongnb)*100) )
print("Recall dari Gradient Boost adalah:",recall_score(y_test, predictiongnb)*100)
print("Precision dari Gradient Boost adalah:",precision_score(y_test, predictiongnb)*100)
print("")


# ## Ada Boost without oversampling

# In[73]:


ada = AdaBoostClassifier(random_state=5, learning_rate=0.5, n_estimators=50)
ada.fit(x_train, y_train)
predictionada = ada.predict(x_test)
print(confusion_matrix(y_test, predictionada))
print("Akurasi dari Ada Boost adalah: %.2f" % (accuracy_score(y_test, predictionada)*100) )
print("Recall dari Ada Boost adalah:",recall_score(y_test, predictionada)*100)
print("Precision dari Ada Boost adalah:",precision_score(y_test, predictionada)*100)
print("")


# ## Ada Boost oversampling 55:45
# 

# In[74]:


ada.fit(x_resample, y_resample)
predictionada = ada.predict(x_test)
print(confusion_matrix(y_test, predictionada))
print("Akurasi dari Ada Boost adalah: %.2f" % (accuracy_score(y_test, predictionada)*100) )
print("Recall dari Ada Boost adalah:",recall_score(y_test, predictionada)*100)
print("Precision dari Ada Boost adalah:",precision_score(y_test, predictionada)*100)
print("")


# ## Ada Boost oversampling 60:40

# In[75]:


ada.fit(x_resample_2, y_resample_2)
predictionada = ada.predict(x_test)
print(confusion_matrix(y_test, predictionada))
print("Akurasi dari Ada Boost adalah: %.2f" % (accuracy_score(y_test, predictionada)*100) )
print("Recall dari Ada Boost adalah:",recall_score(y_test, predictionada)*100)
print("Precision dari Ada Boost adalah:",precision_score(y_test, predictionada)*100)
print("")


# ## LightGBM without oversampling
# 

# In[76]:


lgbm = LGBMClassifier(random_state=5, learning_rate= 0.05, n_estimators= 90, num_leaves= 20, boosting_type='dart')
lgbm.fit(x_train, y_train)
predictionlgbm = lgbm.predict(x_test)
print(confusion_matrix(y_test, predictionlgbm))
print("Akurasi dari LightGBM adalah: %.2f" % (accuracy_score(y_test, predictionlgbm)*100) )
print("Recall dari LightGBM adalah:",recall_score(y_test, predictionlgbm)*100)
print("Precision dari LightGBM adalah:",precision_score(y_test, predictionlgbm)*100)
print("")


# ## LightGBM oversampling 55:45
# 

# In[78]:


lgbm.fit(x_resample, y_resample)
predictionlgbm = lgbm.predict(x_test)
print(confusion_matrix(y_test, predictionlgbm))
print("Akurasi dari LightGBM adalah: %.2f" % (accuracy_score(y_test, predictionlgbm)*100) )
print("Recall dari LightGBM adalah:",recall_score(y_test, predictionlgbm)*100)
print("Precision dari LightGBM adalah:",precision_score(y_test, predictionlgbm)*100)
print("")


# ## LightGBM oversampling 60:40
# 

# In[80]:


lgbm.fit(x_resample_2, y_resample_2)
predictionlgbm = lgbm.predict(x_test)
print(confusion_matrix(y_test, predictionlgbm))
print("Akurasi dari LightGBM adalah: %.2f" % (accuracy_score(y_test, predictionlgbm)*100) )
print("Recall dari LightGBM adalah:",recall_score(y_test, predictionlgbm)*100)
print("Precision dari LightGBM adalah:",precision_score(y_test, predictionlgbm)*100)
print("")


# ## Feature Selection

# Finding out the best feature which will contribute and have good relation with target variable. Following are some of the feature selection methods,
# 
# **heatmap**
# **feature_importance_**
# **SelectKBest**

# In[114]:


pip install -U scikit-learn


# In[115]:


# important feature using extratreeregressor

from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(x_train,y_train)


# In[116]:


print(selection.feature_importances_)


# In[118]:


#plot graph of feature importances for better visualization

plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=x_train.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


# # *Saving Model

# In[354]:


import pickle


# In[355]:


file = 'xgboost_model.pkl'
pickle.dump(xg_randomsearchCV, open(file, 'wb'))


# In[356]:


load_model = pickle.load(open(file, 'rb'))


# In[357]:


y_prep = load_model.predict(x_test)


# In[358]:


y_prep


# In[ ]:




