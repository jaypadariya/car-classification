# -*- coding: utf-8 -*-
"""
Created on Sat May 22 10:42:20 2021

@author: JPadariya
"""


import os
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
from pandas import DataFrame as df
from PIL import Image
#from pytesseract import pytesseract
from PyPDF2 import PdfFileReader
import numpy as np
import re


import cv2
def extract_information(pdf_path):
    with open(pdf_path, 'rb') as f:
        pdf = PdfFileReader(f)
        information = pdf.getDocumentInfo()
        number_of_pages = pdf.getNumPages()

    txt = f"""
    Information about {pdf_path}: 



    Title: {information.title}
    Number of pages: {number_of_pages}
    """

    print(txt)
    return information


pdflist=[]

    # pytesseract.pytesseract.tesseract_cmd = r'C:\Users\jpadariya\tesseract.exe'
    # filePath = r'C:\Users\jpadariya\Downloads\assignment\assignment\dataset\car2.pdf'
folder=r"C:\Users\jpadariya\Downloads\assignment\assignment\dataset"
allcombined=[]
lisysn=os.listdir(folder)
for i in lisysn:
    jj=os.path.join(folder,i)
    list1=[]
    functionality=[]
    total_seats=[]
    model=[]
    length1=[]
    width1=[]
    height1=[]
    words1=''
    new=''
    

    pdflist.append(jj)
    filePath=jj
    doc = convert_from_path(filePath)
    path, fileName = os.path.split(filePath)
    fileBaseName, fileExtension = os.path.splitext(fileName)
    
    
    pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract-OCR\Tesseract-OCR\tesseract.exe'
    for i in doc:
        image = np.array(i)
        img2=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        words = None
        # words = pytesseract.image_to_string(img2, lang = 'eng').encode("utf-8")
        words = pytesseract.image_to_string(img2, lang = 'eng')
        lower= words.lower()
        #words=words.split(",")
        words1 = words1 + lower
        
        
    list1.append(words1)
        
    text=words1.split(",")
    
    targetlist=[ "length" , 'width' , 'height']
    new_target=['petrol','cng','diesel','roof','abs','climate control','automatic transmission']
    
    for s in text:
        words = s.split()
        for target in targetlist:
            
            for i,w in enumerate(words):
                if w == target:
                    # next word
                    try:
                        # print (target,words[i+1],words[i+2],words[i+3],words[i+4])
                        numstr=(target + words[i+1] + words[i+2] + words[i+3] +words[i+4])
                        values=re.findall(r'\d{4}', numstr)
                    
                        if target == 'length':
                            length1.append(values)
                        if target == 'width':
                            width1.append(values)
                        if target == 'height':
                            height1.append(values)
                    except:
                        pass
                try:
                    if w == 'seater':
                        seaters=(words[i-1])
                        seatsa=re.findall(r'\d{1}', seaters)
                        total_seats.append(seatsa)
                        break
                except:
                    pass    
    for ii in new_target:
        if ii in words1:
            print(ii)
            new=new +',' + ii 
    functionality.append(new)
            
                        
    # temp = re.findall(r'\d+', numstr)
    # res = list(map(int, temp))
    all_pic=[]
    all_pic1=df(all_pic,columns=['car_model'])
    try:
        lenth11=max(length1,key=length1.count)
        if lenth11 ==[]:
            lenth11=['nan']

    except:
        lenth11=['nan']
    try:
        widht11=max(width1,key=width1.count)
        if widht11 ==[]:
            widht11=['nan']
    except:
        widht11=['nan']
    try:
        height11=max(height1,key=height1.count)
        if height11 ==[]:
            height11=['nan']
    except:
        height11=['nan']
    try:
        total_seats1=max(total_seats,key=total_seats.count)
        if total_seats1 ==[]:
            total_seats1=['nan']

    except:
        total_seats1=['nan']
        
    
    title=extract_information(filePath)
    try:
        title_value=title['/Title']
        model.append(title_value)    
    except:
        title_value='nan'
        model.append(title_value)    

        
    
    all_pic1=df({'car_model':model , 'extractedd_text' : list1 , 'functionality' : functionality, 'total_seats' : total_seats1, 'height' : height11,'width':widht11,'length':lenth11})
    del doc,list1,functionality,total_seats1,lenth11,widht11,height11,image,img2
    
    
     
    allcombined.append(all_pic1)
    del all_pic1
    

print(len(list1))

len(functionality)
len(total_seats1)
len(model)
len(lenth11)
len(widht11)
len(height11)


#%%
allcombined.to_csv(r"C:\Users\jpadariya\Downloads\assignment\assignment\res\cars_daat_specification.csv")
# import pandas as pd
# merged = pd.concat(allcombined)

#%%
#classification
import numpy as np
import pandas as pd
filename =r'C:\Users\jpadariya\Downloads\assignment\assignment\res\final.csv'
dff = pd.read_csv(filename, encoding= 'unicode_escape')
dff.rename(columns={"": "cng"})
dff['class']='0'

for inde,i in enumerate(dff['car_model']):
    if int(dff['length'][inde]) < 4000  or (int(dff['total_seats'][inde]) < 5) :
        dff['class'][inde] = '1'
    if int(dff['length'][inde]) > 4000 or (int(dff['total_seats'][inde]) > 5):
        dff['class'][inde] = '2'


s  = dff['functionality'].str.replace("'",'').str.split(',').explode().to_frame()

cols = s['functionality'].drop_duplicates(keep='first').tolist()

df2 = pd.concat([dff, pd.crosstab(s.index, s["functionality"])[cols]], axis=1).replace(
    {1: True, 0: False}
)
print(df2)
df2=df2.rename(columns={"": "cng"})
   

df2.to_csv(r"C:\Users\jpadariya\Downloads\assignment\assignment\res\training_file.csv")

#%%

conditions = [
    (dff['length'] < 4) & (dff['total_seats'] < 5),
    (dff['length'] > 4) and (dff['total_seats'] > 5),
    ]


# create a list of the values we want to assign for each condition
values = ['1', '2']
dff['class']='0'
# create a new column and use np.select to assign values to it using our lists as arguments
dff['class'] = np.select(conditions, values)

# display updated DataFrame
dff.head()
    
#%%
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import mean_squared_error
#import libraries
from datetime import datetime, timedelta,date
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_predict
%matplotlib inline
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from __future__ import division
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
#do not show warnings
import warnings
warnings.filterwarnings("ignore")

#import machine learning related libraries
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import pickle
import joblib


df2=pd.read_csv(r"C:\Users\jpadariya\Downloads\assignment\assignment\res\training_file.csv",encoding= 'unicode_escape')

y= df2['class']


X=df2[df2.columns[3:]]


X=X.astype(int)
# s = np.array(byte_list)
# X = np.frombuffer(s, dtype=np.uint8)
# y = np.frombuffer(s, dtype='S1')
# X, y

# XX = np.reshape(X, (-1, 1))
# yy =np.reshape(y, (-1, 1))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

#create an array of models
models = []
models.append(("LR",LogisticRegression()))
models.append(("NB",GaussianNB()))
models.append(("RF",RandomForestClassifier()))
models.append(("SVC",SVC()))
models.append(("Dtree",DecisionTreeClassifier()))
models.append(("XGB",xgb.XGBClassifier()))
models.append(("KNN",KNeighborsClassifier()))


for name,model in models:
    kfold = KFold(n_splits=4, random_state=2)
    cv_result = cross_val_score(model,X_train,y_train, cv = kfold,scoring = "accuracy")
    print(name, cv_result)
    y_pred = cross_val_predict(model, X, y, cv=5)
    
    # with open("models_{}.pckl".format(name), "wb") as f:
    #   pickle.dump(model, f)
#    filename = 'a1q4-{}.sav'.format(name)
#    joblib.dump(model, filename)
    print(confusion_matrix(y, y_pred))
    print(accuracy_score(y, y_pred))
    

 
    # this_column = df.columns[i]
    # df[this_column] = [i, i+1]
    
    
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)

######accuracy scores######

"""
LR [0.71428571 0.85714286 0.85714286 0.42857143]
0.6857142857142857

NB [0.71428571 0.85714286 1.         0.57142857]
0.7428571428571429

RF [0.85714286 0.85714286 1.         0.85714286]
0.8571428571428571


SVC [0.57142857 0.85714286 1.         0.28571429]
0.6

Dtree [1.         1.         1.         0.85714286]
0.9428571428571428

Xgb
0.8857142857142857


KNN [0.71428571 1.         1.         0.28571429]
0.8
"""

#%%

#search engine


# importing pandas package
import pandas as pd

# making data frame from csv file
df2=pd.read_csv(r"C:\Users\jpadariya\Downloads\assignment\assignment\res\training_file.csv",encoding= 'unicode_escape')

df2.columns =[column.replace(" ", "_") for column in df2.columns]

options =['petrol','cng','diesel','sunroof','climate_control']


# Print out your options
for i in range(len(options)):
    print(str(i+1) + ":", options[i])

# Take user input and get the corresponding item from the list
inp = int(input("Enter your choice pecification: "))
if inp in range(1, 7):
    inp = options[inp-1]
    if inp=='petrol':
        df2.query("petrol == True", inplace = True)
        newaa=df2["car_model"]
        print(f'{newaa}')
    if inp=='cng':
        df2.query("cng == True", inplace = True)
        newaa=df2["car_model"]
        print(f'{newaa}')
    if inp=='diesel':
        df2.query("diesel == True", inplace = True)
        newaa=df2["car_model"]
        print(f'{newaa}')
    if inp=='sunroof':
        df2.query("roof == True", inplace = True)
        newaa=df2["car_model"]
        print(f'{newaa}')
    if inp=='climate_control':
        df2.query("climate_control == True", inplace = True)
        newaa=df2["car_model"]
        print(f'{newaa}')


else:
    print("Invalid input!")


#%%
# import os
# nerr=[]
# path=r'C:\Users\jpadariya\Downloads\assignment\assignment\res'
# for i in os.listdir(path):
#     new=os.path.join(path,i)
#     dataf=pd.read_csv(new)
#     nerr.append(dataf)

# import pandas as pd
# merged = pd.concat(nerr)


# merged.drop_duplicates(subset ="extractedd_text",
#                      keep = False, inplace = True)




#%%

    
#%%
###########dashboard for datashowing with flask############

#dash show drop down 
import dash
import dash_html_components as html
import dash_core_components as dcc
df2=pd.read_csv(r"C:\Users\jpadariya\Downloads\assignment\assignment\res\training_file.csv",encoding= 'unicode_escape')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
df2.columns =[column.replace(" ", "_") for column in df2.columns]

df2.query("petrol == True", inplace = True)
petrol=df2["car_model"]


df2.query("cng == True", inplace = True)
cng=df2["car_model"]

df2.query("diesel == True", inplace = True)
diesel=df2["car_model"]


df2.query("roof == True", inplace = True)
sunroof=df2["car_model"]

df2.query("climate_control == True", inplace = True)
climate_control=df2["car_model"]


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    dcc.Dropdown(
        id='demo-dropdown',
        options=[
            {'label': 'petrol', 'value': '{}'.format(petrol)},
            {'label': 'cng', 'value': '{}'.format(cng)},
            {'label': 'diesel', 'value': '{}'.format(diesel)},
            {'label': 'sunroof', 'value': '{}'.format(sunroof)},
            {'label': 'climate_control', 'value': '{}'.format(climate_control)}
        ],
        value='{}'.format(petrol)
    ),
    html.Div(id='dd-output-container')
])


@app.callback(
    dash.dependencies.Output('dd-output-container', 'children'),
    [dash.dependencies.Input('demo-dropdown', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False) 