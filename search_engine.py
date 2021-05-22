# -*- coding: utf-8 -*-
"""
Created on Sat May 22 18:35:05 2021

@author: JPadariya
"""


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
