# -*- coding: utf-8 -*-
"""
Created on Sat May 22 18:34:21 2021

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
