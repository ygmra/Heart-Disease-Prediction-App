import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter.ttk import Combobox
from tkinter import Canvas
from tkinter import ttk
from tkinter import Label
from tkinter import Button
from tkinter import messagebox
from tkinter import Entry
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


#reading dataset


data = pd.read_csv('/Users/YAĞMUR/Downloads/heart.csv') # her kullanıcı kendi pathini koymalı


#/Users/YAĞMUR/Downloads/heart.csv
#/kaggle/input/heart-disease-uci/heart.csv


data.head() #1=male 0=female
data.shape
data.info()

#YAĞMUR
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n\n")
print('Statistical Measurements\n')

pd.set_option("display.float", "{:.2f}".format)
data.describe() #statistical measurements


#missing value check
data.isnull().sum()




window = tk.Tk()

window.title("Heart Disease Prediction System")
window.configure(background='#F0F8FF')
window.geometry("1700x900")


def successMessage():
    messagebox.showinfo(title="Result", message="No Risk! \n"
                                                )

def warningMessage():
    messagebox.showwarning(title="Result", message="There is a risk!"
                                                   )




def predictionWindow():



    # Splitting to Features And Target

    X = data.drop(columns='target', axis=1)
    y = data['target']

    print(X)  # target kolonun yok olduğunu gördük
    print(y)  # targetı tek aldık


    # Splitting train and test data in dataset

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2, stratify=y)
    print(X.shape, X_train.shape, X_test.shape)  # testsize kısmında verdiğimiz değerle oluşuyor

    # Model Traning


    """""
    #Decision Tree Accuracy: 0.6923076923076923
    from sklearn import tree
    dt_clf = tree.DecisionTreeClassifier(max_depth=5)
    dt_clf.fit(X_train, y_train)

    dt_clf.score(X_test, y_test)

    y_predict = dt_clf.predict(X_test)
    a = dt_clf.score(X_test, y_test)
    print('Testing Data Accuracy: ', (a * 100), '%') 

    """
    #Model training


    """""
    #Random Forest 0.8241758241758241
    from sklearn import ensemble
    rf_clf = ensemble.RandomForestClassifier(n_estimators=100)
    rf_clf.fit(X_train, y_train) # columnlarla target arası ilişkiyi kuruyor
    b= rf_clf.score(X_test, y_test)
    print('Testing Data Accuracy: ', (b * 100), '%')

    """""



    """""
    # logistic regression

    model = LogisticRegression()
    model.fit(X_train, y_train)  

    X_train_predict = model.predict(X_train)
    accuracyOfTrainingData = accuracy_score(X_train_predict, y_train)
    print('Training Data Accuracy: ', (accuracyOfTrainingData * 100), '%')

    X_test_predict = model.predict(X_test)
    accuracyOfTestingData = accuracy_score(X_test_predict, y_test)

    print('Testing Data Accuracy: ', (accuracyOfTestingData * 100), '%')
    """




    from sklearn.svm import SVC
    model = SVC(probability=True, kernel='linear')
    model.fit(X_train, y_train)

    c = model.score(X_train, y_train)
    d = model.score(X_test, y_test)
    print('Training Data Accuracy : ', (c * 100), '%')
    print('Testing Data Accuracy : ', (d * 100), '%')




    def get_birth():

        value=int(age_get.get())
        date=2022
        birthyear = (date - value)
        return birthyear


    def sexEdit():
        global sex
        sex_value=sex_get.get()
        if(sex_value == "Female"):
            sex=1
        elif(sex_value == "Male"):
            sex=0
        print(sex)

    def chestPainEdit():
        global cptype
        value = cpt_get.get()
        if(value == "Typical angina"):
            cptype=0
        elif(value == "Atypical angina" ):
            cptype=1
        elif(value == "Non-anginal pain"):
            cptype=2
        elif(value== "Asymptomatic"):
            cptype=3


        print(cptype)

    def trestEdit():

        trestbps= int(trest_get.get()) #return olarak döndürsek alırız

        return trestbps

    def fbsEdit():
        global fbs
        value= fbs_get.get()
        if(value== "fasting blood sugar > 120 mg/dl"):
            fbs= 1
        elif(value== "fasting blood sugar < 120 mg/dl" ):
            fbs=0
        print(fbs)


    def cholEdit():

        chol = int(chol_get.get())

        return chol

    def restecgEdit():
        global restecg
        value = ecg_get.get()
        if(value== "Nothing to note"):
            restecg=0
        elif(value== "ST-T Wave abnormality"):
            restecg =1
        elif(value== "Definite left ventricular hypertrophy" ):
            restecg= 2
        print(restecg)

    def thalachEdit():

        thalach= int(tha_get.get())
        return thalach

    def exangEdit():
        global exang
        value = ex_get.get()
        if(value == "Yes"):
            exang = 1
        elif(value == "No"):
            exang =0
        print(exang)



    def oldpeakEdit():

        oldpeak = float(oldpeak_get.get())
        return oldpeak

    def slopeEdit():
        global slope
        value = slope_get.get()
        if(value == "Upsloping: better heart rate with excercise"):
            slope = 0
        elif(value == "Flatsloping: minimal change" ):
            slope =1
        elif(value== "Downslopins: signs of unhealthy heart"):
            slope =2

        print(slope)


    def thalEdit():
        global thal
        value = thal_get.get()
        if(value == "Normal"):
            thal = 0
        elif(value== "Fixed Defect"):
            thal =1
        elif(value == "Reversable Defect"):
            thal =2

        print(thal)



    def caEdit():
        global ca
        value = ca_get.get()
        if (value == "0"):
            ca=0
        elif(value == "1"):
            ca=1
        elif(value == "2"):
            ca=2
        elif(value == "3"):
            ca=3
        print(ca)


    def resultWindow():

        a =get_birth()
        c= trestEdit()
        d= cholEdit()
        e =thalachEdit()
        f =oldpeakEdit()

        inputVariable = (a,sex,cptype,c,d,fbs,restecg,e,exang,f,slope,ca,thal)  # biz bunu kullanıcıdan alıcaz
        #a,sex,b,c,d,fbs,restecg,e,exang,f,slope,ca,thal
        # change the input array to the numpy array
        inputNumpyArray = np.asarray(inputVariable)

        inputArrayReshape = inputNumpyArray.reshape(1, -1)

        prediction = model.predict(inputArrayReshape)

        print("Prediction result: ", prediction)

        #IŞIK

        if(prediction == [0]):
            successMessage()
        elif(prediction== [1]):
            warningMessage()









    preWindow = tk.Tk()
    preWindow.title("Prediction Of Heart Disease Risk")
    preWindow.configure(background='#F0F8FF')
    preWindow.geometry("1700x900")

    t_label = Label(preWindow, text="Prediction Of Heart Disease Risk", font="helvetica 30", borderwidth=20)
    t_label.place(x=20, y=20)
    t2_label = Label(preWindow, text="Please enter the patient information and press choose buttons", font="helvetica 15", borderwidth=10)
    t2_label.place(x=20, y=100)

    age_label = Label(preWindow, text="Enter birth year", font="helvetica 12", borderwidth=6)
    age_label.place(x=80, y=200)
    age_get = Entry(preWindow, width=10)
    age_get.place(x=80, y=250)


    sex_label = Label(preWindow, text="Enter Sex", font="helvetica 12", borderwidth=6)
    sex_label.place(x=250, y=200)

    sex_array =["Female", "Male"]
    sex_get = Combobox(preWindow, values=sex_array)
    sex_get.place(x=250, y=250)

    sex_button= Button(preWindow, text="Choose", command=sexEdit, font="helvetica 12", borderwidth=3)
    sex_button.place(x=400, y=250)



    cpt_label = Label(preWindow, text="Choose chest pain type"
                                      , font="helvetica 12", borderwidth=6)
    cpt_label.place(x=500, y=200)

    cpt_array = ["Typical angina", "Atypical angina",
                 "Non-anginal pain", "Asymptomatic"]
    cpt_get = Combobox(preWindow, values=cpt_array )
    cpt_get.place(x=500, y=250)

    cpt_button = Button(preWindow, text="Choose", command=chestPainEdit, font="helvetica 12", borderwidth=6)
    cpt_button.place(x=650, y=250)


    ###trestbps
    trest_label = Label(preWindow, text="Enter resting blood pressure (in mm Hg)", font="helvetica 12", borderwidth=6)
    trest_label.place(x=750, y=200)
    trest_get = Entry(preWindow, width=10)
    trest_get.place(x=750, y=250)



    ###fbs

    fbs_label = Label(preWindow, text="Enter fasting blood sugar", font="helvetica 12", borderwidth=6)
    fbs_label.place(x=1100, y=200)

    fbs_array = ["fasting blood sugar > 120 mg/dl", "fasting blood sugar < 120 mg/dl"]
    fbs_get = Combobox(preWindow, values=fbs_array, width=28)
    fbs_get.place(x=1100, y=250)

    fbs_button = Button(preWindow, text="Choose", command=fbsEdit, font="helvetica 12", borderwidth=3)
    fbs_button.place(x=1300, y=250)

    #### chol
    chol_label = Label(preWindow, text="Enter serum cholestoral\n(in mg/dl", font="helvetica 12",
                        borderwidth=6)
    chol_label.place(x=80, y=350)
    chol_get = Entry(preWindow, width=10)
    chol_get.place(x=80, y=400)

    ##### restecg

    ecg_label = Label(preWindow, text="Enter resting electrocardiographic results", font="helvetica 12", borderwidth=6)
    ecg_label.place(x=300, y=350)

    ecg_array = ["Nothing to note", "ST-T Wave abnormality", "Definite left ventricular hypertrophy"]
    ecg_get = Combobox(preWindow, values=ecg_array, width=30)
    ecg_get.place(x=300, y=400)

    ecg_button = Button(preWindow, text="Choose", command=restecgEdit, font="helvetica 12", borderwidth=3)
    ecg_button.place(x=530, y=400)

    ##thalach

    tha_label = Label(preWindow, text="Enter maximum heart rate achieved", font="helvetica 12",
                       borderwidth=6)
    tha_label.place(x=650, y=350)
    tha_get = Entry(preWindow, width=10)
    tha_get.place(x=650, y=400)


    ##exang

    ex_label = Label(preWindow, text="Enter exercise induced angina\n(yes;no)", font="helvetica 12", borderwidth=6)
    ex_label.place(x=950, y=350)

    ex_array = ["Yes", "No"]
    ex_get = Combobox(preWindow, values=ex_array)
    ex_get.place(x=950, y=400)

    ex_button = Button(preWindow, text="Choose", command=exangEdit, font="helvetica 12", borderwidth=3)
    ex_button.place(x=1100, y=400)


    ###oldpeak

    oldpeak_label = Label(preWindow, text="Enter ST depression induced by exercise\nrelative to rest\n(Ex:3.1)", font="helvetica 12",
                      borderwidth=6)
    oldpeak_label.place(x=1200, y=350)
    oldpeak_get = Entry(preWindow, width=10)
    oldpeak_get.place(x=1200, y=400)


    ##slope

    slope_label = Label(preWindow, text="Enter the slope of the peak exercise ST segment", font="helvetica 12", borderwidth=6)
    slope_label.place(x=80, y=500)

    slope_array = ["Upsloping: better heart rate with excercise", "Flatsloping: minimal change", "Downslopins: signs of unhealthy heart"]
    slope_get = Combobox(preWindow, values=slope_array, width=35)
    slope_get.place(x=80, y=550)

    slope_button = Button(preWindow, text="Choose", command=slopeEdit, font="helvetica 12", borderwidth=3)
    slope_button.place(x=330, y=550)



    ##ca

    ca_label = Label(preWindow, text="Enter number of major vessels (0-3) colored by flourosopy\n"
                                     "Note: The more blood movement the better ", font="helvetica 12",
                        borderwidth=6)
    ca_label.place(x=450, y=500)

    ca_array = ["0", "1", "2", "3"]
    ca_get = Combobox(preWindow, values=ca_array)
    ca_get.place(x=450, y=550)

    ca_button = Button(preWindow, text="Choose", command=caEdit, font="helvetica 12", borderwidth=3)
    ca_button.place(x=600, y=550)


    ###thal

    thal_label = Label(preWindow, text="Enter thalium stress result", font="helvetica 12", borderwidth=6)
    thal_label.place(x=900, y=500)

    thal_array = ["Normal", "Fixed Defect", "Reversable Defect"]
    thal_get = Combobox(preWindow, values=thal_array)
    thal_get.place(x=900, y=550)

    thal_button = Button(preWindow, text="Choose", command=thalEdit, font="helvetica 12", borderwidth=3)
    thal_button.place(x=1050, y=550)


    ##calculation button

    calculation_button = Button(preWindow, text="Predict the risk of Heart Disease", command=resultWindow, font="helvetica 12", borderwidth=10)
    calculation_button.place(x=650, y=650)







##ana window

#IŞIK
title_label = Label(window, text="Anticipate The Risk Of Heart Disease", font="helvetica 30", borderwidth=20, padx=400, pady=20, background= "#E0EEEE" )
title_label.place(x=20, y=20)
title2_label = Label(window, text="Welcome to the YIZApp\n\n\n"
                                  "Health Comes First", font="helvetica 15", borderwidth=15, padx=400, pady=20, background= "#E0EEEE" )
title2_label.place(x=200, y=120)



prediction_label = Label(text="Hit the Prediction Button to Find Out Heart Disease Risk", font ="helvetica 20", borderwidth=20)
prediction_label.place(x=400, y=300)
prediction_button = Button(window, text="Prediction", command=predictionWindow, font="helvetica 15", borderwidth=6)
prediction_button.place(x=700, y=400)


window.mainloop()

