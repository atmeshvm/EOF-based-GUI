from tkinter import * 
from tkinter.ttk import *
from tkdocviewer import *
from tkinter.filedialog import askopenfile
import eofs
from eofs.standard import Eof
from eofs.standard import EEof
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv
import os
import sys
import glob
import scipy.linalg as la

from numpy import eye, asarray, dot, sum, diag
from numpy.linalg import svd

import scipy.fftpack
import glob

np.set_printoptions(threshold=sys.maxsize)

#For Add Export to CSV Functionality (Possibly Implement if user wants)
def exp_to_csv(res):
    
    res.to_csv()
  

def proceed_acn():
    MainPage = Tk()
    MainPage.geometry('450x450')
    ptn0 = Button(MainPage, text ='View Dataset', command = view_win)
    ptn0.place(x=175, y=10)
    a = Button(MainPage, text ='Regular EOF', command = SimpEOF)
    a.place(x=175, y=70)
    b = Button(MainPage, text ='Types of EOF', command = TypeEOF)
    b.place(x=175, y=130)
    c = Button(MainPage, text ='EOF by different decomposition', command = DecompEOF)
    c.place(x=135, y=190) 
    MainPage.mainloop()


def SimpEOF():
    simple_eof = Tk()
    simple_eof.geometry('400x400')
    simple_eof.title('Regular EOF')
    ptn1 = Button(simple_eof, text ='See PC', command = cal_pc)
    ptn1.place(x=175, y=10)
    ptn2 = Button(simple_eof, text ='See EOF', command = cal_eof)
    ptn2.place(x=165, y=60)
    ptn3 = Button(simple_eof, text ='Find EOF as Correlation', command = cal_creof)
    ptn3.place(x=150, y=110)
    ptn4 = Button(simple_eof, text ='Find EOF as Covariance', command = cal_cveof)
    ptn4.place(x=150, y=160)
    ptn5 = Button(simple_eof, text ='Eigenvalues', command = cal_eig)
    ptn5.place(x=175, y=210)
    ptn6 = Button(simple_eof, text ='Variance Fraction', command = cal_varfrac)
    ptn6.place(x=165, y=260)
    ptnhome = Button(simple_eof, text ='Home', command = proceed_acn)
    ptnhome.place(x=175, y=310)
    simple_eof.mainloop()


def TypeEOF():
    type_eof = Tk()
    type_eof.geometry('400x400')
    type_eof.title('Tyes of EOF')
    ptn2t = Button(type_eof, text ='Regular EOF', command = cal_eof)
    ptn2t.place(x=175, y=10)
    ptn7 = Button(type_eof, text ='Rotated EOF', command = cal_reof)
    ptn7.place(x=175, y=80)
    ptn8 = Button(type_eof, text ='Extended EOF', command = accept_lag)
    ptn8.place(x=170, y=150)   
    ptn9 = Button(type_eof, text ='Complex EOF', command = cal_ceof)
    ptn9.place(x=175, y=220)  
    ptnhome1 = Button(type_eof, text ='Home', command = proceed_acn)
    ptnhome1.place(x=175, y=290)
    type_eof.mainloop()


def DecompEOF():
    decomp_eof = Tk()
    decomp_eof.geometry('400x400')
    decomp_eof.title('EOF by Decomposition')
    ptn12c = Button(decomp_eof, text ='EOF by SVD', command = cal_eof)
    ptn12c.place(x=175, y=10)
    ptn10 = Button(decomp_eof, text ='EOF by Eigen Decomposition', command = eof_eig_decomp)
    ptn10.place(x=175, y=80)  
    ptn11 = Button(decomp_eof, text ='EOF by Decomposition EOF', command = eof_lu_decomp)
    ptn11.place(x=175, y=150) 
    ptnhome2 = Button(decomp_eof, text ='Home', command = proceed_acn)
    ptnhome2.place(x=175, y=220)
    decomp_eof.mainloop()

def open_csvfile():
    global file
    file = askopenfile(mode ='r', filetypes =[('*.txt', '*.csv')])
    if file is not None:
        global content
        content = file.read()

def view_win():
    view_window = Tk()
    view_window.geometry('400x400')
    v = DocViewer(view_window)
    v.pack(side="top", expand=1, fill="both")
    v.display_file(content)
    view_window.mainloop()

  

def cal_pc(): 
    file1 = open("result.txt","w")
    file1.truncate(0)
    df = pd.read_csv(file.name)
    df = df.drop(df.columns[0],axis=1)
    solver = Eof(df.to_numpy())
    pcs = solver.pcs()
    t = np.array_str(pcs)
    file1.write(t)
    file1.close()
    file2 = open("result.txt","r+")
    view_window = Tk()
    view_window.geometry('400x400')
    v = DocViewer(view_window)
    v.pack(side="top", expand=1, fill="both")
    v.display_file(file2.read())
    view_window.mainloop()
    file2.close()

def cal_eof(): 
    file1 = open("result.txt","w")
    file1.truncate(0)
    df = pd.read_csv(file.name)
    df = df.drop(df.columns[0],axis=1)
    solver = Eof(df.to_numpy())
    eofs = solver.eofs()
    t = np.array_str(eofs)
    file1.write(t)
    file1.close()
    file2 = open("result.txt","r+")
    view_window1 = Tk()
    view_window1.geometry('400x400')
    v = DocViewer(view_window1)
    v.pack(side="top", expand=1, fill="both")
    v.display_file(file2.read())
    view_window.mainloop()
    file2.close()


def cal_creof(): 
    file1 = open("result.txt","w")
    file1.truncate(0)
    df = pd.read_csv(file.name)
    df = df.drop(df.columns[0],axis=1)
    solver = Eof(df.to_numpy())
    creofs = solver.eofsAsCorrelation()
    print(solver.totalAnomalyVariance())
    t = np.array_str(creofs)
    file1.write(t)
    file1.close()
    file2 = open("result.txt","r+")
    view_window = Tk()
    view_window.geometry('400x400')
    v = DocViewer(view_window)
    v.pack(side="top", expand=1, fill="both")
    v.display_file(file2.read())
    view_window.mainloop()
    file2.close()

def cal_cveof(): 
    file1 = open("result.txt","w")
    file1.truncate(0)
    df = pd.read_csv(file.name)
    df = df.drop(df.columns[0],axis=1)
    solver = Eof(df.to_numpy())
    cveofs = solver.eofsAsCovariance()
    t = np.array_str(cveofs)
    file1.write(t)
    file1.close()
    file2 = open("result.txt","r+")
    view_window = Tk()
    view_window.geometry('400x400')
    v = DocViewer(view_window)
    v.pack(side="top", expand=1, fill="both")
    v.display_file(file2.read())
    view_window.mainloop()
    file2.close()

def cal_eig(): 
    file1 = open("result.txt","w")
    file1.truncate(0)
    df = pd.read_csv(file.name)
    df = df.drop(df.columns[0],axis=1)
    solver = Eof(df.to_numpy())
    eig = solver.eigenvalues()
    t = np.array_str(eig)
    file1.write(t)
    file1.close()
    file2 = open("result.txt","r+")
    view_window = Tk()
    view_window.geometry('400x400')
    v = DocViewer(view_window)
    v.pack(side="top", expand=1, fill="both")
    v.display_file(file2.read())
    view_window.mainloop()
    file2.close()

def cal_varfrac(): 
    file1 = open("result.txt","w")
    file1.truncate(0)
    df = pd.read_csv(file.name)
    df = df.drop(df.columns[0],axis=1)
    solver = Eof(df.to_numpy())
    eig = solver.varianceFraction()
    t = np.array_str(eig)
    file1.write(t)
    file1.close()
    file2 = open("result.txt","r+")
    view_window = Tk()
    view_window.geometry('400x400')
    v = DocViewer(view_window)
    v.pack(side="top", expand=1, fill="both")
    v.display_file(file2.read())
    view_window.mainloop()
    file2.close()

def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6):
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d_old!=0 and d/d_old < 1 + tol: break
    return dot(Phi, R)

def cal_reof():
    file1 = open("result.txt","w")
    file1.truncate(0)
    df = pd.read_csv(file.name)
    df = df.drop(df.columns[0],axis=1)
    solver = Eof(df.to_numpy())
    eofs = solver.eofs()
    reofs = varimax(eofs)
    t = np.array_str(reofs)
    file1.write(t)
    file1.close()
    file2 = open("result.txt","r+")
    view_window = Tk()
    view_window.geometry('400x400')
    v = DocViewer(view_window)
    v.pack(side="top", expand=1, fill="both")
    v.display_file(file2.read())
    Lb1 = Label(view_window,text =("First two variance fraction = " + str(solver.varianceFraction(2))))
    Lb1.pack()
    view_window.mainloop()
    file2.close()

def accept_lag():
    Convert_To_Int = 1
    acc_lag = Tk()
    # specify size of window.
    acc_lag.geometry("250x170") 

    def printInput():
        inp = int(inputtxt.get(1.0, "end-1c"))
        cal_eeof(inp)
  
    # TextBox Creation
    inputtxt = Text(acc_lag,
                   height = 5,
                   width = 20)
  
    inputtxt.pack()
  
    # Button Creation
    printButton = Button(acc_lag,
                        text = "Print", 
                        command = printInput)
    printButton.pack()
    acc_lag.mainloop()



def cal_eeof(a_lag):
    file1 = open("result.txt","w")
    file1.truncate(0)
    df = pd.read_csv(file.name)
    df = df.drop(df.columns[0],axis=1)
    #accept lag
    solver = EEof(df.to_numpy(),2)
    efs = solver.eeofs()
    t = np.array_str(efs)
    file1.write(t)
    file1.close()
    file2 = open("result.txt","r+")
    view_window = Tk()
    view_window.geometry('400x400')
    v = DocViewer(view_window)
    v.pack(side="top", expand=1, fill="both")
    v.display_file(file2.read())
    Lb1 = Label(view_window,text =("First two variance fraction = " + solver.evarianceFraction(2)))
    Lb1.pack()
    view_window.mainloop()
    file2.close()

def cal_ceof():
    file1 = open("result.txt","w")
    file1.truncate(0)
    df = pd.read_csv(file.name)
    df = df.drop(df.columns[0],axis=1)
    input_H = np.empty(df.shape,df.dtypes)
    for i in range(df.shape[1]):
        input_H[:,i] = scipy.fftpack.hilbert(df.iloc[:,i])
    df1 = df + 1j*input_H
 #   solver = Eof(df1.to_numpy())
 #   eofs = solver.eofs()
    eofs, eig, pcs = svd(dot(df1,df1.T).astype(complex))
    eofs = dot(eofs,df1)
    t = np.array_str(eofs)
    file1.write(t)
    file1.close()
    file2 = open("result.txt","r+")
    view_window = Tk()
    view_window.geometry('400x400')
    v = DocViewer(view_window)
    v.pack(side="top", expand=1, fill="both")
    v.display_file(file2.read())
    Lb1 = Label(view_window,text =("First two Variance Fraction = " + str(eig[0]/eig.sum())+ "," + str(eig[1]/eig.sum())))
    Lb1.pack()
    view_window.mainloop()
    file2.close()

#Function of EOF by LU decomposition
def eof_lu_decomp():
    file1 = open("result.txt","w")
    file1.truncate(0)
    df = pd.read_csv(file.name)
    df = df.drop(df.columns[0],axis=1)
    eof, L, U = la.lu(df)
    eof = dot(eof,df)
    t = np.array_str(eof)
    file1.write(t)
    file1.close()
    file2 = open("result.txt","r+")
    view_windowe = Tk()
    view_windowe.geometry('400x400')
    v = DocViewer(view_windowe)
    v.pack(side="top", expand=1, fill="both")
    v.display_file(file2.read())
    view_windowe.mainloop()
    file2.close()


#Function of EOF by Eigendecomposition
def eof_eig_decomp():
    file1 = open("result.txt","w")
    file1.truncate(0)
    df = pd.read_csv(file.name)
    df = df.drop(df.columns[0],axis=1)
    eof, pca = la.eig(dot(df,df.T))
    eof = dot(eof,df)
    t = np.array_str(eof)
    file1.write(t)
    file1.close()
    file2 = open("result.txt","r+")
    view_windowd = Tk()
    view_windowd.geometry('400x400')
    v = DocViewer(view_windowd)
    v.pack(side="top", expand=1, fill="both")
    v.display_file(file2.read())
    view_windowd.mainloop()
    file2.close()


#Setup of GUI for landing page

LandingPage = Tk()
LandingPage.geometry('400x400')
LandingPage.title('GUI')

FAbtn = Button(LandingPage, text = "Open CSV File", command = open_csvfile)
FAbtn.pack(side = TOP, pady = 30)

Prbtn = Button(LandingPage, text ='Proceed', command = proceed_acn)
Prbtn.pack(side = BOTTOM, pady = 30)


mainloop()

