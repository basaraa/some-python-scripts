import sys
import os
from os import listdir
from os.path import isfile, join
import pysubs2
try:
  from tkinter import *
except ImportError:
  os.system('python -m pip install tkinter')
  from tkinter import *
master = Tk()
def shift_time():
    dir=E1.get()
    time=float(E2.get())
    if var2.get() == 1:
        time=-abs(time)
        smer="dozadu"
    else:
        smer="dopredu"
    files = [f for f in listdir(dir) if isfile(join(dir, f)) & (f.endswith('.ass') | f.endswith('.srt'))] 
    pocet=0
    for meno in files:
        subs = pysubs2.load(dir+meno)
        subs.shift(s=time)
        subs.save(dir+meno)
        pocet+=1
    message.config(text="Shifted "+str(pocet)+" files by "+str(abs(time))+" seconds "+smer)
def shift_time_word():
    dir=E1.get()
    time=float(E2.get())
    slovo=E3.get()
    if var2.get() == 1:
        time=-abs(time)
        smer="forward"
    else:
        smer="backward"
    files = [f for f in listdir(dir) if isfile(join(dir, f)) & (f.endswith('.ass') | f.endswith('.srt'))] 
    pocet=0
    for meno in files:
        subs = pysubs2.load(dir+meno)
        casuj=0
        iter=0
        for word in subs:           
            if (word.text==slovo):
                casuj=1
                pocet+=1
            if (casuj==1):
                subs[iter].shift(s=time)            
            iter+=1 
        subs.save(dir+meno)                   
    message.config(text="Shifted "+str(pocet)+" files by "+str(abs(time))+" seconds "+smer)
master.geometry("500x260")        
Label(master, text="Enter dir:",fg="red").grid(row=0, sticky=W,column=0)
E1 = Entry(master, bd =5,width=40)
E1.grid(row=1,column=0)
Label(master, text="Enter shift time:",fg="red").grid(row=2, sticky=W,column=0)
E2 = Entry(master, bd =5,width=40)
E2.grid(row=3,column=0)
Label(master, text="Enter word from which you want shift(in case of 2nd option):",fg="red").grid(row=4, sticky=W,column=0)
E3 = Entry(master, bd =5,width=40)
E3.grid(row=5,column=0)
E1.insert(0,"./")
E2.insert(0,"0.0")
E3.insert(0,"From here")
var1 = IntVar()
Checkbutton(master, text="Forward", variable=var1).grid(row=6, sticky=W, column=0)
var2 = IntVar()
Checkbutton(master, text="Backward", variable=var2).grid(row=6,column=1)
message=Label(master, text="",fg="green")
message.grid(row=7,column=0)
Button(master, text='Shift subtitles', command=shift_time,fg="blue").grid(row=8,column=0)
Button(master, text='Shift from certain word', command=shift_time_word,fg="blue").grid(row=8,column=1)
mainloop()