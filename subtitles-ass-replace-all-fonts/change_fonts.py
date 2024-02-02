import sys
import os
from os import listdir
from os.path import isfile, join
try:
  import pysubs2
except ImportError:
  os.system('python -m pip install pysubs2')
  import pysubs2
try:
  from tkinter import *
except ImportError:
  os.system('python -m pip install tkinter')
  from tkinter import *

master = Tk()

def change_font():
    dir=E1.get()
    newFont=E2.get()
    files = [f for f in listdir(dir) if isfile(join(dir, f)) & f.endswith('.ass')] 
    pocet=0
    
    for fileName in files:
        with open(fileName, 'r+', encoding="utf8") as file:
            lineCount=0
            styles=0
            lines = file.readlines()
            for line in lines:
                if (line.find ("Format:")):
                    styles=1
                if (line.find ("Style:") != -1 and styles==1):
                    pos=line.find (",")+1
                    newline=line[0:pos] + newFont+"\n";
                    lines[lineCount]=newline
                elif (line.find("PlayResX:") != -1 or line.find("PlayResY:") != -1):
                    lines[lineCount]="";    
                elif (line.find("[Events]") != -1):
                    break                                               
                lineCount+=1 
            file.seek(0)        
            file.writelines(lines)
            file.close()
            pocet+=1
    message.config(text="Edited "+str(pocet)+" files")
   
master.geometry("400x260")        
Label(master, text="Enter dir:",fg="red").grid(row=0, sticky=W,column=0)
E1 = Entry(master, bd =5,width=40)
E1.grid(row=1,column=0)
Label(master, text="Enter new font:",fg="red").grid(row=2, sticky=W,column=0)
E2 = Entry(master, bd =5,width=40)
E2.grid(row=3,column=0)
E1.insert(0,"./")
E2.insert(0,"Roboto Medium,23,&H00FFFFFF,&H00000000,&H00000000,&HA2000000,0,0,0,0,100,100,0,0,1,1,1,2,30,30,5,1")
message=Label(master, text="",fg="green")
message.grid(row=4,column=0)
Button(master, text='Edit all fonts to the entered one', command=change_font,fg="blue").grid(row=5,column=0)
mainloop()