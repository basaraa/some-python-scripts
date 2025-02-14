import sys
import os
from os import listdir
from os.path import isfile, join

try:
  from tkinter import *
except ImportError:
  os.system('python -m pip install tkinter')
  from tkinter import *
  
try:
  import asstosrt
except ImportError:
  os.system('python -m pip install asstosrt')
  import asstosrt

master = Tk()

def recursive_convert_to_srt(dir,recursive=1,delOrigin=1): 
    pocet=0
    files = [f for f in listdir(dir) if isfile(join(dir, f)) & f.endswith('.ass')]
    for fileName in files:
        fileName=join(dir, fileName)
        with open(fileName, 'r+', encoding="utf8") as file:
            srt_lines = asstosrt.convert(file)
            with open(fileName.replace(".ass",".srt"), 'w', encoding="utf8",newline='') as file_srt:
                file_srt.write(srt_lines)
                pocet+=1
                file_srt.close()
            file.close()
            if (delOrigin==1):
                os.remove(fileName)
    if (recursive==1):
        dirs = [f for f in os.listdir(dir) if not isfile(join(dir, f))]
        if dirs:
            for xdir in dirs:
                pocet+=recursive_convert_to_srt(join(dir, xdir),recursive,delOrigin)
    return pocet    

    
def convert_to_srt():
    dir=E1.get()
    subdir=var1.get()
    delOrigin=var2.get()
    pocet=0
    pocet+=recursive_convert_to_srt(dir,subdir,delOrigin)
    message.config(text="Converted "+str(pocet)+" files")        
master.geometry("270x160")        
Label(master, text="Zadaj priečinok:",fg="red").grid(row=0, sticky=W,column=0)
E1 = Entry(master, bd =5,width=40)
E1.grid(row=1,column=0)
E1.insert(0,"./")
var1 = IntVar()
Checkbutton(master, text="Sub-directories", variable=var1).grid(row=2, sticky=W, column=0)
var2 = IntVar()
Checkbutton(master, text="Delete .ass", variable=var2).grid(row=3, sticky=W, column=0)
message=Label(master, text="",fg="green")
message.grid(row=4,column=0)
Button(master, text='Convert to .srt', command=convert_to_srt,fg="blue").grid(row=5,column=0)
mainloop()            
            