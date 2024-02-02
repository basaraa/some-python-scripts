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

def convert_to_srt():
    dir=E1.get()
    files = [f for f in listdir(dir) if isfile(join(dir, f)) & f.endswith('.ass')] 
    pocet=0
    for fileName in files:
        with open(fileName, 'r+', encoding="utf8") as file:
            srt_lines = asstosrt.convert(file)
            with open(fileName.replace(".ass",".srt"), 'w', encoding="utf8",newline='') as file_srt:
                file_srt.write(srt_lines)
                pocet+=1
                file_srt.close()
            file.close()
    message.config(text="Converted "+str(pocet)+" files")        
master.geometry("300x160")        
Label(master, text="Zadaj priečinok:",fg="red").grid(row=0, sticky=W,column=0)
E1 = Entry(master, bd =5,width=40)
E1.grid(row=1,column=0)
E1.insert(0,"./")
message=Label(master, text="",fg="green")
message.grid(row=2,column=0)
Button(master, text='Convert to .srt', command=convert_to_srt,fg="blue").grid(row=3,column=0)
mainloop()            
            