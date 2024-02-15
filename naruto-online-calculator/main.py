from tkinter import *
import tkinter as tk
master = Tk()
def get_count_day_II():
    selected_indices = listbox.curselection()
    if not selected_indices:
        vysl=0
    else:
        vysl=selected_indices[0]
    return vysl
def one_day():
    suc = 0
    if var1.get() == 1:
        suc = suc + 19
    if var4.get() == 1:
        suc = suc + 10
    if var5.get() == 1:
        suc = suc + 130
    if var7.get() == 1:
        suc = suc + 40
    if var8.get() == 1:
        suc = suc + 50
    if var9.get() == 1:
        suc = suc + 100
    if var10.get() == 1:
        suc = suc + 20
    return suc

def coups_total():
    dni_typ=get_count_day_II()
    i=int(E1.get())
    j=int(E2.get())
    k=int(E3.get())
    II1=int(I1.get())
    II2=int(I2.get())
    II3=int(I3.get())
    ll=II1+II2+II3
    vysl=int (i+(k*one_day()))
    if k>29:
        if var2.get()==1:
            vysl+=int ((k/30)*250)
        if var8.get()==1:
            vysl+=int ((k/30)*300)
        if var9.get()==1:
            vysl+=int ((k/30)*600)
    if k>6:
        if var6.get()==1:
            vysl+=int ((k/7)*80)
        if var3.get()==1:
            vysl+=int ((k/7)*40)
        if var10.get()==1:
            vysl+=int ((k/7)*20)
        vysl+=int (int(((k+dni_typ)/7))*ll)    
    
    varx.set(str(vysl)+" coups")

def days_total():
    dni_typ=get_count_day_II()
    i = int(E1.get())
    j = int(E2.get())
    k = int(E3.get())
    II1=int(I1.get())
    II2=int(I2.get())
    II3=int(I3.get())
    ll=II1+II2+II3
    x=0
    sucet=i
    vysl=0
    helpvysl=0
    helpvysl2=0
    days_count=0
    while (sucet<j):
        days_count+=1
        vysl+=1
        helpvysl+=1
        helpvysl2+=1
        if (dni_typ<=6):
            dni_typ+=1
        if (dni_typ==7):
            days_count=0
            dni_typ+=1
            sucet+=ll
        if days_count/7==1:
            sucet+=ll
            days_count=0
        sucet+=one_day()
        if helpvysl > 29:
            if var2.get() == 1:
                sucet += 250
            if var8.get() == 1:
                sucet +=300
            if var9.get() == 1:
                sucet +=600
            helpvysl = 0
        if helpvysl2 > 6:
            if var6.get() == 1:
                sucet +=80
            if var3.get() == 1:
                sucet +=40
            helpvysl2=0
    varx.set(str(vysl)+" days")

Label(master, text="Non group coups:",fg="red").grid(row=0, sticky=W)
var1 = IntVar()
Checkbutton(master, text="Tree", variable=var1).grid(row=1, sticky=W, column=0)
var2 = IntVar()
Checkbutton(master, text="Full month attendance", variable=var2).grid(row=1,column=1)
var3 = IntVar()
Checkbutton(master, text="Weekend puzzle", variable=var3).grid(row=1,column=2)
var4 = IntVar()
Checkbutton(master, text="Daily 1h reward", variable=var4).grid(row=1,column=3)
#group
Label(master, text="Group coups:",fg="red").grid(row=2, sticky=W)
var5 = IntVar()
Checkbutton(master, text="Convoys & plunder", variable=var5).grid(row=3, sticky=W,column=0)
var6 = IntVar()
Checkbutton(master, text="Summon", variable=var6).grid(row=3, sticky=W,column=1)
var7 = IntVar()
Checkbutton(master, text="Group wheel", variable=var7).grid(row=3,column=2)
Label(master, text="P2W things:",fg="red").grid(row=8, sticky=W)
#p2w
var8 = IntVar()
Checkbutton(master, text="Monthly Card 1", variable=var8).grid(row=9, sticky=W,column=0)
var9 = IntVar()
Checkbutton(master, text="Monthly Card 2", variable=var9).grid(row=9,column=1)
var10 = IntVar()
Checkbutton(master, text="Jonin Medal", variable=var10).grid(row=9,column=2)
#illusion
Label(master, text="Infinity Illusion I rewards:",fg="red").grid(row=10, sticky=W,column=0)
I1 = Entry(master, bd =5)
I1.grid(row=10,column=1)
Label(master, text="Infinity Illusion II rewards:",fg="red").grid(row=11, sticky=W,column=0)
I2 = Entry(master, bd =5)
I2.grid(row=11,column=1)
Label(master, text="Infinity Illusion III rewards:",fg="red").grid(row=12, sticky=W,column=0)
I3 = Entry(master, bd =5)
I3.grid(row=12,column=1)
#select box
dni = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
        'Saturday', 'Sunday')
dni_var = tk.StringVar(value=dni)

listbox = tk.Listbox(
    master,
    listvariable=dni_var,
    height=4,
    selectmode='browse'
    )

listbox.grid(
    column=1,
    row=13,
    sticky='nwes'
)
Label(master, text="What day is today ?",fg="red").grid(row=13, sticky=W,column=0)
Label(master, text="",fg="red").grid(row=14, sticky=W,column=0)
Label(master, text="",fg="red").grid(row=15, sticky=W,column=0)
#rest
Label(master, text="How much coups you have:",fg="red").grid(row=17, sticky=W,column=0)
E1 = Entry(master, bd =5)
E1.grid(row=17,column=1)
Label(master, text="How much coups you want:",fg="red").grid(row=18, sticky=W,column=0)
E2 = Entry(master, bd =5)
E2.grid(row=18,column=1)

Label(master, text="How much days you will be saving:",fg="red").grid(row=19, sticky=W,column=0)
E3 = Entry(master, bd =5)
E3.grid(row=19,column=1)
E1.insert(0,"0")
E2.insert(0,"0")
E3.insert(0,"0")
I1.insert(0,"350")
I2.insert(0,"350")
I3.insert(0,"350")
Label(master, text="Answer:",fg="red").grid(row=20, sticky=W,column=0)
varx = StringVar()
label = Label(master, textvariable=varx, relief=RAISED ).grid(row=20,column=1)
Button(master, text='Quit', command=master.quit,fg="blue").grid(row=21, sticky=W,column=2)
#button
Button(master, text='Calculate days', command=days_total,fg="blue").grid(row=21,column=1)
Button(master, text='Calculate coups', command=coups_total,fg="blue").grid(row=21,column=0)
mainloop()
