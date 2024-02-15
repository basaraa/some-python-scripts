from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import time
from tkinter import *
import os.path

def encrypting():
    if bool(E1.get()) & os.path.isfile(E1.get()):
        umiestnenie_kluca = "key.bin"
        key = get_random_bytes(32)
        key_file = open(umiestnenie_kluca, "wb")
        key_file.write(key)
        key_file.close()
        cipher = AES.new(key, AES.MODE_CBC)
        file_in = open(E1.get(),"rb")
        start = time.time()
        ciphered_data = cipher.encrypt(pad(file_in.read(), AES.block_size))
        file_out = open(E1.get()+'.encr', "wb")
        file_out.write(cipher.iv)
        file_out.write(ciphered_data)
        end = time.time()
        cas=str(end-start)
        message.config(text="Time to encrypt: " + cas + " seconds")
        print()
        file_out.close()
    else:
        message.config(text = "Bad file name")

def decrypting():
    if bool(E1.get()) & os.path.isfile(E1.get()) & bool(E1.get()[-5:]=='.encr'):
        key_file = open("key.bin", "rb")
        key = key_file.read()
        file_in = open(E1.get(), 'rb')
        iv = file_in.read(16)
        ciphered_data = file_in.read()
        file_in.close()
        cipher = AES.new(key, AES.MODE_CBC, iv=iv)
        start = time.time()
        original_data = unpad(cipher.decrypt(ciphered_data), AES.block_size)
        output_file=open(E1.get().strip(".encr"),"wb")
        output_file.write(original_data)
        end = time.time()
        cas = str(end - start)
        message.config(text="Time to decrypt: " + cas + " seconds")
        output_file.close()
    else:
        message.config(text = "Bad file name")

master = Tk()
master.title('Zadanie č.2: Implementácia aplikácie na šifrovanie súborov')
master.geometry("420x200")
Label(master, text="Zadaj nazov suboru",fg="black").place(x=150, y=0)
E1 = Entry(master, bd =5, width=50)
E1.place(x=70, y=40)
message=Label(master, text="",fg="red")
message.place(x=150, y=80)
Button(master, text='Encrypt', command=encrypting,fg="blue").place(x=100, y=120)
Button(master, text='Decrypt', command=decrypting,fg="blue").place(x=250, y=120)
mainloop()