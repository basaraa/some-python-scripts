from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Random import get_random_bytes
from Crypto.PublicKey import RSA
from Crypto.Protocol.KDF import scrypt
#from base64 import b64encode
#import json
import time
from tkinter import *
import os.path

def encrypting():
    if bool(E1.get()) & os.path.isfile(E1.get()):
        key_RSA = RSA.generate(2048)
        private_key = key_RSA.export_key()
        public_key = key_RSA.publickey()
        cipher_rsa = PKCS1_OAEP.new(public_key)
        public_key=public_key.export_key()
        RSA_store_key(public_key, private_key)
        salt = get_random_bytes(32)
        passw = os.urandom(16)
        key = scrypt(passw, salt, key_len=32, N=2 ** 17, r=8, p=1)
        rsa_encrypted_key = cipher_rsa.encrypt(key)
        cipher = AES.new(key, AES.MODE_GCM)
        cipher.update(rsa_encrypted_key+cipher.nonce)
        file_in = open(E1.get(),"rb")
        start = time.time()
        ciphered_data = cipher.encrypt(file_in.read())
        tag = cipher.digest()
        file_out = open(E1.get()+'.encr', "wb")
        file_out.write(rsa_encrypted_key)
        file_out.write(cipher.nonce)
        file_out.write(ciphered_data)
        file_out.write(tag)
        end = time.time()
        cas=str(end-start)
        message.config(text="Time to encrypt: " + cas + " seconds")
        file_out.close()
    else:
        message.config(text = "Bad file name")

def RSA_store_key(public_key,private_key):
    public_file = open("RSA_public_key.bin", "wb")
    private_file = open("RSA_private_key.bin", "wb")
    public_file.write(public_key)
    private_file.write(private_key)
    public_file.close()
    private_file.close()

def decrypting():
    if bool(E1.get()) & os.path.isfile(E1.get()) & bool(E1.get()[-5:]=='.encr'):
        file_in = open(E1.get(), 'rb')
        key2=file_in.read(256)
        nonce = file_in.read(16)
        file_in_size = os.path.getsize(E1.get())
        encr_size = file_in_size - 256 - 16 - 16
        data = file_in.read(encr_size)
        tag = file_in.read(16)
        private_key = RSA.import_key(open("RSA_private_key.bin").read())
        cipher_rsa = PKCS1_OAEP.new(private_key)
        decrypted_key = cipher_rsa.decrypt(key2)
        cipher = AES.new(decrypted_key, AES.MODE_GCM, nonce=nonce)
        cipher.update(key2+nonce)
        output_file = open(E1.get().strip(".encr"), "wb")
        start = time.time()
        try:
            original_data = cipher.decrypt_and_verify(data,tag)
        except ValueError as e:
            file_in.close()
            output_file.close()
            os.remove(E1.get().strip(".encr"))
            message.config(text="Error: Došlo k zmene šifrovaného súboru")
        else:
            output_file.write(original_data)
            end = time.time()
            cas = str(end - start)
            message.config(text="Time to decrypt: " + cas + " seconds")
            file_in.close()
            output_file.close()
    else:
        message.config(text = "Bad file name")


master = Tk()
master.title('Zadanie č.3')
master.geometry("360x200")
Label(master, text="Zadaj nazov suboru",fg="black").place(x=100, y=0)
E1 = Entry(master, bd =5, width=50)
E1.place(x=20, y=40)
message=Label(master, text="",fg="red")
message.place(x=50, y=80)
Button(master, text='Encrypt', command=encrypting,fg="blue").place(x=80, y=120)
Button(master, text='Decrypt', command=decrypting,fg="blue").place(x=230, y=120)
mainloop()
