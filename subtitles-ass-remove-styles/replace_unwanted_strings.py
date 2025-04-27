import os
from os import listdir
from os.path import isfile, join
from tkinter import *
from tkinter import messagebox

try:
  import pysubs2
except ImportError:
  os.system('python -m pip install pysubs2')
  import pysubs2

try:
  import re
except ImportError:
  os.system('python -m pip install re')
  import re
  
def clean_ass_file(file_path):
    try:
        subs = pysubs2.load(file_path)
        subs.info.pop("PlayResX", None)
        subs.info.pop("PlayResY", None)

        for line in subs:
            # Odstráni {…}
            line.text = re.sub(r"\{.*?\}", "", line.text)
            line.text = re.sub(r"\\N", "", line.text)
            line.text = re.sub(r"\\H", "", line.text)

        
        subs.events = [line for line in subs if (line.text.strip() != "" and not re.search(r"\bm\s[\d\s\-]+", line.text))]

        
        subs.save(file_path)
        print(f"Changed file: {file_path}")
    except Exception as e:
        print(f"Problem while processing {file_path}: {e}")

def process_directory(dir):
    ass_files = [f for f in listdir(dir) if isfile(join(dir, f)) & f.endswith('.ass')]


    if not ass_files:
        messagebox.showinfo("Info", "Not found .ass files in selected directory.")
        return

    for ass_file in ass_files:
        clean_ass_file(ass_file)

    messagebox.showinfo("Finished.", f"Checked {len(ass_files)} .ass files.")

def main():
    root = Tk()
    root.title("ASS Cleaner")

    Label(root, text="Path to folder:").grid(row=0, column=0, padx=5, pady=5)

    entry_path = Entry(root, width=50)
    entry_path.insert(0, "./")
    entry_path.grid(row=0, column=1, padx=5, pady=5)

    replace_btn = Button(root, text="Replace", width=20, command=lambda: process_directory(entry_path.get()))
    replace_btn.grid(row=1, column=0, columnspan=3, pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()