import os
import subprocess
import json
import sys
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

def install_dependencies():
    dependencies = ['pysubs2', 'asstosrt']
    for lib in dependencies:
        try:
            __import__(lib)
        except ImportError:
            print(f"Inštalujem chýbajúcu knižnicu: {lib}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

install_dependencies()

import pysubs2
import asstosrt

# --- POMOCNÉ FUNKCIE ---
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov")

def is_video_file(filename):
    return filename.lower().endswith(VIDEO_EXTENSIONS)

# --- HLAVNÁ STRUKTÚRA APLIKÁCIE ---
class SubtitleToolbox:
    def __init__(self, root):
        self.root = root
        self.root.title("Subtitle Toolbox All-in-One")
        self.root.geometry("700x550")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        # Inicializácia tabov
        self.setup_extractor_tab()
        self.setup_shifter_tab()
        self.setup_converter_tab()
        self.setup_cleaner_tab()
        self.setup_ass_to_srt_tab()

    # 1. TAB: EXTRACTOR (Extrakcia titulkov z videa)
    def setup_extractor_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Extractor")

        self.ext_subtitle_data = {}
        self.ext_current_langs = []

        lbl = tk.Label(tab, text="Extrahovať titulky z video súborov (vyžaduje FFmpeg)", font=("Arial", 10, "bold"))
        lbl.pack(pady=5)

        btn_frame = tk.Frame(tab)
        btn_frame.pack(pady=5)

        tk.Button(btn_frame, text="Načítať priečinok", command=self.ext_load_folder).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Vybrať súbor", command=self.ext_load_file).pack(side=tk.LEFT, padx=5)

        self.ext_listbox = tk.Listbox(tab, selectmode=tk.MULTIPLE, width=50, height=10)
        self.ext_listbox.pack(pady=10, padx=10)

        tk.Button(tab, text="Extrahovať vybrané jazyky", command=self.ext_action, bg="lightblue").pack(pady=5)

    def ext_run_ffprobe(self, filepath):
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", filepath],
                capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            return json.loads(result.stdout)
        except Exception: return None

    def ext_load_folder(self):
        directory = os.getcwd()
        videos = [f for f in os.listdir(directory) if is_video_file(f)]
        self.ext_process_list(videos, directory)

    def ext_load_file(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mkv *.avi *.mov")])
        if path:
            self.ext_process_list([os.path.basename(path)], os.path.dirname(path))

    def ext_process_list(self, videos, directory):
        self.ext_subtitle_data = {}
        for vid in videos:
            full_path = os.path.join(directory, vid)
            data = self.ext_run_ffprobe(full_path)
            if not data: continue
            for s in data.get("streams", []):
                if s.get("codec_type") == "subtitle":
                    lang = s.get("tags", {}).get("language", "unknown")
                    if lang not in self.ext_subtitle_data: self.ext_subtitle_data[lang] = []
                    self.ext_subtitle_data[lang].append((full_path, s["index"]))
        
        self.ext_current_langs = sorted(self.ext_subtitle_data.keys())
        self.ext_listbox.delete(0, tk.END)
        for lang in self.ext_current_langs:
            self.ext_listbox.insert(tk.END, f"{lang} ({len(self.ext_subtitle_data[lang])} súborov)")

    def ext_action(self):
        selection = self.ext_listbox.curselection()
        if not selection: return
        for i in selection:
            lang = self.ext_current_langs[i]
            for fpath, idx in self.ext_subtitle_data[lang]:
                base, _ = os.path.splitext(fpath)
                out = f"{base}_{lang}_{idx}.srt"
                subprocess.run(["ffmpeg", "-i", fpath, "-map", f"0:{idx}", "-c:s", "srt", out, "-y"])
        messagebox.showinfo("Hotovo", "Titulky extrahované.")

    # 2. TAB: SHIFTER (Posun času)
    def setup_shifter_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Shifter")

        tk.Label(tab, text="Posunúť čas titulkov", font=("Arial", 10, "bold")).grid(row=0, columnspan=2, pady=5)
        
        tk.Label(tab, text="Priečinok:").grid(row=1, column=0, sticky="e")
        self.sh_dir = tk.Entry(tab, width=40); self.sh_dir.insert(0, "./"); self.sh_dir.grid(row=1, column=1, sticky="w")

        tk.Label(tab, text="Čas (sekundy):").grid(row=2, column=0, sticky="e")
        self.sh_val = tk.Entry(tab, width=10); self.sh_val.insert(0, "0.0"); self.sh_val.grid(row=2, column=1, sticky="w")

        self.sh_dir_var = tk.IntVar(value=1)
        tk.Radiobutton(tab, text="Dopredu (+)", variable=self.sh_dir_var, value=1).grid(row=3, column=1, sticky="w")
        tk.Radiobutton(tab, text="Dozadu (-)", variable=self.sh_dir_var, value=2).grid(row=4, column=1, sticky="w")

        tk.Label(tab, text="Mód:").grid(row=5, column=0, sticky="e")
        self.sh_mode = tk.IntVar(value=1)
        tk.Radiobutton(tab, text="Všetky riadky", variable=self.sh_mode, value=1).grid(row=5, column=1, sticky="w")
        tk.Radiobutton(tab, text="Od slova:", variable=self.sh_mode, value=2).grid(row=6, column=1, sticky="w")
        self.sh_word = tk.Entry(tab, width=20); self.sh_word.insert(0, "Slovo"); self.sh_word.grid(row=6, column=1, padx=(100,0))
        tk.Radiobutton(tab, text="Od OP (90s gap)", variable=self.sh_mode, value=3).grid(row=7, column=1, sticky="w")

        tk.Button(tab, text="Aplikovať posun", bg="orange", command=self.sh_apply).grid(row=8, columnspan=2, pady=10)

    def sh_apply(self):
        dpath = self.sh_dir.get()
        try:
            shift = float(self.sh_val.get())
            if self.sh_dir_var.get() == 2: shift = -abs(shift)
            files = [f for f in os.listdir(dpath) if f.endswith(('.ass', '.srt'))]
            count = 0
            for fn in files:
                subs = pysubs2.load(os.path.join(dpath, fn))
                triggered = False
                prev_end = 0
                for line in subs:
                    if self.sh_mode.get() == 1: line.shift(s=shift)
                    elif self.sh_mode.get() == 2:
                        if self.sh_word.get() in line.text: triggered = True
                        if triggered: line.shift(s=shift)
                    elif self.sh_mode.get() == 3:
                        if (int(line.start) - prev_end > 90000): triggered = True
                        if triggered: line.shift(s=shift)
                        prev_end = int(line.end)
                subs.save(os.path.join(dpath, fn))
                count += 1
            messagebox.showinfo("Hotovo", f"Upravených {count} súborov.")
        except Exception as e: messagebox.showerror("Chyba", str(e))

    # 3. TAB: SRT <-> TXT (Prekladací pomocník)
    def setup_converter_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="SRT <-> TXT")

        tk.Label(tab, text="Konvertuje titulky na čistý text a späť pre ľahký preklad", wraplength=400).pack(pady=10)
        tk.Button(tab, text="SRT → TXT (v aktuálnom priečinku)", command=self.conv_srt_to_txt).pack(pady=5)
        tk.Button(tab, text="TXT → SRT (v aktuálnom priečinku)", command=self.conv_txt_to_srt).pack(pady=5)

    def parse_blocks(self, content):
        blocks, current = [], []
        for line in content.splitlines():
            if line.strip() == "":
                if current: blocks.append(current); current = []
            else: current.append(line)
        if current: blocks.append(current)
        return blocks

    def conv_srt_to_txt(self):
        for file in os.listdir("."):
            if file.lower().endswith(".srt"):
                with open(file, "r", encoding="utf-8") as f: blocks = self.parse_blocks(f.read())
                txt_out, time_out = [], []
                for b in blocks:
                    if len(b) >= 3:
                        time_out.extend([b[0], b[1], ""])
                        txt_out.extend(b[2:] + [""])
                with open(file.replace(".srt", ".txt"), "w", encoding="utf-8") as f: f.write("\n".join(txt_out))
                with open(file.replace(".srt", "_timing.txt"), "w", encoding="utf-8") as f: f.write("\n".join(time_out))
        messagebox.showinfo("Hotovo", "Konverzia SRT na TXT dokončená.")

    def conv_txt_to_srt(self):
        for file in os.listdir("."):
            if file.lower().endswith(".txt") and not file.lower().endswith("_timing.txt"):
                base = file.replace(".txt", "")
                if os.path.exists(base + "_timing.txt"):
                    with open(file, "r", encoding="utf-8") as f: t_blocks = self.parse_blocks(f.read())
                    with open(base + "_timing.txt", "r", encoding="utf-8") as f: tm_blocks = self.parse_blocks(f.read())
                    srt_out = []
                    for i in range(min(len(t_blocks), len(tm_blocks))):
                        srt_out.extend([tm_blocks[i][0], tm_blocks[i][1]] + t_blocks[i] + [""])
                    with open(base + ".srt", "w", encoding="utf-8") as f: f.write("\n".join(srt_out))
        messagebox.showinfo("Hotovo", "Konverzia TXT na SRT dokončená.")

    # 4. TAB: CLEANER (Čistenie značiek)
    def setup_cleaner_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Cleaner")

        tk.Label(tab, text=r"Odstráni formátovanie {tags} a \N z ASS/SRT", font=("Arial", 10, "bold")).pack(pady=10)
        
        self.cl_path = tk.Entry(tab, width=50); self.cl_path.insert(0, "./"); self.cl_path.pack(pady=5)
        tk.Button(tab, text="Vyčistiť súbory", bg="lightgreen", command=self.cl_run).pack(pady=10)

    def cl_run(self):
        d = self.cl_path.get()
        files = [f for f in os.listdir(d) if f.endswith(('.ass', '.srt'))]
        for fn in files:
            p = os.path.join(d, fn)
            subs = pysubs2.load(p)
            for line in subs:
                line.text = re.sub(r"\{.*?\}", " ", line.text)
                line.text = line.text.replace("\\N", " ").replace("\\H", " ").replace("\\h", " ")
            subs.events = [l for l in subs if l.text.strip() != ""]
            subs.save(p)
        messagebox.showinfo("Hotovo", f"Vyčistených {len(files)} súborov.")

    # 5. TAB: ASS TO SRT (Konverzia formátu)
    def setup_ass_to_srt_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="ASS to SRT")

        tk.Label(tab, text="Hromadná konverzia .ass na .srt", font=("Arial", 10, "bold")).pack(pady=10)
        self.as_path = tk.Entry(tab, width=50); self.as_path.insert(0, "./"); self.as_path.pack(pady=5)
        
        # PREVOLENÉ: Recursive FALSE (0), Delete TRUE (1)
        self.as_rec = tk.IntVar(value=0)
        tk.Checkbutton(tab, text="Aj podadresáre", variable=self.as_rec).pack()
        self.as_del = tk.IntVar(value=1)
        tk.Checkbutton(tab, text="Vymazať pôvodné .ass", variable=self.as_del).pack()

        tk.Button(tab, text="Spustiť konverziu", bg="magenta", fg="white", command=self.as_run).pack(pady=10)

    def as_convert_recursive(self, directory):
        pocet = 0
        if not os.path.exists(directory): return 0
        
        for item in os.listdir(directory):
            p = os.path.join(directory, item)
            if os.path.isfile(p) and item.endswith('.ass'):
                success = False
                try:
                    # Načítame dáta a HNEĎ zavrieme súbor
                    with open(p, 'r', encoding="utf8") as f:
                        srt_data = asstosrt.convert(f)
                    
                    # Zapíšeme SRT
                    with open(p.replace(".ass", ".srt"), 'w', encoding="utf8", newline='') as out:
                        out.write(srt_data)
                    
                    success = True
                    pocet += 1
                except Exception as e:
                    print(f"Chyba pri konverzii {p}: {e}")
                
                # Mazanie prebehne až po uzavretí všetkých file handlerov
                if success and self.as_del.get():
                    try:
                        os.remove(p)
                    except Exception as e:
                        print(f"Chyba pri mazaní {p}: {e}")

            elif os.path.isdir(p) and self.as_rec.get():
                pocet += self.as_convert_recursive(p)
        return pocet

    def as_run(self):
        total = self.as_convert_recursive(self.as_path.get())
        messagebox.showinfo("Hotovo", f"Prekonvertovaných {total} súborov.")

if __name__ == "__main__":
    root = tk.Tk()
    app = SubtitleToolbox(root)
    root.mainloop()