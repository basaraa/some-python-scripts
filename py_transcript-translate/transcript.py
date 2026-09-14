import os
from os import listdir
from os.path import isfile, join
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
from faster_whisper import WhisperModel

# Hugging Face Transformers for Meta NLLB-200 AI translation
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    HAS_NLLB = True
except ImportError:
    HAS_NLLB = False


def format_time(seconds: float) -> str:
    """Formats seconds into standard SRT timestamp (HH:MM:SS,mmm)."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"


# NLLB-200 language code mappings
NLLB_LANGUAGES = {
    "English (en) [Whisper Native]": ("en", "eng_Latn"),
    "Slovak (sk) [Meta NLLB AI]": ("sk", "slk_Latn"),
    "Czech (cs) [Meta NLLB AI]": ("cs", "ces_Latn"),
    "German (de) [Meta NLLB AI]": ("de", "deu_Latn"),
    "Spanish (es) [Meta NLLB AI]": ("es", "spa_Latn"),
    "French (fr) [Meta NLLB AI]": ("fr", "fra_Latn"),
    "Polish (pl) [Meta NLLB AI]": ("pl", "pol_Latn"),
    "Italian (it) [Meta NLLB AI]": ("it", "ita_Latn"),
    "Ukrainian (uk) [Meta NLLB AI]": ("uk", "ukr_Cyrl"),
    "Hungarian (hu) [Meta NLLB AI]": ("hu", "hun_Latn"),
    "Japanese (ja) [Meta NLLB AI]": ("ja", "jpn_Jpan"),
    "Chinese Simplified (zh) [Meta NLLB AI]": ("zh", "zho_Hans")
}


class NLLBTranslator:
    """Manages the local Meta NLLB-200 neural translation model."""
    def __init__(self, device: str):
        self.device = device
        self.model_name = "facebook/nllb-200-distilled-600M"
        self.tokenizer = None
        self.model = None

    def load(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, 
                torch_dtype=dtype
            ).to(self.device)

    def translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str, batch_size: int = 16) -> list[str]:
        """Translates text in batches on GPU/CPU for maximum performance."""
        self.tokenizer.src_lang = src_lang
        translated_texts = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=512
                )
            
            decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            translated_texts.extend(decoded)

        return translated_texts


class WhisperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper + Meta NLLB AI Subtitle Generator")
        self.root.geometry("680x660")
        self.root.resizable(True, True)

        self.is_running = False
        self.translator_ai = None
        self.setup_ui()

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use("clam")

        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. Directory Selection
        folder_frame = ttk.LabelFrame(main_frame, text="Video Directory", padding="10")
        folder_frame.pack(fill=tk.X, pady=(0, 10))

        self.folder_path_var = tk.StringVar(value=os.path.dirname(os.path.abspath(__file__)))
        folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_path_var)
        folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        btn_browse = ttk.Button(folder_frame, text="Browse...", command=self.browse_folder)
        btn_browse.pack(side=tk.RIGHT)

        # 2. Model & Hardware Settings
        model_frame = ttk.LabelFrame(main_frame, text="Whisper Model & Hardware", padding="10")
        model_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(model_frame, text="Whisper Model:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_var = tk.StringVar(value="large-v3-turbo")
        model_cb = ttk.Combobox(
            model_frame, 
            textvariable=self.model_var, 
            values=["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"], 
            state="readonly",
            width=15
        )
        model_cb.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        device_str = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
        ttk.Label(model_frame, text=f"Hardware Target: {device_str}").grid(row=0, column=2, sticky=tk.E, padx=15, pady=5)

        # 3. Translation & AI Engine Settings
        task_frame = ttk.LabelFrame(main_frame, text="Translation Settings (Meta NLLB AI)", padding="10")
        task_frame.pack(fill=tk.X, pady=(0, 10))

        self.translate_var = tk.BooleanVar(value=False)
        chk_translate = ttk.Checkbutton(
            task_frame, 
            text="Enable AI Translation", 
            variable=self.translate_var,
            command=self.toggle_translation_controls
        )
        chk_translate.grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

        ttk.Label(task_frame, text="Target Language:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.target_lang_var = tk.StringVar(value="Slovak (sk) [Meta NLLB AI]")
        self.lang_cb = ttk.Combobox(
            task_frame, 
            textvariable=self.target_lang_var, 
            values=list(NLLB_LANGUAGES.keys()), 
            state="disabled",
            width=36
        )
        self.lang_cb.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # 4. Console Log
        log_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=10, state="disabled", bg="#1e1e1e", fg="#ffffff", font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)

        # 5. Start Button
        self.btn_start = ttk.Button(main_frame, text="Start Transcription & Translation", command=self.start_processing)
        self.btn_start.pack(fill=tk.X, ipady=6)

    def browse_folder(self):
        folder = filedialog.askdirectory(initialdir=self.folder_path_var.get())
        if folder:
            self.folder_path_var.set(folder)

    def toggle_translation_controls(self):
        if self.translate_var.get():
            self.lang_cb.config(state="readonly")
        else:
            self.lang_cb.config(state="disabled")

    def log(self, message: str):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def start_processing(self):
        if self.is_running:
            return
        
        selected_key = self.target_lang_var.get()
        short_code, _ = NLLB_LANGUAGES[selected_key]

        if self.translate_var.get() and short_code != "en" and not HAS_NLLB:
            messagebox.showerror(
                "Missing AI Library",
                "Translating to other languages using Meta AI requires 'transformers' and 'sentencepiece'.\n\nInstall via:\npip install transformers sentencepiece"
            )
            return

        self.is_running = True
        self.btn_start.config(state="disabled")
        threading.Thread(target=self.process_videos, daemon=True).start()

    def process_videos(self):
        folder = self.folder_path_var.get()
        model_name = self.model_var.get()
        do_translate = self.translate_var.get()
        selected_key = self.target_lang_var.get()
        short_code, nllb_tgt_code = NLLB_LANGUAGES[selected_key]

        valid_extensions = ('.mkv', '.mp4', '.avi', '.mov', '.mp3', '.wav', '.m4a')
        video_files = [f for f in listdir(folder) if isfile(join(folder, f)) and f.lower().endswith(valid_extensions)]

        if not video_files:
            self.log("[-] No supported video/audio files found.")
            self.finish()
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        self.log(f"[*] Initializing Whisper model '{model_name}' on {device.upper()} ({compute_type})...")
        try:
            whisper_model = WhisperModel(model_name, device=device, compute_type=compute_type)
        except Exception as e:
            self.log(f"[!] Error loading Whisper: {e}")
            self.finish()
            return

        # Load Meta NLLB AI model if translation to non-English language is requested
        if do_translate and short_code != "en":
            self.log("[*] Loading Meta NLLB-200 AI translation model...")
            try:
                if self.translator_ai is None:
                    self.translator_ai = NLLBTranslator(device=device)
                    self.translator_ai.load()
                self.log("[+] Meta NLLB-200 loaded successfully.")
            except Exception as e:
                self.log(f"[!] Error loading NLLB model: {e}")
                self.finish()
                return

        self.log(f"[*] Found {len(video_files)} file(s) to process.\n" + "-" * 55)

        for filename in video_files:
            video_path = join(folder, filename)
            base_name = os.path.splitext(video_path)[0]
            srt_path = f"{base_name}.{short_code}.srt" if do_translate else f"{base_name}.srt"

            self.log(f"--> Processing: {filename}")

            try:
                # Case 1: Direct Whisper translation to English
                if do_translate and short_code == "en":
                    segments_gen, _ = whisper_model.transcribe(video_path, task="translate", vad_filter=True)
                    self.write_srt(list(segments_gen), srt_path)

                # Case 2: Meta NLLB-200 AI Translation (Batch GPU accelerated)
                elif do_translate and short_code != "en":
                    segments_gen, info = whisper_model.transcribe(video_path, vad_filter=True)
                    segments = list(segments_gen)
                    
                    self.log(f"    Detected language: {info.language} (Probability: {info.language_probability:.2f})")
                    self.log(f"    Running Meta NLLB AI batch translation into {short_code}...")

                    # Map Whisper's detected language code to NLLB source code
                    src_nllb = f"{info.language}_Latn"
                    # Try to find mapped code or default to English/auto fallback
                    for _, (code, nllb_code) in NLLB_LANGUAGES.items():
                        if code == info.language:
                            src_nllb = nllb_code
                            break

                    original_texts = [seg.text.strip() for seg in segments]
                    translated_texts = self.translator_ai.translate_batch(
                        original_texts, 
                        src_lang=src_nllb, 
                        tgt_lang=nllb_tgt_code
                    )

                    self.write_srt(segments, srt_path, translated_texts=translated_texts)

                # Case 3: Standard transcription (Original language)
                else:
                    segments_gen, _ = whisper_model.transcribe(video_path, vad_filter=True)
                    self.write_srt(list(segments_gen), srt_path)

                self.log(f"[+] Subtitles generated: {os.path.basename(srt_path)}\n")

            except Exception as e:
                self.log(f"[!] Error processing {filename}: {e}\n")

        self.log("[✓] All tasks finished successfully!")
        self.finish()

    def write_srt(self, segments, srt_path: str, translated_texts: list[str] = None):
        with open(srt_path, "w", encoding="utf-8") as srt_file:
            for i, segment in enumerate(segments, start=1):
                start = format_time(segment.start)
                end = format_time(segment.end)
                
                if translated_texts and i - 1 < len(translated_texts):
                    text = translated_texts[i - 1].strip()
                else:
                    text = segment.text.strip()

                srt_file.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    def finish(self):
        self.is_running = False
        self.btn_start.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = WhisperApp(root)
    root.mainloop()