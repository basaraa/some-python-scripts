from modules.config import *
from modules.audio_utils import (
    run_cmd, get_vram_status, clear_vram, get_audio_duration,
    format_timestamp_srt, scan_voices_directory, get_all_available_voices,
    resolve_voice_for_subtitle, sort_treeview, extract_audio_from_video,
    separate_sources, get_or_create_reference_transcription,
    trim_audio_silence, convert_to_mp3_160k
)
from modules.diarization import (
    load_speaker_profiles_meta, save_speaker_profiles_meta, _safe_profile_path,
    create_speaker_profile_from_segments, assign_supertonic_voices_to_speakers,
    diarize_segments_direct, build_speakers_dict, generate_speaker_previews,
    get_valid_voice_path, merge_continuous_subtitles
)
from modules.subtitles import (
    parse_srt_file, split_text_into_sentences_list, create_project_structure,
    load_or_init_project, save_project, sync_subtitles_with_disk,
    find_matching_subtitle, export_working_file, extract_clean_text_from_srt,
    import_working_file, is_batch_video_up_to_date
)
from modules.mixer import (
    apply_studio_vocal_clarity, fit_audio_strictly_to_slot,
    assemble_dialogue_track_anticollision, create_adaptive_master_mix,
    mix_and_export_video
)
from modules.tts import DynamicDubbingEngine, play_any_voice_sample

def create_app():
    root = tk.Tk()
    root.title("Local AI Video Dubbing & Localization Studio")
    root.geometry("1440x960")
    root.minsize(960, 680)

    dub_state = {
        "video_path": "",
        "sub_path": "",
        "folders": {},
        "proj_file": "",
        "data": {},
        "tts_engine": None,
        "cancel_requested": False
    }

    narr_state = {
        "cancel_requested": False,
        "last_mp3": ""
    }

    narr_phrases = []

    log_queue = queue.Queue()

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Treeview", rowheight=23)
    style.configure("TButton", padding=3)
    style.configure("Accent.TButton", font=('Segoe UI', 9, 'bold'))

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=8, pady=(3, 2))

    tab_dubbing = ttk.Frame(notebook)
    tab_narration = ttk.Frame(notebook)

    notebook.add(tab_dubbing, text=" 🎬 1. Video Dubbing Studio (Supertonic, Edge & Clone) ")
    notebook.add(tab_narration, text=" 🎙️ 2. Text Narration & Multi-Voice Speech ")

    # Sort states for Treeview tables
    sub_tree_sort_state = {"col": None, "reverse": False}
    phrase_tree_sort_state = {"col": None, "reverse": False}

    # ==========================================================================
    # TAB 1: VIDEO DUBBING STUDIO
    # ==========================================================================
    top_frame = ttk.LabelFrame(tab_dubbing, text=" 1. Input Files & Configuration ", padding=5)
    top_frame.pack(fill="x", padx=10, pady=2)
    top_frame.columnconfigure(0, weight=0)
    top_frame.columnconfigure(1, weight=1)
    top_frame.columnconfigure(2, weight=0)

    ttk.Label(top_frame, text="Video/Audio:").grid(row=0, column=0, sticky="w", padx=(0, 4), pady=2)
    video_entry = ttk.Entry(top_frame)
    video_entry.grid(row=0, column=1, padx=(2, 4), pady=2, sticky="we")

    def auto_load_matching_subtitles(v_path):
        if not v_path or not os.path.exists(v_path):
            return
        tgt_l = lang_combo.get().strip() if 'lang_combo' in locals() or 'lang_combo' in globals() else "cs"
        matched_sub = find_matching_subtitle(v_path, target_lang=tgt_l)
        if matched_sub:
            sub_entry.delete(0, tk.END)
            sub_entry.insert(0, matched_sub)
            mode_var.set("EXISTING_SRT")
            dub_state["data"]["sub_path"] = matched_sub
            log(f"[Subtitles] Automatically loaded matching subtitles: {os.path.basename(matched_sub)}")

    def browse_video():
        fn = filedialog.askopenfilename(filetypes=[("Video/Audio", "*.mp4 *.mkv *.avi *.mov *.mp3 *.wav")])
        if fn:
            video_entry.delete(0, tk.END)
            video_entry.insert(0, fn)
            init_project_state(fn)
            auto_load_matching_subtitles(fn)

    def on_video_entry_change(event=None):
        v = video_entry.get().strip()
        if v and os.path.exists(v):
            init_project_state(v)
            if not sub_entry.get().strip():
                auto_load_matching_subtitles(v)

    video_entry.bind("<FocusOut>", on_video_entry_change)
    video_entry.bind("<Return>", on_video_entry_change)

    vid_btn_box = ttk.Frame(top_frame)
    vid_btn_box.grid(row=0, column=2, padx=(4, 0), pady=2, sticky="w")
    ttk.Button(vid_btn_box, text="Browse...", command=browse_video).pack(side="left")

    mode_var = tk.StringVar(value="EXISTING_SRT")
    ttk.Radiobutton(top_frame, text="Mode A: Existing subtitles in target language (.srt/.ass)", variable=mode_var, value="EXISTING_SRT").grid(row=1, column=1, sticky="w", pady=1)
    ttk.Radiobutton(top_frame, text="Mode B: No subtitles (auto-transcribe for translation)", variable=mode_var, value="NO_SRT").grid(row=2, column=1, sticky="w", pady=1)

    ttk.Label(top_frame, text="Subtitles (.srt):").grid(row=3, column=0, sticky="w", padx=(0, 4), pady=2)
    sub_entry = ttk.Entry(top_frame)
    sub_entry.grid(row=3, column=1, padx=(2, 4), pady=2, sticky="we")

    def browse_sub():
        fn = filedialog.askopenfilename(filetypes=[("Subtitles", "*.srt *.ass")])
        if fn:
            sub_entry.delete(0, tk.END)
            sub_entry.insert(0, fn)

    sub_btn_box = ttk.Frame(top_frame)
    sub_btn_box.grid(row=3, column=2, padx=(4, 0), pady=2, sticky="w")
    ttk.Button(sub_btn_box, text="Browse...", command=browse_sub).pack(side="left")
    whisper_sub_btn = ttk.Button(sub_btn_box, text="🎙 Transcribe with Whisper", command=lambda: threading.Thread(target=task_transcribe_whisper, daemon=True).start())
    whisper_sub_btn.pack(side="left", padx=(4, 0))

    settings_frame = ttk.Frame(top_frame)
    settings_frame.grid(row=4, column=1, sticky="w", pady=3)

    ttk.Label(settings_frame, text="Default Engine:").pack(side="left", padx=(0, 2))
    engine_var = tk.StringVar(value="Supertonic (Local)")
    engine_combo = ttk.Combobox(settings_frame, textvariable=engine_var, values=["Supertonic (Local)", "Edge-TTS (Online)", "Voice Cloning (XTTS-v2)"], width=18, state="readonly")
    engine_combo.pack(side="left", padx=4)

    sep_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(settings_frame, text="Separate Audio (HTDemucs)", variable=sep_var).pack(side="left", padx=5)

    ttk.Label(settings_frame, text="Language:").pack(side="left", padx=(8, 2))
    lang_combo = ttk.Combobox(settings_frame, values=["cs", "sk", "ja", "en", "de", "pl"], width=5, state="readonly")
    lang_combo.set("cs")
    lang_combo.pack(side="left")

    ttk.Label(settings_frame, text="Gender / F0:").pack(side="left", padx=(8, 2))
    pitch_profile_var = tk.StringVar(value="Asia (China, Japan, Korea)")
    pitch_combo = ttk.Combobox(settings_frame, textvariable=pitch_profile_var, values=["Asia (China, Japan, Korea)", "Western / Europe (US, EU)"], width=25, state="readonly")
    pitch_combo.pack(side="left", padx=2)

    def on_pitch_profile_change(event=None):
        speakers = dub_state["data"].get("speakers", {})
        if speakers:
            cur_p = pitch_profile_var.get()
            assign_supertonic_voices_to_speakers(speakers, profile=cur_p)
            save_project(dub_state["data"], dub_state["proj_file"])
            refresh_speaker_table()
            refresh_subtitle_table()
            log(f"[Profile] Character voices re-assigned according to profile: {cur_p}")

    pitch_combo.bind("<<ComboboxSelected>>", on_pitch_profile_change)

    ttk.Label(settings_frame, text="Max Characters:").pack(side="left", padx=(10, 2))
    max_spk_spin = ttk.Spinbox(settings_frame, from_=2, to=15, width=4)
    max_spk_spin.set(6)
    max_spk_spin.pack(side="left")

    force_reanalyze_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(settings_frame, text="Force re-analyze characters", variable=force_reanalyze_var).pack(side="left", padx=8)

    vol_frame = ttk.Frame(top_frame)
    vol_frame.grid(row=5, column=1, sticky="w", pady=(3, 2))

    ttk.Label(vol_frame, text="Dubbing Volume:").pack(side="left", padx=(0, 2))
    dlg_vol_var = tk.DoubleVar(value=0.90)
    dlg_scale = ttk.Scale(vol_frame, from_=0.50, to=1.40, variable=dlg_vol_var, orient="horizontal", length=75)
    dlg_scale.pack(side="left")
    dlg_lbl = ttk.Label(vol_frame, text="90%")
    dlg_lbl.pack(side="left", padx=(2, 10))
    dlg_scale.config(command=lambda v: dlg_lbl.config(text=f"{int(float(v)*100)}%"))

    ttk.Label(vol_frame, text="Background Volume:").pack(side="left", padx=(0, 2))
    bg_vol_var = tk.DoubleVar(value=1.00)
    bg_scale = ttk.Scale(vol_frame, from_=0.50, to=1.40, variable=bg_vol_var, orient="horizontal", length=75)
    bg_scale.pack(side="left")
    bg_lbl = ttk.Label(vol_frame, text="100%")
    bg_lbl.pack(side="left", padx=(2, 10))
    bg_scale.config(command=lambda v: bg_lbl.config(text=f"{int(float(v)*100)}%"))

    ttk.Label(vol_frame, text="Background Ducking:").pack(side="left", padx=(0, 2))
    duck_var = tk.DoubleVar(value=0.0)
    duck_scale = ttk.Scale(vol_frame, from_=0.0, to=0.50, variable=duck_var, orient="horizontal", length=75)
    duck_scale.pack(side="left")
    duck_lbl = ttk.Label(vol_frame, text="0% (Disabled - full background preserved)")
    duck_lbl.pack(side="left", padx=(2, 4))
    duck_scale.config(command=lambda v: duck_lbl.config(text=f"{int(float(v)*100)}%" if float(v) > 0.01 else "0% (Disabled - full background preserved)"))

    # 2. Rámec pre detegované postavy
    mid_frame = ttk.LabelFrame(tab_dubbing, text=" 2. Detected Characters (Default Voices - M1..M5 / F1..F5) ", padding=4)
    mid_frame.pack(fill="x", padx=10, pady=2)

    columns = ("speaker", "duration", "gender", "mode", "voice_ref")
    spk_tree = ttk.Treeview(mid_frame, columns=columns, show="headings", height=3, selectmode="extended")
    
    spk_tree.heading("speaker", text="Character")
    spk_tree.heading("duration", text="Speech Duration")
    spk_tree.heading("gender", text="Gender")
    spk_tree.heading("mode", text="Mode")
    spk_tree.heading("voice_ref", text="Assigned Voice / Sample")

    spk_tree.column("speaker", width=95, anchor="center")
    spk_tree.column("duration", width=85, anchor="center")
    spk_tree.column("gender", width=95, anchor="center")
    spk_tree.column("mode", width=140, anchor="w")
    spk_tree.column("voice_ref", width=380, anchor="w")
    spk_tree.pack(side="left", fill="both", expand=True)

    tree_scroll = ttk.Scrollbar(mid_frame, orient="vertical", command=spk_tree.yview)
    spk_tree.configure(yscrollcommand=tree_scroll.set)
    tree_scroll.pack(side="right", fill="y")

    tbl_btn_frame = ttk.Frame(tab_dubbing)
    tbl_btn_frame.pack(fill="x", padx=10, pady=2)

    def refresh_speaker_table():
        if 'update_spk_voices_dropdown' in locals() or 'update_spk_voices_dropdown' in globals():
            update_spk_voices_dropdown()
        for row in spk_tree.get_children():
            spk_tree.delete(row)
        speakers = dub_state["data"].get("speakers", {})
        eng = engine_var.get()

        for spk_id, d in sorted(speakers.items()):
            dur_str = f"{d.get('total_duration', 0):.1f} s"

            # Always display gender directly from the assigned character metadata
            if d.get("gender") == "female" or d.get("is_female") is True:
                gender_label = "Female 👩"
            elif d.get("gender") == "male" or d.get("is_female") is False:
                gender_label = "Male 👨"
            else:
                prof_val = pitch_profile_var.get()
                is_asia_prof = "ázia" in prof_val.lower() or "azia" in prof_val.lower() or "asia" in prof_val.lower()
                split_thresh = 165.0 if is_asia_prof else 155.0
                gender_label = "Female 👩" if d.get("avg_f0", 140.0) >= split_thresh else "Male 👨"

            if d.get("is_profile") and d.get("assigned_voice"):
                v_name = f"🎭 {d['assigned_voice']}"
                mode_str = "Voice Profile"
            elif d.get("custom_voice_path"):
                v_name = f"Custom WAV: {os.path.basename(d['custom_voice_path'])}"
                mode_str = "Custom Clone"
            elif "Supertonic" in eng:
                v_name = f"⚡ Supertonic [{d.get('supertonic_voice', 'M1')}]"
                mode_str = "Supertonic"
            elif "Edge" in eng:
                edge_v = "cs-CZ-VlastaNeural" if d.get("avg_f0", 140.0) >= 175.0 else "cs-CZ-AntoninNeural"
                v_name = f"🎙️ Edge: {edge_v}"
                mode_str = "Edge-TTS"
            else:
                v_name = os.path.basename(d.get("voice_path") or d.get("preview_file") or "clean_voice.wav")
                mode_str = "Voice Cloning (XTTS)"

            spk_tree.insert("", tk.END, values=(
                spk_id,
                dur_str,
                gender_label,
                mode_str,
                v_name
            ))

    def play_preview():
        sel = spk_tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Please select a character from the list.")
            return
        spk_id = spk_tree.item(sel[0])["values"][0]
        spk_data = dub_state["data"].get("speakers", {}).get(spk_id, {})
        eng = engine_var.get()

        try:
            if "Supertonic" in eng and not spk_data.get("custom_voice_path"):
                v_name = spk_data.get("supertonic_voice", "M1")
                test_wav = os.path.join(VOICES_DIR, f"preview_st_{v_name}.wav")
                if not os.path.exists(test_wav):
                    engine = DynamicDubbingEngine(target_lang=lang_combo.get().strip(), engine_type="supertonic")
                    engine.ensure_supertonic_loaded(log_cb=log)
                    style = engine.supertonic_model.get_voice_style(voice_name=v_name)
                    sample_txt = "This is a sample voice for dubbing synthesized with Supertonic."
                    wav, _ = engine.supertonic_model.synthesize(text=sample_txt, voice_style=style, lang=lang_combo.get().strip(), total_steps=8, speed=1.0)
                    engine.supertonic_model.save_audio(wav, test_wav)
                data, sr = sf.read(test_wav)
                sd.stop()
                sd.play(data, sr)
                return

            v_path = get_valid_voice_path(spk_id, dub_state)
            if not v_path or not os.path.exists(v_path):
                messagebox.showwarning("Notice", f"Voice for {spk_id} was not found.")
                return
            data, sr = sf.read(v_path)
            sd.stop()
            sd.play(data, sr)
        except Exception as e:
            messagebox.showwarning("Notice", f"Playback error: {str(e)}")

    def cycle_supertonic_voice():
        sel = spk_tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Please select a character from the table.")
            return
        spk_id = spk_tree.item(sel[0])["values"][0]
        spk_data = dub_state["data"]["speakers"][spk_id]
        is_fem = spk_data.get("avg_f0", 140.0) >= 165.0
        pool = SUPERTONIC_FEMALE_VOICES if is_fem else SUPERTONIC_MALE_VOICES
        curr_v = spk_data.get("supertonic_voice", pool[0])
        
        curr_idx = pool.index(curr_v) if curr_v in pool else -1
        next_v = pool[(curr_idx + 1) % len(pool)]
        spk_data["supertonic_voice"] = next_v
        
        # Aktualizujeme aj repliky danej postavy, ktoré nemajú explicitný prepis
        for sub in dub_state["data"].get("subtitles", []):
            if sub.get("speaker") == spk_id:
                sub["assigned_voice"] = f"⚡ Supertonic [{next_v}]"

        save_project(dub_state["data"], dub_state["proj_file"])
        refresh_speaker_table()
        refresh_subtitle_table()
        log(f"[Voice] {spk_id} switched to Supertonic [{next_v}]")

    def assign_custom_voice():
        sel = spk_tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Please select a character from the list.")
            return
        spk_id = spk_tree.item(sel[0])["values"][0]
        fn = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.m4a")])
        if fn:
            target_wav = os.path.join(dub_state["folders"]["speakers"], f"{spk_id}_custom_{os.path.basename(fn)}.wav")
            cmd = ["ffmpeg", "-y", "-i", fn, "-ar", str(SAMPLE_RATE), "-ac", "1", target_wav]
            run_cmd(cmd)
            custom_voice_name = os.path.splitext(os.path.basename(fn))[0]
            
            # Ak pri zdrojovom súbore existuje .txt, skopírujeme ho k cieľovému
            src_txt = os.path.splitext(fn)[0] + ".txt"
            if os.path.exists(src_txt):
                shutil.copy2(src_txt, os.path.splitext(target_wav)[0] + ".txt")

            dub_state["data"]["speakers"][spk_id]["custom_voice_path"] = target_wav
            dub_state["data"]["speakers"][spk_id]["mode"] = f"Vlastný WAV ({custom_voice_name})"
            
            # Nastavíme aj pre repliky tejto postavy
            for sub in dub_state["data"].get("subtitles", []):
                if sub.get("speaker") == spk_id:
                    sub["assigned_voice"] = f"🧬 [Klon] {custom_voice_name}"

            save_project(dub_state["data"], dub_state["proj_file"])
            refresh_speaker_table()
            refresh_subtitle_table()

    ttk.Label(tbl_btn_frame, text="Character Voice:").pack(side="left", padx=(2, 2))
    spk_voice_combo = ttk.Combobox(tbl_btn_frame, width=34, state="readonly")
    spk_voice_combo.pack(side="left", padx=2)
    spk_voice_combo.bind("<Return>", lambda e: play_speaker_voice_preview())

    def update_spk_voices_dropdown():
        tgt_l = lang_combo.get().strip() if 'lang_combo' in locals() or 'lang_combo' in globals() else "cs"
        cat, vmap = get_all_available_voices(tgt_l)
        spk_voice_combo['values'] = cat
        if cat and not spk_voice_combo.get():
            spk_voice_combo.set(cat[0])

    # Populate immediately on creation!
    update_spk_voices_dropdown()

    def on_speaker_select(event=None):
        if not spk_voice_combo['values']:
            update_spk_voices_dropdown()
        sel = spk_tree.selection()
        if not sel:
            return
        spk_id = spk_tree.item(sel[0])["values"][0]
        spk_data = dub_state["data"].get("speakers", {}).get(spk_id, {})
        cur_v = spk_data.get("assigned_voice") or spk_data.get("supertonic_voice")
        if cur_v:
            cur_v_lower = str(cur_v).lower()
            for val in spk_voice_combo["values"]:
                if cur_v_lower in val.lower() or val.lower() in cur_v_lower:
                    spk_voice_combo.set(val)
                    break

    spk_tree.bind("<<TreeviewSelect>>", on_speaker_select)

    def assign_selected_voice_to_speaker():
        sel = spk_tree.selection()
        if not sel:
            messagebox.showinfo("Selection", "Please select a character from the table first.")
            return
        chosen = spk_voice_combo.get()
        if not chosen:
            messagebox.showinfo("Voice", "Please select a voice from the dropdown list.")
            return

        cat, vmap = get_all_available_voices(lang_combo.get().strip())
        v_info = vmap.get(chosen, {})

        for item_id in sel:
            spk_id = spk_tree.item(item_id)["values"][0]
            spk_data = dub_state["data"]["speakers"].setdefault(spk_id, {})

            if v_info.get("type") == "supertonic":
                v_id = v_info["voice_id"]
                spk_data["supertonic_voice"] = v_id
                spk_data["custom_voice_path"] = ""
                spk_data["mode"] = "Supertonic"
                assigned_label = f"⚡ Supertonic [{v_id}]"
                # If user manually assigns an F voice, mark as female; if M voice, mark as male
                if str(v_id).upper().startswith("F"):
                    spk_data["gender"] = "female"
                    spk_data["is_female"] = True
                elif str(v_id).upper().startswith("M"):
                    spk_data["gender"] = "male"
                    spk_data["is_female"] = False
            elif v_info.get("type") == "edge":
                v_id = v_info["voice_id"]
                spk_data["edge_voice"] = v_id
                spk_data["custom_voice_path"] = ""
                spk_data["mode"] = f"Edge-TTS ({v_id})"
                assigned_label = f"🎙️ [Edge] {v_id}"
            elif v_info.get("type") == "clone":
                v_path = v_info.get("path", "")
                spk_data["custom_voice_path"] = v_path
                spk_data["mode"] = f"Vlastný WAV ({v_info.get('voice_id')})"
                assigned_label = f"🧬 [Klon] {v_info.get('voice_id')}"
            else:
                assigned_label = chosen

            spk_data["assigned_voice"] = assigned_label

            # Aktualizujeme aj všetky doterajšie repliky danej postavy
            for sub in dub_state["data"].get("subtitles", []):
                if sub.get("speaker") == spk_id:
                    sub["assigned_voice"] = assigned_label

        save_project(dub_state["data"], dub_state["proj_file"])
        refresh_speaker_table()
        refresh_subtitle_table()
        log(f"[Character] Voice successfully assigned to character: {chosen}")

    def play_speaker_voice_preview():
        chosen = spk_voice_combo.get()
        if not chosen:
            sel = spk_tree.selection()
            if sel:
                spk_id = spk_tree.item(sel[0])["values"][0]
                spk_data = dub_state["data"].get("speakers", {}).get(spk_id, {})
                chosen = spk_data.get("assigned_voice")
        if not chosen:
            messagebox.showinfo("Info", "Select a voice in the dropdown to listen to a preview.")
            return
        ok, msg = play_any_voice_sample(chosen, target_lang=lang_combo.get().strip(), log_cb=log)
        if not ok:
            messagebox.showwarning("Notice", msg)

    def play_speaker_original_clip():
        sel = spk_tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Please select a character from the table first.")
            return
        spk_id = spk_tree.item(sel[0])["values"][0]
        spk_data = dub_state["data"].get("speakers", {}).get(spk_id, {})
        clip_path = spk_data.get("preview_file")
        if not (clip_path and os.path.exists(clip_path)):
            spk_dir = dub_state.get("folders", {}).get("speakers")
            if spk_dir:
                clip_path = os.path.join(spk_dir, f"{spk_id}_clean_voice.wav")
        if not (clip_path and os.path.exists(clip_path)):
            messagebox.showwarning(
                "Notice",
                f"Original voice clip for {spk_id} was not found.\n\n"
                "Please run '1. Analyze & Diarize' first to separate audio and extract character clips."
            )
            return
        try:
            data, sr = sf.read(clip_path)
            sd.stop()
            sd.play(data, sr)
            log(f"[Playback] Playing original clip for {spk_id}: {os.path.basename(clip_path)}")
        except Exception as e:
            messagebox.showwarning("Notice", f"Playback error: {e}")

    def merge_selected_speakers():
        sel = spk_tree.selection()
        if len(sel) < 2:
            messagebox.showinfo(
                "Merge Characters",
                "Please select at least 2 characters in the table to merge (hold Ctrl or Shift while clicking)."
            )
            return

        spk_ids = [spk_tree.item(item_id)["values"][0] for item_id in sel]
        target_spk = spk_ids[0]
        source_spks = spk_ids[1:]

        speakers = dub_state["data"].setdefault("speakers", {})
        if target_spk not in speakers:
            messagebox.showerror("Error", f"Target character '{target_spk}' not found in project.")
            return

        confirm = messagebox.askyesno(
            "Confirm Character Merge",
            f"Are you sure you want to merge character(s) {', '.join(source_spks)} into '{target_spk}'?\n\n"
            f"All dialogue lines belonging to {', '.join(source_spks)} will be reassigned to '{target_spk}'."
        )
        if not confirm:
            return

        target_data = speakers[target_spk]
        target_voice = target_data.get("assigned_voice") or f"⚡ Supertonic [{target_data.get('supertonic_voice', 'M1')}]"

        # 1. Merge segments and durations
        for src_id in source_spks:
            if src_id in speakers:
                src_data = speakers[src_id]
                target_data["segments"].extend(src_data.get("segments", []))
                target_data["total_duration"] += src_data.get("total_duration", 0.0)
                del speakers[src_id]

        # 2. Recalculate robust avg_f0 for target character across combined segments
        all_pitches = [s["f0"] for s in target_data["segments"] if "f0" in s and 65.0 < s["f0"] < 350.0]
        if len(all_pitches) >= 2:
            p_arr = np.array(all_pitches, dtype=np.float32)
            p20, p80 = np.percentile(p_arr, 20), np.percentile(p_arr, 80)
            robust = p_arr[(p_arr >= p20) & (p_arr <= p80)]
            target_data["avg_f0"] = float(np.median(robust if len(robust) > 0 else p_arr))
        elif all_pitches:
            target_data["avg_f0"] = float(all_pitches[0])

        # 3. Update all subtitle lines belonging to the merged characters
        subtitles = dub_state["data"].get("subtitles", [])
        reassigned_count = 0
        for sub in subtitles:
            if sub.get("speaker") in source_spks:
                sub["speaker"] = target_spk
                sub["assigned_voice"] = target_voice
                reassigned_count += 1

        save_project(dub_state["data"], dub_state["proj_file"])
        refresh_speaker_table()
        refresh_subtitle_table()

        log(f"[Characters] Successfully merged {', '.join(source_spks)} into {target_spk} ({reassigned_count} dialogue lines reassigned).")
        msg_txt = f"Successfully merged {', '.join(source_spks)} into {target_spk}!\n\nUpdated {reassigned_count} dialogue lines to {target_spk}."
        messagebox.showinfo("Characters Merged", msg_txt)

    ttk.Button(tbl_btn_frame, text="▶ Test", width=7, command=play_speaker_voice_preview).pack(side="left", padx=2)
    ttk.Button(tbl_btn_frame, text="✔ Assign", style="Accent.TButton", command=assign_selected_voice_to_speaker).pack(side="left", padx=2)
    ttk.Separator(tbl_btn_frame, orient="vertical").pack(side="left", fill="y", padx=8, pady=2)
    ttk.Button(tbl_btn_frame, text="▶ Play Original Clip (Video)", command=play_speaker_original_clip).pack(side="left", padx=3)
    ttk.Button(tbl_btn_frame, text="🔗 Merge Characters (Ctrl/Shift)", command=merge_selected_speakers).pack(side="left", padx=3)

    # Compact Known Voice Profiles bar (Auto-Detect across videos)
    prof_frame = ttk.LabelFrame(tab_dubbing, text=" Known Voice Profiles (Auto-Detect) ", padding=4)
    prof_frame.pack(fill="x", padx=10, pady=2)

    prof_left = ttk.Frame(prof_frame)
    prof_left.pack(side="left", fill="both", expand=True)

    prof_tree = ttk.Treeview(prof_left, columns=("active", "name", "voice"), show="headings", height=2, selectmode="browse")
    prof_tree.heading("active", text="On")
    prof_tree.heading("name", text="Profile Name")
    prof_tree.heading("voice", text="Assigned Voice")
    prof_tree.column("active", width=40, anchor="center", stretch=False)
    prof_tree.column("name", width=220, anchor="w")
    prof_tree.column("voice", width=260, anchor="w")
    prof_tree.pack(side="left", fill="both", expand=True)

    prof_scroll = ttk.Scrollbar(prof_left, orient="vertical", command=prof_tree.yview)
    prof_tree.configure(yscrollcommand=prof_scroll.set)
    prof_scroll.pack(side="right", fill="y")

    def refresh_profiles_table():
        for r in prof_tree.get_children():
            prof_tree.delete(r)
        meta = load_speaker_profiles_meta()
        for pname, info in sorted(meta.items()):
            mark = "✓" if info.get("enabled", True) else "☐"
            prof_tree.insert("", tk.END, iid=pname, values=(mark, pname, info.get("voice", "⚡ Supertonic [M1]")))

    def toggle_selected_profile(event=None):
        if event is not None:
            item = prof_tree.identify_row(event.y)
            col = prof_tree.identify_column(event.x)
            if not item:
                return
            if event.type == tk.EventType.ButtonPress and col != "#1":
                return
        else:
            sel = prof_tree.selection()
            if not sel:
                messagebox.showinfo("Profiles", "Select a profile to toggle On/Off.")
                return
            item = sel[0]
        meta = load_speaker_profiles_meta()
        if item in meta:
            meta[item]["enabled"] = not meta[item].get("enabled", True)
            save_speaker_profiles_meta(meta)
            refresh_profiles_table()

    prof_tree.bind("<Button-1>", toggle_selected_profile)
    prof_tree.bind("<Double-1>", lambda e: toggle_selected_profile())

    def delete_selected_profile():
        sel = prof_tree.selection()
        if not sel:
            messagebox.showinfo("Profiles", "Select a profile to delete.")
            return
        pname = sel[0]
        if not messagebox.askyesno("Delete Profile", f"Delete voice profile '{pname}'?"):
            return
        meta = load_speaker_profiles_meta()
        fname = meta.get(pname, {}).get("file", f"{pname}.wav")
        for candidate in (fname, f"{pname}.npy"):
            try:
                p = _safe_profile_path(candidate)
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        if pname in meta:
            del meta[pname]
            save_speaker_profiles_meta(meta)
        refresh_profiles_table()
        log(f"[Profiles] Deleted profile '{pname}'.")

    def get_selected_profile_threshold():
        val = prof_sens_var.get()
        if "Strict" in val:
            return 0.58
        if "Loose" in val:
            return 0.42
        return 0.50

    def reapply_profiles_to_current_project():
        segments = dub_state["data"].get("subtitles", [])
        if not segments:
            messagebox.showinfo("Profiles", "Load or analyze a video/subtitle project first.")
            return
        speech_audio = dub_state["data"].get("vocals_audio") or dub_state["data"].get("extracted_audio")
        if not speech_audio or not os.path.exists(speech_audio):
            messagebox.showwarning("Profiles", "Audio track not found yet. Please run 'Analyze & Detect Characters' first.")
            return

        thresh = get_selected_profile_threshold()
        max_spks = int(max_spk_spin.get())
        cur_prof = pitch_profile_var.get()

        def worker():
            try:
                log(f"[Profiles] Re-applying active voice profiles (threshold={thresh:.2f})...")
                segs_copy = [dict(s) for s in segments]
                new_segs, new_spks = diarize_segments_direct(
                    speech_audio, segs_copy, max_speakers=max_spks,
                    log_cb=log, profile=cur_prof, similarity_threshold=thresh
                )
                generate_speaker_previews(speech_audio, new_spks, dub_state["folders"]["speakers"])
                for seg in new_segs:
                    spk = seg.get("speaker", "Speaker_01")
                    spk_data = new_spks.get(spk, {})
                    if spk_data.get("is_profile") and spk_data.get("assigned_voice"):
                        seg["assigned_voice"] = spk_data["assigned_voice"]
                    else:
                        st_v = spk_data.get("supertonic_voice", "M1")
                        seg["assigned_voice"] = f"⚡ Supertonic [{st_v}]"

                dub_state["data"]["subtitles"] = new_segs
                dub_state["data"]["speakers"] = new_spks
                dub_state["data"]["steps_completed"]["diarization"] = True
                save_project(dub_state["data"], dub_state["proj_file"])
                root.after(0, refresh_speaker_table)
                root.after(0, refresh_subtitle_table)
                log("[Profiles] Voice profiles applied to current project.")
            except Exception as exc:
                log(f"[Profiles Error] {exc}")
                root.after(0, lambda: messagebox.showerror("Profile Error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    prof_ctrl = ttk.Frame(prof_frame)
    prof_ctrl.pack(side="right", fill="y", padx=(6, 2))

    prof_r1 = ttk.Frame(prof_ctrl)
    prof_r1.pack(fill="x", pady=1)
    ttk.Label(prof_r1, text="Match:").pack(side="left", padx=(0, 3))
    prof_sens_var = tk.StringVar(value="Balanced (0.50)")
    prof_sens_combo = ttk.Combobox(prof_r1, textvariable=prof_sens_var, values=["Strict (0.58)", "Balanced (0.50)", "Loose (0.42)"], width=15, state="readonly")
    prof_sens_combo.pack(side="left", padx=2)
    ttk.Button(prof_r1, text="🔄 Apply Profiles Now", style="Accent.TButton", command=reapply_profiles_to_current_project).pack(side="left", padx=3)

    prof_r2 = ttk.Frame(prof_ctrl)
    prof_r2.pack(fill="x", pady=1)
    ttk.Button(prof_r2, text="Toggle On/Off", command=lambda: toggle_selected_profile(None)).pack(side="left", padx=2)
    ttk.Button(prof_r2, text="🗑 Delete", command=delete_selected_profile).pack(side="left", padx=2)
    ttk.Button(prof_r2, text="📁 Folder", command=lambda: os.startfile(SPEAKER_PROFILES_DIR) if os.name == 'nt' else subprocess.run(['xdg-open', SPEAKER_PROFILES_DIR])).pack(side="left", padx=2)

    refresh_profiles_table()

    # 3. Nový rámec: Zoznam replík / titulkov a priradenie hlasov (Tab 1)
    sub_frame = ttk.LabelFrame(tab_dubbing, text=" 3. Subtitles & Voice Assignment (Ctrl/Shift selection, sortable columns) ", padding=6)
    sub_frame.pack(fill="both", expand=True, padx=10, pady=3)

    sub_toolbar = ttk.Frame(sub_frame)
    sub_toolbar.pack(fill="x", pady=(0, 4))

    ttk.Label(sub_toolbar, text="Line Voice:").pack(side="left", padx=(2, 2))
    sub_voice_combo = ttk.Combobox(sub_toolbar, width=34, state="readonly")
    sub_voice_combo.pack(side="left", padx=2)
    sub_voice_combo.bind("<Return>", lambda e: play_sub_voice_preview())

    def update_tab1_voices_dropdown():
        cat, vmap = get_all_available_voices(lang_combo.get().strip())
        sub_voice_combo['values'] = cat
        if cat and not sub_voice_combo.get():
            sub_voice_combo.set(cat[0])

    def assign_voice_to_selected_subtitles():
        sel = sub_tree.selection()
        if not sel:
            messagebox.showinfo("Selection", "Select at least one subtitle line (Ctrl/Shift selection supported).")
            return
        chosen = sub_voice_combo.get()
        if not chosen:
            messagebox.showinfo("Voice", "Please choose a voice from the dropdown list.")
            return

        sub_dict = {str(s["id"]): s for s in dub_state["data"].get("subtitles", [])}
        for item_id in sel:
            row_id_val = str(sub_tree.set(item_id, "id"))
            if row_id_val in sub_dict:
                sub_dict[row_id_val]["assigned_voice"] = chosen
            sub_tree.set(item_id, "voice", chosen)

        save_project(dub_state["data"], dub_state["proj_file"])
        log(f"[Subtitles] Voice '{chosen}' assigned to {len(sel)} subtitle lines.")

    def create_profile_from_selected_subtitles():
        sel = sub_tree.selection()
        if not sel:
            messagebox.showinfo("Selection", "Select 2-10 subtitle lines (Ctrl/Shift) of the same character first.")
            return

        chosen_voice = sub_voice_combo.get().strip()
        sub_dict = {str(s["id"]): s for s in dub_state["data"].get("subtitles", [])}
        selected_segs = []
        for item_id in sel:
            row_id_val = str(sub_tree.set(item_id, "id"))
            if row_id_val in sub_dict:
                selected_segs.append(sub_dict[row_id_val])

        if not selected_segs:
            return
        if not chosen_voice:
            chosen_voice = selected_segs[0].get("assigned_voice") or "⚡ Supertonic [M1]"

        v_path = dub_state["data"].get("video_path") or video_entry.get().strip()
        speech_audio = dub_state["data"].get("vocals_audio") or dub_state["data"].get("extracted_audio")
        use_sep = sep_var.get()

        def worker():
            try:
                pname, dur_sec = create_speaker_profile_from_segments(
                    v_path, selected_segs, chosen_voice,
                    speech_audio_path=speech_audio, use_separation=use_sep, log_cb=log
                )
                for s in selected_segs:
                    s["speaker"] = pname
                    s["assigned_voice"] = chosen_voice
                save_project(dub_state["data"], dub_state["proj_file"])
                root.after(0, refresh_profiles_table)
                root.after(0, refresh_subtitle_table)
                root.after(0, lambda: messagebox.showinfo(
                    "Voice Profile Created",
                    f"Profile '{pname}' ({dur_sec:.1f}s) saved to speaker_profiles/ -> {chosen_voice}\n\n"
                    "Click '🔄 Apply Profiles Now' to auto-detect this voice in the current video, "
                    "or run Analyze on any other episode!"
                ))
            except Exception as exc:
                log(f"[Profiles Error] {exc}")
                root.after(0, lambda: messagebox.showerror("Profile Error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def reset_selected_subtitles_voice():
        sel = sub_tree.selection()
        if not sel:
            sel = sub_tree.get_children()

        speakers = dub_state["data"].get("speakers", {})
        sub_dict = {str(s["id"]): s for s in dub_state["data"].get("subtitles", [])}

        for item_id in sel:
            row_id_val = str(sub_tree.set(item_id, "id"))
            if row_id_val in sub_dict:
                sub = sub_dict[row_id_val]
                spk = sub.get("speaker", "Speaker_01")
                spk_data = speakers.get(spk, {})
                if spk_data.get("is_profile") and spk_data.get("assigned_voice"):
                    v_label = spk_data["assigned_voice"]
                elif spk_data.get("custom_voice_path"):
                    v_label = f"🧬 [Klon] {os.path.basename(spk_data['custom_voice_path'])}"
                elif spk_data.get("supertonic_voice"):
                    v_label = f"⚡ Supertonic [{spk_data['supertonic_voice']}]"
                else:
                    v_label = "⚡ Supertonic [M1]"
                sub["assigned_voice"] = v_label
                sub_tree.set(item_id, "voice", v_label)

        save_project(dub_state["data"], dub_state["proj_file"])
        log(f"[Subtitles] Lines reset to character default voices ({len(sel)} lines).")

    def play_sub_voice_preview():
        chosen = sub_voice_combo.get()
        if not chosen:
            messagebox.showinfo("Voice", "Please select a voice from the dropdown list.")
            return
        ok, msg = play_any_voice_sample(chosen, target_lang=lang_combo.get().strip(), log_cb=log)
        if not ok:
            messagebox.showwarning("Warning", msg)

    ttk.Button(sub_toolbar, text="▶ Test", width=7, command=play_sub_voice_preview).pack(side="left", padx=2)
    ttk.Button(sub_toolbar, text="✓ Apply to Selected", style="Accent.TButton", command=assign_voice_to_selected_subtitles).pack(side="left", padx=3)
    ttk.Button(sub_toolbar, text="🎭 Create Profile from Selected", command=create_profile_from_selected_subtitles).pack(side="left", padx=3)
    ttk.Button(sub_toolbar, text="↺ Reset to Character", command=reset_selected_subtitles_voice).pack(side="left", padx=3)
    ttk.Button(sub_toolbar, text="Select All", command=lambda: sub_tree.selection_set(sub_tree.get_children())).pack(side="left", padx=3)

    sub_columns = ("id", "time", "speaker", "voice", "text")
    sub_tree = ttk.Treeview(sub_frame, columns=sub_columns, show="headings", height=5, selectmode="extended")

    sub_tree.heading("id", text="#", command=lambda: sort_treeview(sub_tree, "id", sub_tree_sort_state))
    sub_tree.heading("time", text="Timing", command=lambda: sort_treeview(sub_tree, "time", sub_tree_sort_state))
    sub_tree.heading("speaker", text="Character", command=lambda: sort_treeview(sub_tree, "speaker", sub_tree_sort_state))
    sub_tree.heading("voice", text="Assigned Voice", command=lambda: sort_treeview(sub_tree, "voice", sub_tree_sort_state))
    sub_tree.heading("text", text="Subtitle Text", command=lambda: sort_treeview(sub_tree, "text", sub_tree_sort_state))

    sub_tree.column("id", width=45, anchor="center")
    sub_tree.column("time", width=180, anchor="center")
    sub_tree.column("speaker", width=95, anchor="center")
    sub_tree.column("voice", width=220, anchor="w")
    sub_tree.column("text", width=480, anchor="w")

    sub_tree.pack(side="left", fill="both", expand=True)
    sub_scroll = ttk.Scrollbar(sub_frame, orient="vertical", command=sub_tree.yview)
    sub_tree.configure(yscrollcommand=sub_scroll.set)
    sub_scroll.pack(side="right", fill="y")

    sub_tree.bind("<Control-a>", lambda e: sub_tree.selection_set(sub_tree.get_children()))
    sub_tree.bind("<Double-1>", lambda e: assign_voice_to_selected_subtitles())

    def refresh_subtitle_table():
        for r in sub_tree.get_children():
            sub_tree.delete(r)

        speakers = dub_state["data"].get("speakers", {})
        subtitles = dub_state["data"].get("subtitles", [])

        for s in subtitles:
            sid = s.get("id", 1)
            spk = s.get("speaker", "Speaker_01")
            assigned_v = s.get("assigned_voice")

            # Ak replika nemá explicitne priradený hlas, získa predvolený hlas svojej postavy
            if not assigned_v:
                spk_data = speakers.get(spk, {})
                if spk_data.get("custom_voice_path"):
                    assigned_v = f"🧬 [Klon] {os.path.basename(spk_data['custom_voice_path'])}"
                elif spk_data.get("supertonic_voice"):
                    assigned_v = f"⚡ Supertonic [{spk_data['supertonic_voice']}]"
                else:
                    assigned_v = "⚡ Supertonic [M1]"
                s["assigned_voice"] = assigned_v

            time_str = f"{format_timestamp_srt(s['start'])} --> {format_timestamp_srt(s['end'])}"
            sub_tree.insert("", tk.END, iid=str(sid), values=(sid, time_str, spk, assigned_v, s.get("text", "")))

    engine_combo.bind("<<ComboboxSelected>>", lambda e: [refresh_speaker_table(), refresh_subtitle_table()])
    lang_combo.bind("<<ComboboxSelected>>", lambda e: [update_tab1_voices_dropdown(), update_spk_voices_dropdown(), refresh_speaker_table(), refresh_subtitle_table()])

    # 4. Rámec ovládania spustenia dabingu
    dub_ctrl_frame = ttk.LabelFrame(tab_dubbing, text=" 4. Dubbing Execution ", padding=5)
    dub_ctrl_frame.pack(fill="x", padx=10, pady=3)

    btn_row = ttk.Frame(dub_ctrl_frame)
    btn_row.pack(fill="x", pady=2)

    analyze_btn = ttk.Button(btn_row, text="▶ 1. Analyze & Diarize", style="Accent.TButton")
    analyze_btn.pack(side="left", padx=5)

    export_work_btn = ttk.Button(btn_row, text="Export Translation (Mode B)")
    export_work_btn.pack(side="left", padx=5)

    import_work_btn = ttk.Button(btn_row, text="Import Translation (Mode B)")
    import_work_btn.pack(side="left", padx=5)

    dub_btn = ttk.Button(btn_row, text="🚀 2. Generate Dub & Export Video", style="Accent.TButton")
    dub_btn.pack(side="left", padx=5)

    batch_dub_btn = ttk.Button(btn_row, text="🚀 Batch Dubbing (Folder)", style="Accent.TButton")
    batch_dub_btn.pack(side="left", padx=5)

    reset_proj_btn = ttk.Button(btn_row, text="🗑️ Clear Project Cache")
    reset_proj_btn.pack(side="left", padx=5)

    cancel_btn = ttk.Button(btn_row, text="⏹ Cancel", state="disabled")
    cancel_btn.pack(side="right", padx=5)

    # ==========================================================================
    # TAB 2: TEXT NARRATION & VIACHLASOVÝ PREDNES
    # ==========================================================================
    narr_top = ttk.LabelFrame(tab_narration, text=" 1. Language & Default Voice (Supertonic, Edge & Clone) ", padding=8)
    narr_top.pack(fill="x", padx=10, pady=3)

    cfg_bar = ttk.Frame(narr_top)
    cfg_bar.pack(fill="x", pady=2)

    ttk.Label(cfg_bar, text="Language:").pack(side="left", padx=(5, 2))
    narr_lang_combo = ttk.Combobox(cfg_bar, values=["cs (Czech)", "sk (Slovak)", "ja (Japanese)", "en (English)", "de (German)", "pl (Polish)"], width=15, state="readonly")
    narr_lang_combo.set("cs (Czech)")
    narr_lang_combo.pack(side="left", padx=2)

    ttk.Label(cfg_bar, text="Default Voice:").pack(side="left", padx=(12, 2))
    voice_combo = ttk.Combobox(cfg_bar, width=40, state="readonly")
    voice_combo.pack(side="left", padx=2)

    ttk.Label(cfg_bar, text="Speed:").pack(side="left", padx=(12, 2))
    speed_combo = ttk.Combobox(cfg_bar, values=["0.85", "0.90", "0.95", "1.0", "1.05", "1.10"], width=5)
    speed_combo.set("1.0")
    speed_combo.pack(side="left", padx=2)

    voice_mapping_tab2 = {}

    def update_narr_voices_dropdown(event=None):
        nonlocal voice_mapping_tab2
        cat, vmap = get_all_available_voices(narr_lang_combo.get().split()[0])
        voice_mapping_tab2 = vmap
        voice_combo['values'] = cat
        if cat and not voice_combo.get():
            voice_combo.set(cat[0])
        phrase_voice_combo['values'] = cat
        if cat and not phrase_voice_combo.get():
            phrase_voice_combo.set(cat[0])

    narr_lang_combo.bind("<<ComboboxSelected>>", lambda e: [update_narr_voices_dropdown(), parse_and_update_tab2_phrases(force_default=False)])

    btn_voice_row = ttk.Frame(narr_top)
    btn_voice_row.pack(fill="x", pady=(4, 2))

    def play_tab2_voice():
        chosen = voice_combo.get()
        if not chosen:
            messagebox.showwarning("Voice", "Please select a voice from the list.")
            return
        ok, msg = play_any_voice_sample(chosen, target_lang=narr_lang_combo.get().split()[0], log_cb=log)
        if not ok:
            messagebox.showwarning("Upozornenie", msg)

    def add_custom_voice_tab2():
        fn = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg *.m4a")])
        if fn:
            clean_name = os.path.splitext(os.path.basename(fn))[0] + ".wav"
            clean_wav = os.path.join(VOICES_DIR, clean_name)
            cmd = ["ffmpeg", "-y", "-i", fn, "-ar", str(SAMPLE_RATE), "-ac", "1", clean_wav]
            run_cmd(cmd)
            
            # Kontrola a skopírovanie .txt súboru, ak existuje
            src_txt = os.path.splitext(fn)[0] + ".txt"
            if os.path.exists(src_txt):
                shutil.copy2(src_txt, os.path.splitext(clean_wav)[0] + ".txt")

            update_narr_voices_dropdown()
            update_tab1_voices_dropdown()
            update_spk_voices_dropdown()
            messagebox.showinfo("Success", f"Voice added to ./voices/{clean_name}")

    ttk.Button(btn_voice_row, text="▶ Preview Default Voice", command=play_tab2_voice).pack(side="left", padx=3)
    ttk.Button(btn_voice_row, text="➕ Add Custom Voice (.wav/.mp3)", command=add_custom_voice_tab2).pack(side="left", padx=3)
    ttk.Button(btn_voice_row, text="📁 Open voices/ Folder", command=lambda: os.startfile(VOICES_DIR) if os.name == 'nt' else subprocess.run(['xdg-open', VOICES_DIR])).pack(side="left", padx=3)

    # 2. Vstup hlavného textu
    narr_cfg = ttk.LabelFrame(tab_narration, text=" 2. Main Text Input ", padding=6)
    narr_cfg.pack(fill="x", padx=10, pady=3)

    def load_narr_file():
        fn = filedialog.askopenfilename(filetypes=[("Text & Subtitle Files", "*.txt *.srt *.vtt"), ("All Files", "*.*")])
        if fn:
            try:
                if fn.lower().endswith((".srt", ".vtt")):
                    content = extract_clean_text_from_srt(fn)
                else:
                    with open(fn, 'r', encoding='utf-8-sig', errors='replace') as f:
                        raw = f.read()
                    # If user chose a file with SRT timestamps structure
                    if re.search(r'\d+\s*\n\d{2}:\d{2}:\d{2}[,\.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,\.]\d{3}', raw) or '-->' in raw:
                        content = extract_clean_text_from_srt(raw)
                    else:
                        content = raw
                narr_text_box.delete("1.0", tk.END)
                narr_text_box.insert("1.0", content)
                parse_and_update_tab2_phrases(force_default=False)
            except Exception as e:
                messagebox.showwarning("Notice", f"Error: {e}")

    def clean_pasted_subtitles():
        current = narr_text_box.get("1.0", tk.END).strip()
        if '-->' in current:
            cleaned = extract_clean_text_from_srt(current)
            narr_text_box.delete("1.0", tk.END)
            narr_text_box.insert("1.0", cleaned)
            parse_and_update_tab2_phrases(force_default=False)
            messagebox.showinfo("Cleaned", "Subtitle timing and line numbers stripped successfully.")
        else:
            messagebox.showinfo("Notice", "No subtitle timestamps ('-->') detected in text.")

    top_txt_bar = ttk.Frame(narr_cfg)
    top_txt_bar.pack(fill="x", pady=2)
    ttk.Label(top_txt_bar, text="Type or paste text below (automatically split into sentences):").pack(side="left")
    btn_bar = ttk.Frame(top_txt_bar)
    btn_bar.pack(side="right")
    ttk.Button(btn_bar, text="🧹 Clean Timings", command=clean_pasted_subtitles).pack(side="left", padx=2)
    ttk.Button(btn_bar, text="📂 Load from File (.txt / .srt)", command=load_narr_file).pack(side="left", padx=2)

    narr_text_box = tk.Text(narr_cfg, height=5, wrap="word", font=("Segoe UI", 10))
    narr_text_box.pack(fill="both", expand=True, padx=4, pady=3)
    
    default_text = "Azeroth calls! Deep in the forests of Ashenvale, the drums of war thunder across the land! Thrall, Warchief of the Horde, rallies his fiercest champions at the gates of Orgrimmar!"
    narr_text_box.insert("1.0", default_text)

    # 3. Zoznam viet & priradenie hlasov jednotlivým frázam (Tab 2)
    narr_phrase_frame = ttk.LabelFrame(tab_narration, text=" 3. Multi-Voice Phrase Assignment (Ctrl/Shift selection, sortable columns) ", padding=6)
    narr_phrase_frame.pack(fill="both", expand=True, padx=10, pady=3)

    phrase_toolbar = ttk.Frame(narr_phrase_frame)
    phrase_toolbar.pack(fill="x", pady=(0, 4))

    ttk.Label(phrase_toolbar, text="Voice for selected phrases:").pack(side="left", padx=(2, 4))
    phrase_voice_combo = ttk.Combobox(phrase_toolbar, width=40, state="readonly")
    phrase_voice_combo.pack(side="left", padx=2)

    def parse_and_update_tab2_phrases(force_default=False):
        raw_text = narr_text_box.get("1.0", tk.END).strip()
        if not raw_text:
            narr_phrases.clear()
            for r in phrase_tree.get_children():
                phrase_tree.delete(r)
            return

        sentences = split_text_into_sentences_list(raw_text)
        cur_cat, cur_map = get_all_available_voices(narr_lang_combo.get().split()[0])
        def_label = voice_combo.get()
        if not def_label and cur_cat:
            def_label = cur_cat[0]
        def_info = cur_map.get(def_label, {"type": "supertonic", "voice_id": "M1", "label": def_label})

        # Zachovanie predchádzajúcich priradení hlasov pre rovnaké vety
        old_by_idx = {i: p for i, p in enumerate(narr_phrases)}
        old_by_text = {p["text"].strip(): p for p in narr_phrases}

        new_phrases = []
        for idx, s in enumerate(sentences):
            s_clean = s.strip()
            v_label = def_label
            v_info = def_info

            if not force_default:
                if idx in old_by_idx and old_by_idx[idx]["text"].strip() == s_clean:
                    v_label = old_by_idx[idx]["voice_label"]
                    v_info = old_by_idx[idx]["voice_info"]
                elif s_clean in old_by_text:
                    v_label = old_by_text[s_clean]["voice_label"]
                    v_info = old_by_text[s_clean]["voice_info"]

            new_phrases.append({
                "id": idx + 1,
                "text": s_clean,
                "voice_label": v_label,
                "voice_info": v_info
            })

        narr_phrases.clear()
        narr_phrases.extend(new_phrases)

        for r in phrase_tree.get_children():
            phrase_tree.delete(r)

        for p in narr_phrases:
            phrase_tree.insert("", tk.END, iid=str(p["id"]), values=(p["id"], p["voice_label"], p["text"]))

    def assign_voice_to_selected_phrases():
        sel = phrase_tree.selection()
        if not sel:
            messagebox.showinfo("Selection", "Select at least one sentence from the table (Ctrl/Shift selection supported).")
            return
        chosen = phrase_voice_combo.get()
        if not chosen:
            messagebox.showinfo("Voice", "Please select a voice from the dropdown list.")
            return

        cur_cat, cur_map = get_all_available_voices(narr_lang_combo.get().split()[0])
        chosen_info = cur_map.get(chosen, {"type": "supertonic", "voice_id": "M1", "label": chosen})

        phrase_dict = {str(p["id"]): p for p in narr_phrases}
        for item_id in sel:
            if str(item_id) in phrase_dict:
                phrase_dict[str(item_id)]["voice_label"] = chosen
                phrase_dict[str(item_id)]["voice_info"] = chosen_info
            phrase_tree.set(item_id, "voice", chosen)

        log(f"[Narration] Voice '{chosen}' assigned to {len(sel)} sentences.")

    def reset_all_phrases_to_default():
        parse_and_update_tab2_phrases(force_default=True)
        log("[Narration] Default voice applied to all sentences.")

    def play_tab2_phrase_voice_preview():
        chosen = phrase_voice_combo.get()
        if not chosen:
            messagebox.showinfo("Voice", "Please select a voice from the dropdown list.")
            return
        ok, msg = play_any_voice_sample(chosen, target_lang=narr_lang_combo.get().split()[0], log_cb=log)
        if not ok:
            messagebox.showwarning("Upozornenie", msg)

    ttk.Button(phrase_toolbar, text="✓ Apply to Selected (Ctrl/Shift)", style="Accent.TButton", command=assign_voice_to_selected_phrases).pack(side="left", padx=4)
    ttk.Button(phrase_toolbar, text="🔄 Parse Sentences", command=lambda: parse_and_update_tab2_phrases(force_default=False)).pack(side="left", padx=3)
    ttk.Button(phrase_toolbar, text="↺ Reset All to Default", command=reset_all_phrases_to_default).pack(side="left", padx=3)
    ttk.Button(phrase_toolbar, text="▶ Preview Voice", command=play_tab2_phrase_voice_preview).pack(side="left", padx=3)
    ttk.Button(phrase_toolbar, text="Select All", command=lambda: phrase_tree.selection_set(phrase_tree.get_children())).pack(side="left", padx=3)

    phrase_columns = ("id", "voice", "text")
    phrase_tree = ttk.Treeview(narr_phrase_frame, columns=phrase_columns, show="headings", height=8, selectmode="extended")

    phrase_tree.heading("id", text="#", command=lambda: sort_treeview(phrase_tree, "id", phrase_tree_sort_state))
    phrase_tree.heading("voice", text="Assigned Voice", command=lambda: sort_treeview(phrase_tree, "voice", phrase_tree_sort_state))
    phrase_tree.heading("text", text="Sentence / Phrase Text", command=lambda: sort_treeview(phrase_tree, "text", phrase_tree_sort_state))

    phrase_tree.column("id", width=45, anchor="center")
    phrase_tree.column("voice", width=260, anchor="w")
    phrase_tree.column("text", width=680, anchor="w")

    phrase_tree.pack(side="left", fill="both", expand=True)
    phrase_scroll = ttk.Scrollbar(narr_phrase_frame, orient="vertical", command=phrase_tree.yview)
    phrase_tree.configure(yscrollcommand=phrase_scroll.set)
    phrase_scroll.pack(side="right", fill="y")

    phrase_tree.bind("<Control-a>", lambda e: phrase_tree.selection_set(phrase_tree.get_children()))
    phrase_tree.bind("<Double-1>", lambda e: assign_voice_to_selected_phrases())

    # Debounce for text editing in Tab 2
    _typing_timer = None
    def on_narr_text_modified(event=None):
        nonlocal _typing_timer
        if _typing_timer is not None:
            root.after_cancel(_typing_timer)
        _typing_timer = root.after(700, lambda: parse_and_update_tab2_phrases(force_default=False))

    narr_text_box.bind("<KeyRelease>", on_narr_text_modified)

    # 4. Spodné akcie pre Tab 2
    narr_action_f = ttk.Frame(tab_narration)
    narr_action_f.pack(fill="x", padx=10, pady=5)

    narr_gen_btn = ttk.Button(narr_action_f, text="🎙️ Generate Speech (MP3 160 kbps)", style="Accent.TButton")
    narr_gen_btn.pack(side="left", padx=5)

    narr_play_out_btn = ttk.Button(narr_action_f, text="▶ Play Generated MP3", state="disabled")
    narr_play_out_btn.pack(side="left", padx=5)

    narr_save_btn = ttk.Button(narr_action_f, text="💾 Save MP3 As...", state="disabled")
    narr_save_btn.pack(side="left", padx=5)

    narr_cancel_btn = ttk.Button(narr_action_f, text="⏹ Cancel", state="disabled")
    narr_cancel_btn.pack(side="right", padx=5)

    # ==========================================================================
    # SPOLOČNÝ PANEL: PROGRESS & LOGS
    # ==========================================================================
    bottom_frame = ttk.LabelFrame(root, text=" System Status & Log ", padding=4)
    bottom_frame.pack(before=notebook, side="bottom", fill="x", padx=10, pady=(2, 4))

    prog_var = ttk.Progressbar(bottom_frame, mode="determinate")
    prog_var.pack(fill="x", padx=5, pady=2)

    status_row = ttk.Frame(bottom_frame)
    status_row.pack(fill="x", padx=5)
    
    status_lbl = ttk.Label(status_row, text="Ready.", font=('Segoe UI', 9))
    status_lbl.pack(side="left")

    vram_lbl = ttk.Label(status_row, text=get_vram_status(), font=('Segoe UI', 8, 'italic'))
    vram_lbl.pack(side="right")

    log_txt = tk.Text(bottom_frame, height=5, bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 8))
    log_txt.pack(fill="both", expand=True, padx=5, pady=3)

    def log(msg):
        log_queue.put(msg)

    def process_logs():
        while not log_queue.empty():
            msg = log_queue.get_nowait()
            log_txt.insert(tk.END, msg + "\n")
            log_txt.see(tk.END)
            status_lbl.config(text=msg)
            vram_lbl.config(text=get_vram_status())
        root.after(120, process_logs)

    def init_project_state(video_path):
        folders, proj_file = create_project_structure(video_path)
        dub_state["video_path"] = video_path
        dub_state["folders"] = folders
        dub_state["proj_file"] = proj_file
        dub_state["data"] = load_or_init_project(video_path, proj_file)
        refresh_speaker_table()
        refresh_subtitle_table()
        log(f"[Project] Directory: {folders['root']}")

    # ==========================================================================
    # LOGIKA PRE TAB 1 (VIDEO DUBBING & DÁVKOVÝ DABING)
    # ==========================================================================
    def find_matching_subtitle(video_path, target_lang="cs"):
        """
        Searches for a subtitle file (.srt/.ass) associated with the given video:
        1. With target language suffix: e.g. video.en.srt / video.cs.srt / video.ja.srt
        2. Exact match: video.srt / video.ass
        3. Any language suffix: video.*.srt / video.*.ass (e.g. video.en.srt)
        """
        folder = os.path.dirname(os.path.abspath(video_path))
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        sub_exts = ('.srt', '.ass')
        
        # 1. Presná zhoda s cieľovým jazykom
        lang_code = target_lang.split()[0].lower() if target_lang else "cs"
        for ext in sub_exts:
            p = os.path.join(folder, f"{base_name}.{lang_code}{ext}")
            if os.path.exists(p):
                return p
                
        # 2. Presná zhoda s kmeňom videa
        for ext in sub_exts:
            p = os.path.join(folder, f"{base_name}{ext}")
            if os.path.exists(p):
                return p

        # 3. Zhoda s bodkou a akýmkoľvek jazykom (napr. video.en.srt)
        prefix = base_name.lower() + "."
        candidates = []
        try:
            for f in os.listdir(folder):
                if f.lower().startswith(prefix):
                    for ext in sub_exts:
                        if f.lower().endswith(ext):
                            candidates.append(os.path.join(folder, f))
        except Exception:
            pass
            
        if candidates:
            return candidates[0]
            
        return None

    def scan_directory_for_video_pairs(folder_path, target_lang="cs"):
        """
        Finds all video files in a folder that have an existing matching .srt/.ass subtitle file.
        Ignores already dubbed videos (_dubbed_) and subdirectories.
        """
        pairs = []
        if not os.path.exists(folder_path):
            return pairs
        valid_vid_exts = ('.mp4', '.mkv', '.avi', '.mov', '.webm', '.wmv', '.flv')
        for f in sorted(os.listdir(folder_path)):
            full_vid_path = os.path.join(folder_path, f)
            if not os.path.isfile(full_vid_path):
                continue
            if "_dubbed_" in f.lower():
                continue
            if not f.lower().endswith(valid_vid_exts):
                continue
            sub_path = find_matching_subtitle(full_vid_path, target_lang=target_lang)
            if sub_path:
                pairs.append((full_vid_path, sub_path))
        return pairs

    def process_analyze(v_path, srt_p=None, is_batch=False):
        try:
            if not os.path.exists(v_path):
                if not is_batch:
                    messagebox.showerror("Error", "Video file does not exist!")
                log(f"[Error] Video file does not exist: {v_path}")
                return False
            
            init_project_state(v_path)
            if not is_batch:
                dub_state["cancel_requested"] = False
                analyze_btn.config(state="disabled")
                dub_btn.config(state="disabled")
                batch_dub_btn.config(state="disabled")
                cancel_btn.config(state="normal")

            prog_var.config(value=10)

            if dub_state["cancel_requested"]:
                return False

            extracted_audio = os.path.join(dub_state["folders"]["audio"], "extracted_audio.wav")
            if not os.path.exists(extracted_audio):
                log("[Audio] Extracting audio track...")
                extract_audio_from_video(v_path, extracted_audio)
            dub_state["data"]["extracted_audio"] = extracted_audio

            if dub_state["cancel_requested"]:
                return False

            use_sep = sep_var.get()
            dub_state["data"]["use_separation"] = use_sep
            speech_audio = extracted_audio

            if use_sep:
                vocals_path = os.path.join(dub_state["folders"]["separation"], "vocals.wav")
                bgm_path = os.path.join(dub_state["folders"]["separation"], "bgm.wav")
                if not (os.path.exists(vocals_path) and os.path.exists(bgm_path)):
                    prog_var.config(value=25)
                    separate_sources(extracted_audio, dub_state["folders"]["separation"], log_cb=log)
                speech_audio = vocals_path
                dub_state["data"]["vocals_audio"] = vocals_path
                dub_state["data"]["bgm_audio"] = bgm_path

            if dub_state["cancel_requested"]:
                return False

            prog_var.config(value=50)
            m = mode_var.get()
            dub_state["data"]["mode"] = m
            
            if m == "EXISTING_SRT" or is_batch:
                actual_srt = srt_p if srt_p else sub_entry.get().strip()
                if not (actual_srt and os.path.exists(actual_srt)):
                    if not is_batch:
                        ans = messagebox.askyesno(
                            "Subtitles Missing",
                            "No subtitle (.srt) file was specified.\n\n"
                            "Would you like to automatically transcribe dialogue from the video using Whisper now?"
                        )
                        if ans:
                            log("[STT] Auto-transcribing dialogue with Whisper...")
                            from faster_whisper import WhisperModel
                            whisper = WhisperModel("medium", device=DEVICE, compute_type="float16" if "cuda" in DEVICE else "int8")
                            segments_raw, _ = whisper.transcribe(speech_audio, vad_filter=True)
                            segments = []
                            srt_lines = []
                            for idx_s, s in enumerate(segments_raw, 1):
                                if s.text.strip():
                                    s_start = round(s.start, 3)
                                    s_end = round(s.end, 3)
                                    srt_lines.append(f"{idx_s}\n{format_timestamp_srt(s_start)} --> {format_timestamp_srt(s_end)}\n{s.text.strip()}\n")
                                    segments.append({
                                        "id": idx_s,
                                        "start": s_start,
                                        "end": s_end,
                                        "text": s.text.strip(),
                                        "speaker": "Speaker_01",
                                        "assigned_voice": ""
                                    })
                            del whisper
                            clear_vram()
                            base_name = os.path.splitext(os.path.basename(v_path))[0]
                            auto_srt = os.path.join(dub_state["folders"]["subtitles"], f"{base_name}_whisper.srt")
                            with open(auto_srt, "w", encoding="utf-8") as f:
                                f.write("\n".join(srt_lines) + "\n")
                            sub_entry.delete(0, tk.END)
                            sub_entry.insert(0, auto_srt)
                            dub_state["data"]["sub_path"] = auto_srt
                            dub_state["data"]["subtitles"] = segments
                        else:
                            log(f"[Error] Subtitle file not found: {actual_srt}")
                            return False
                    else:
                        log(f"[Error] Subtitle file not found: {actual_srt}")
                        return False
                else:
                    log(f"[Subtitles] Loading subtitles: {os.path.basename(actual_srt)}...")
                    segments = parse_srt_file(actual_srt)
            else:
                log("[STT] Transcribing speech to text with Whisper...")
                from faster_whisper import WhisperModel
                whisper = WhisperModel("medium", device=DEVICE, compute_type="float16" if "cuda" in DEVICE else "int8")
                segments_raw, _ = whisper.transcribe(speech_audio, vad_filter=True)
                segments = []
                for s in segments_raw:
                    if s.text.strip():
                        segments.append({
                            "id": len(segments) + 1,
                            "start": round(s.start, 3),
                            "end": round(s.end, 3),
                            "text": s.text.strip(),
                            "speaker": "Speaker_01",
                            "assigned_voice": ""
                        })
                del whisper
                clear_vram()

            if dub_state["cancel_requested"]:
                return False

            has_diarization = dub_state["data"].get("steps_completed", {}).get("diarization", False)
            existing_speakers = dub_state["data"].get("speakers", {})
            force_re = force_reanalyze_var.get()

            if has_diarization and existing_speakers and not force_re:
                log(f"[Analysis] Using existing character analysis and voice assignments from project ({len(existing_speakers)} characters). Diarization skipped.")
                speakers = existing_speakers
                segments = sync_subtitles_with_disk(dub_state["data"].get("subtitles", []), segments, speakers)
                prog_var.config(value=80)
            else:
                max_spks = int(max_spk_spin.get())
                cur_prof = pitch_profile_var.get()
                prof_thresh = get_selected_profile_threshold()
                segments, speakers = diarize_segments_direct(
                    speech_audio, segments, max_speakers=max_spks,
                    log_cb=log, profile=cur_prof, similarity_threshold=prof_thresh
                )
                prog_var.config(value=80)

                if dub_state["cancel_requested"]:
                    return False

                log("[Preview] Preparing clean voice samples...")
                generate_speaker_previews(speech_audio, speakers, dub_state["folders"]["speakers"])

                # Assign each subtitle line its character's default voice (Supertonic or Voice Profile)
                for seg in segments:
                    spk = seg.get("speaker", "Speaker_01")
                    spk_data = speakers.get(spk, {})
                    if spk_data.get("is_profile") and spk_data.get("assigned_voice"):
                        seg["assigned_voice"] = spk_data["assigned_voice"]
                    else:
                        st_v = spk_data.get("supertonic_voice", "M1")
                        seg["assigned_voice"] = f"⚡ Supertonic [{st_v}]"

                dub_state["data"]["steps_completed"]["diarization"] = True

            dub_state["data"]["subtitles"] = segments
            dub_state["data"]["speakers"] = speakers

            save_project(dub_state["data"], dub_state["proj_file"])
            root.after(0, refresh_speaker_table)
            root.after(0, refresh_subtitle_table)
            prog_var.config(value=100)
            log(f"[Success] Analysis ready. Characters: {len(speakers)}, Subtitle lines: {len(segments)}.")
            return True

        except Exception as e:
            log(f"[ANALYSIS ERROR] {str(e)}")
            if not is_batch:
                messagebox.showerror("Error", f"Error during character analysis:\n{str(e)}")
            return False
        finally:
            if not is_batch:
                analyze_btn.config(state="normal")
                dub_btn.config(state="normal")
                batch_dub_btn.config(state="normal")
                cancel_btn.config(state="disabled")

    def process_dub(is_batch=False):
        try:
            if not dub_state["data"].get("speakers"):
                if not is_batch:
                    messagebox.showwarning("Notice", "Please run Step 1 (Analysis) first.")
                return False

            if not is_batch:
                analyze_btn.config(state="disabled")
                dub_btn.config(state="disabled")
                batch_dub_btn.config(state="disabled")
                cancel_btn.config(state="normal")
                dub_state["cancel_requested"] = False

            tgt_lang = lang_combo.get().strip()
            dub_state["data"]["target_lang"] = tgt_lang

            # Automatically check and sync with changes in the .srt file on disk
            actual_srt = sub_entry.get().strip() or dub_state["data"].get("sub_path")
            if actual_srt and os.path.exists(actual_srt):
                disk_segments = parse_srt_file(actual_srt)
                dub_state["data"]["subtitles"] = sync_subtitles_with_disk(
                    dub_state["data"].get("subtitles", []),
                    disk_segments,
                    dub_state["data"].get("speakers", {})
                )
                save_project(dub_state["data"], dub_state["proj_file"])
                root.after(0, refresh_subtitle_table)

            raw_subtitles = dub_state["data"].get("subtitles", [])
            spks_map = dub_state["data"].get("speakers", {})
            subtitles = merge_continuous_subtitles(raw_subtitles, max_gap=0.40, speakers=spks_map)

            # Clean legacy index-based audio chunks (hash-based chunks are preserved)
            gen_dir = dub_state["folders"]["generated"]
            if os.path.exists(gen_dir):
                _valid_keys = set()
                for _s in subtitles:
                    _valid_keys.add(_s.get("audio_chunk_key", ""))
                for f in os.listdir(gen_dir):
                    m = re.match(r'^(?:raw|sub)_(\d{5})\.(?:wav|txt)$', f)
                    if m:
                        sub_num = int(m.group(1))
                        if sub_num > len(subtitles):
                            try:
                                os.remove(os.path.join(gen_dir, f))
                            except Exception:
                                pass

            eng_choice = engine_var.get()
            dub_engine = "supertonic" if "Supertonic" in eng_choice else ("xtts" if "Klonovanie" in eng_choice else "edge")

            dub_state["tts_engine"] = DynamicDubbingEngine(target_lang=tgt_lang, engine_type=dub_engine, state=dub_state)

            total_subs = len(subtitles)
            generated_items = []
            cached_lines_count = 0
            new_lines_count = 0

            for idx, item in enumerate(subtitles):
                if dub_state["cancel_requested"]:
                    log("[Dubbing] Dubbing cancelled by user.")
                    return False

                sub_id = item["id"]
                spk = item["speaker"]
                text = " ".join(item["text"].replace('\r', ' ').replace('\n', ' ').split())
                
                allocated_time = max(0.6, item["end"] - item["start"])
                for nxt_idx in range(idx + 1, min(idx + 5, total_subs)):
                    nxt = subtitles[nxt_idx]
                    if nxt["speaker"] == spk:
                        if nxt["start"] > item["start"]:
                            allocated_time = min(allocated_time, max(0.5, nxt["start"] - item["start"] - 0.04))
                        break

                # Resolve TTS engine and voice for this specific subtitle line
                assigned_v_str = item.get("assigned_voice", "")
                act_engine, act_voice, act_ref_wav = resolve_voice_for_subtitle(assigned_v_str, spk, dub_state, lang_code=tgt_lang)

                # Stable content-hash chunk naming (prevents line deletions from causing cascade re-generation)
                import hashlib as _hashlib
                _voice_key = act_voice or os.path.basename(act_ref_wav or "")
                cache_sig = f"[{act_engine}:{_voice_key}]\n{text}".strip()
                _chunk_hash = _hashlib.md5(cache_sig.encode("utf-8")).hexdigest()[:12]
                # Backward compatibility: reuse legacy sub_{sub_id} audio chunk if identical
                _old_key = item.get("audio_chunk_key", f"sub_{sub_id:05d}")
                raw_wav = os.path.join(dub_state["folders"]["generated"], f"raw_{_chunk_hash}.wav")
                final_gen_wav = os.path.join(dub_state["folders"]["generated"], f"sub_{_chunk_hash}.wav")
                sub_txt = os.path.join(dub_state["folders"]["generated"], f"sub_{_chunk_hash}.txt")
                # Backward compatibility: check legacy sub_{sub_id} files
                _old_wav = os.path.join(dub_state["folders"]["generated"], f"{_old_key}.wav")
                _old_txt = os.path.join(dub_state["folders"]["generated"], f"{_old_key}.txt")
                if not os.path.exists(final_gen_wav) and os.path.exists(_old_wav):
                    try:
                        import shutil as _shutil
                        _shutil.copy2(_old_wav, final_gen_wav)
                        if os.path.exists(_old_txt):
                            _shutil.copy2(_old_txt, sub_txt)
                    except Exception:
                        pass
                cache_valid = False

                if os.path.exists(final_gen_wav) and os.path.getsize(final_gen_wav) > 100 and os.path.exists(sub_txt):
                    try:
                        with open(sub_txt, "r", encoding="utf-8-sig") as f:
                            if f.read().strip() == cache_sig:
                                cache_valid = True
                    except Exception:
                        cache_valid = False

                if cache_valid:
                    cached_lines_count += 1
                else:
                    new_lines_count += 1
                    log(f"[Synthesis ({act_engine})] ({idx+1}/{total_subs}) [{spk} | {act_voice or os.path.basename(act_ref_wav or '')}] '{text[:35]}...'")
                    
                    _, persona_eq = dub_state["tts_engine"].generate_speech(
                        text=text,
                        spk_id=spk,
                        ref_voice_wav=act_ref_wav if act_engine == "xtts" else None,
                        output_wav_path=raw_wav,
                        target_duration=allocated_time,
                        log_cb=log,
                        explicit_voice=act_voice,
                        explicit_engine=act_engine
                    )
                    fit_audio_strictly_to_slot(raw_wav, final_gen_wav, allocated_time, eq_persona=persona_eq)

                    with open(sub_txt, "w", encoding="utf-8") as f:
                        f.write(cache_sig + "\n")

                item["final_audio"] = final_gen_wav
                item["audio_chunk_key"] = f"sub_{_chunk_hash}"
                item["actual_start"] = item["start"]
                generated_items.append(item)
                prog_var.config(value=int((idx + 1) / max(1, total_subs) * 75))

            log(f"[Synthesis] Done: {cached_lines_count} lines loaded from cache, {new_lines_count} newly synthesized.")
            dub_state["tts_engine"].unload_model()

            if dub_state["cancel_requested"]:
                log("[Dubbing] Dubbing was cancelled by user.")
                return False

            v_path = dub_state["video_path"]
            total_video_dur = get_audio_duration(v_path)
            
            v_rms = 0.075
            vocals_file = dub_state["data"].get("vocals_audio")
            if vocals_file and os.path.exists(vocals_file):
                v_data, _ = sf.read(vocals_file)
                if len(v_data.shape) > 1:
                    v_data = np.mean(v_data, axis=1)
                active_v = v_data[np.abs(v_data) > 0.015]
                if len(active_v) > 0:
                    v_rms = float(np.clip(np.sqrt(np.mean(active_v**2)), 0.050, 0.085))

            log(f"[Mixer] Running Anti-Collision Manager (Matched RMS: {v_rms:.3f})...")
            dialogue_track = assemble_dialogue_track_anticollision(generated_items, total_video_dur, target_rms=v_rms)

            mixed_master_wav = os.path.join(dub_state["folders"]["output"], "final_master_audio.wav")
            orig_audio = dub_state["data"]["extracted_audio"]
            bgm_audio = dub_state["data"].get("bgm_audio", orig_audio)

            d_vol = float(dlg_vol_var.get())
            b_vol = float(bg_vol_var.get())
            duck_val = float(duck_var.get())

            log(f"[Mixer] Applying Master Pass-through (Dubbing: {int(d_vol*100)}%, Background: {int(b_vol*100)}%, Ducking: {int(duck_val*100)}%)...")
            create_adaptive_master_mix(orig_audio, bgm_audio, dialogue_track, generated_items, mixed_master_wav, dlg_vol=d_vol, bg_vol=b_vol, ducking=duck_val)

            ts_str = time.strftime("%Y%m%d_%H%M%S")
            base_vname = os.path.splitext(os.path.basename(v_path))[0]
            out_video = os.path.join(dub_state["folders"]["output"], f"{base_vname}_dubbed_{tgt_lang}_{ts_str}.mkv")

            mix_and_export_video(v_path, mixed_master_wav, out_video, target_lang=tgt_lang)

            prog_var.config(value=100)
            dub_state["data"]["steps_completed"]["final_export"] = True
            save_project(dub_state["data"], dub_state["proj_file"])

            log(f"[Done] Video created: {out_video}")
            if not is_batch:
                messagebox.showinfo("Done", f"Dubbing completed!\nOutput file:\n{out_video}")
            return True

        except Exception as e:
            log(f"[DUBBING ERROR] {str(e)}")
            if not is_batch:
                messagebox.showerror("Error", f"Error during dubbing:\n{str(e)}")
            return False
        finally:
            if not is_batch:
                analyze_btn.config(state="normal")
                dub_btn.config(state="normal")
                batch_dub_btn.config(state="normal")
                cancel_btn.config(state="disabled")

    def task_transcribe_whisper():
        v = video_entry.get().strip()
        if not (v and os.path.exists(v)):
            messagebox.showwarning("Notice", "Please select a valid video/audio file first.")
            return

        init_project_state(v)
        dub_state["cancel_requested"] = False
        analyze_btn.config(state="disabled")
        dub_btn.config(state="disabled")
        batch_dub_btn.config(state="disabled")
        if 'whisper_sub_btn' in locals() or 'whisper_sub_btn' in globals():
            whisper_sub_btn.config(state="disabled")
        cancel_btn.config(state="normal")

        try:
            log("[Whisper] Starting automated subtitle transcription...")
            prog_var.config(value=10)

            extracted_audio = os.path.join(dub_state["folders"]["audio"], "extracted_audio.wav")
            if not os.path.exists(extracted_audio):
                log("[Audio] Extracting audio track from video...")
                extract_audio_from_video(v, extracted_audio)
            dub_state["data"]["extracted_audio"] = extracted_audio

            if dub_state["cancel_requested"]:
                log("[Whisper] Cancelled by user.")
                return

            vocals_path = os.path.join(dub_state["folders"]["separation"], "vocals.wav")
            speech_audio = vocals_path if os.path.exists(vocals_path) else extracted_audio

            prog_var.config(value=30)
            log(f"[Whisper] Transcribing speech from '{os.path.basename(speech_audio)}' (Medium model)...")

            from faster_whisper import WhisperModel
            whisper = WhisperModel("medium", device=DEVICE, compute_type="float16" if "cuda" in DEVICE else "int8")
            segments_raw, info = whisper.transcribe(speech_audio, vad_filter=True)

            srt_lines = []
            parsed_segments = []
            idx = 1
            for s in segments_raw:
                if dub_state["cancel_requested"]:
                    break
                txt = s.text.strip()
                if txt:
                    s_start = round(s.start, 3)
                    s_end = round(s.end, 3)
                    srt_lines.append(f"{idx}\n{format_timestamp_srt(s_start)} --> {format_timestamp_srt(s_end)}\n{txt}\n")
                    parsed_segments.append({
                        "id": idx,
                        "start": s_start,
                        "end": s_end,
                        "text": txt,
                        "speaker": "Speaker_01",
                        "assigned_voice": ""
                    })
                    idx += 1

            del whisper
            clear_vram()

            if dub_state["cancel_requested"]:
                log("[Whisper] Cancelled by user.")
                return

            if not parsed_segments:
                log("[Whisper] No speech detected in the audio track.")
                messagebox.showinfo("Whisper", "No speech detected in the audio track.")
                return

            base_name = os.path.splitext(os.path.basename(v))[0]
            srt_out_path = os.path.join(dub_state["folders"]["subtitles"], f"{base_name}_whisper.srt")
            with open(srt_out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(srt_lines) + "\n")

            sub_entry.delete(0, tk.END)
            sub_entry.insert(0, srt_out_path)
            mode_var.set("EXISTING_SRT")
            dub_state["data"]["sub_path"] = srt_out_path
            dub_state["data"]["subtitles"] = parsed_segments
            save_project(dub_state["data"], dub_state["proj_file"])

            root.after(0, refresh_subtitle_table)
            prog_var.config(value=100)
            log(f"[Whisper] Generated {len(parsed_segments)} subtitle lines -> {srt_out_path}")
            messagebox.showinfo("Whisper Subtitles", f"Successfully generated {len(parsed_segments)} subtitle lines!\n\nSaved to:\n{srt_out_path}")

        except Exception as e:
            log(f"[Whisper Error] {e}")
            messagebox.showerror("Whisper Error", f"Failed to transcribe subtitles:\n{e}")
        finally:
            analyze_btn.config(state="normal")
            dub_btn.config(state="normal")
            batch_dub_btn.config(state="normal")
            if 'whisper_sub_btn' in locals() or 'whisper_sub_btn' in globals():
                whisper_sub_btn.config(state="normal")
            cancel_btn.config(state="disabled")

    def task_analyze():
        v = video_entry.get().strip()
        s = sub_entry.get().strip()
        process_analyze(v, srt_p=s, is_batch=False)

    def task_dub():
        process_dub(is_batch=False)

    def task_batch_dub():
        try:
            v_entry_val = video_entry.get().strip()
            if v_entry_val and os.path.exists(v_entry_val):
                folder_to_scan = os.path.dirname(os.path.abspath(v_entry_val))
            else:
                folder_to_scan = SCRIPT_DIR

            log(f"[Batch] Scanning folder: {folder_to_scan}")
            tgt_lang = lang_combo.get().strip()
            pairs = scan_directory_for_video_pairs(folder_to_scan, target_lang=tgt_lang)

            if not pairs and folder_to_scan != SCRIPT_DIR:
                log(f"[Batch] No videos in {folder_to_scan}, checking script directory: {SCRIPT_DIR}")
                folder_to_scan = SCRIPT_DIR
                pairs = scan_directory_for_video_pairs(folder_to_scan, target_lang=tgt_lang)

            if not pairs:
                log(f"[Batch] No videos with matching subtitles (.srt/.ass) found in '{folder_to_scan}'.")
                messagebox.showinfo(
                    "Batch Dubbing",
                    f"In folder:\n{folder_to_scan}\n\nNo video files found with matching subtitle files (.srt/.ass with identical name or language suffix like video.en.srt)."
                )
                return

            analyze_btn.config(state="disabled")
            dub_btn.config(state="disabled")
            batch_dub_btn.config(state="disabled")
            cancel_btn.config(state="normal")
            dub_state["cancel_requested"] = False

            total_videos = len(pairs)
            log(f"[Batch] Found {total_videos} videos with subtitles queued for processing.")
            for v_f, s_f in pairs:
                log(f"   -> Video: {os.path.basename(v_f)} | Subtitles: {os.path.basename(s_f)}")

            completed_count = 0
            for b_idx, (vid_file, sub_file) in enumerate(pairs):
                if dub_state["cancel_requested"]:
                    log("[Batch] Batch dubbing was interrupted by user.")
                    break

                # Check: if video is already dubbed and .srt is unchanged, skip
                if not force_reanalyze_var.get() and is_batch_video_up_to_date(vid_file, sub_file, tgt_lang):
                    log(f"[Batch] ({b_idx + 1}/{total_videos}) Video {os.path.basename(vid_file)} already up to date (.srt unchanged). Skipping.")
                    completed_count += 1
                    continue

                log("[Batch] ============================================================")
                log(f"[Batch] ({b_idx + 1}/{total_videos}) STARTING: {os.path.basename(vid_file)}")
                log(f"[Batch] Subtitles: {os.path.basename(sub_file)}")
                log(f"[Dávka] ============================================================")
                
                # Aktualizácia polí v GUI
                root.after(0, lambda v=vid_file, s=sub_file: [
                    video_entry.delete(0, tk.END),
                    video_entry.insert(0, v),
                    sub_entry.delete(0, tk.END),
                    sub_entry.insert(0, s)
                ])

                # Step 1: Character Analysis & Audio Separation
                ok_analyze = process_analyze(vid_file, srt_p=sub_file, is_batch=True)
                if not ok_analyze or dub_state["cancel_requested"]:
                    log(f"[Batch] Analysis cancelled or failed for {os.path.basename(vid_file)}.")
                    break

                # Step 2: Speech Synthesis & Video Export
                ok_dub = process_dub(is_batch=True)
                if not ok_dub or dub_state["cancel_requested"]:
                    log(f"[Batch] Dubbing cancelled or failed for {os.path.basename(vid_file)}.")
                    break

                completed_count += 1
                log(f"[Batch] ({b_idx + 1}/{total_videos}) COMPLETED: {os.path.basename(vid_file)}")

            if dub_state["cancel_requested"]:
                log(f"[Batch] Interrupted on user request. Completed: {completed_count} of {total_videos} videos.")
                messagebox.showinfo("Cancelled", f"Batch processing was cancelled by user.\nCompleted {completed_count} of {total_videos} videos.")
            else:
                log(f"[Batch] Done! All {completed_count} videos successfully dubbed and exported.")
                messagebox.showinfo("Batch Dubbing Done", f"Successfully completed batch dubbing for all {completed_count} videos!")

        except Exception as e:
            log(f"[BATCH ERROR] {str(e)}")
            messagebox.showerror("Batch Error", f"Error during batch processing:\n{str(e)}")
        finally:
            analyze_btn.config(state="normal")
            dub_btn.config(state="normal")
            batch_dub_btn.config(state="normal")
            cancel_btn.config(state="disabled")

    def handle_export_work():
        if not dub_state["data"].get("subtitles"):
            messagebox.showinfo("Info", "Please run character analysis first.")
            return
        fn = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if fn:
            export_working_file(dub_state["data"]["subtitles"], fn)
            log(f"[Mode B] Exported to: {fn}")
            messagebox.showinfo("Done", f"File saved successfully:\n{fn}")

    def handle_import_work():
        fn = filedialog.askopenfilename(filetypes=[("Text", "*.txt")])
        if fn:
            items = import_working_file(fn)
            if items:
                dub_state["data"]["subtitles"] = items
                save_project(dub_state["data"], dub_state["proj_file"])
                refresh_subtitle_table()
                log(f"[Mode B] Loaded {len(items)} subtitle lines.")
                messagebox.showinfo("Done", f"Loaded {len(items)} edited subtitle lines.")

    # ==========================================================================
    # LOGIKA PRE TAB 2 (TEXT NARRATION / MP3 S VIACHLASOVÝMI FRÁZAMI)
    # ==========================================================================
    def task_generate_narration():
        try:
            speed = float(speed_combo.get())
            lang_code = narr_lang_combo.get().split()[0]

            if not narr_phrases:
                parse_and_update_tab2_phrases(force_default=False)

            if not narr_phrases:
                messagebox.showerror("Error", "Input text is empty!")
                return

            narr_gen_btn.config(state="disabled")
            narr_cancel_btn.config(state="normal")
            narr_play_out_btn.config(state="disabled")
            narr_save_btn.config(state="disabled")
            narr_state["cancel_requested"] = False
            prog_var.config(value=5)

            log(f"[Narration] Starting multi-voice narration ({len(narr_phrases)} sentences, Language: {lang_code})...")

            temp_narr_dir = os.path.join(SCRIPT_DIR, "output_narration")
            os.makedirs(temp_narr_dir, exist_ok=True)

            tot = len(narr_phrases)
            audio_pieces = []
            OUTPUT_SR = 44100
            silence_sentence = np.zeros(int(OUTPUT_SR * 0.05), dtype=np.float32)

            # Inicializujeme generátor
            active_engine_obj = DynamicDubbingEngine(target_lang=lang_code, engine_type="supertonic")

            for idx, item in enumerate(narr_phrases):
                if narr_state["cancel_requested"]:
                    break

                chunk_text = item["text"].strip()
                v_info = item["voice_info"]
                v_type = v_info.get("type", "supertonic")
                v_id = v_info.get("voice_id")
                ref_wav = v_info.get("path") if v_type == "clone" else None

                # Referenčný prepis pre klonovanie
                if ref_wav and os.path.exists(ref_wav):
                    get_or_create_reference_transcription(ref_wav, log_cb=log)

                tmp_piece = os.path.join(temp_narr_dir, f"chunk_{idx:04d}.wav")
                tmp_txt = os.path.join(temp_narr_dir, f"chunk_{idx:04d}.txt")

                # Cache signature: voice type + voice ID/path + speed + phrase text
                voice_sig = f"[{v_type}:{v_id or os.path.basename(ref_wav or '')}:{speed}]"
                expected_cache = f"{voice_sig}\n{chunk_text}".strip()

                cache_valid = False
                if os.path.exists(tmp_piece) and os.path.getsize(tmp_piece) > 1000 and os.path.exists(tmp_txt):
                    try:
                        with open(tmp_txt, "r", encoding="utf-8-sig") as f:
                            if f.read().strip() == expected_cache:
                                cache_valid = True
                    except Exception:
                        cache_valid = False

                if cache_valid:
                    log(f"[Narration] ({idx+1}/{tot}) Using cached chunk ({item['voice_label'][:20]}): '{chunk_text[:35]}...'")
                else:
                    log(f"[Narration] ({idx+1}/{tot}) Synthesizing [{item['voice_label'][:25]}]: '{chunk_text[:35]}...'")
                    active_engine_obj.generate_speech(
                        text=chunk_text,
                        spk_id="Narrator",
                        ref_voice_wav=ref_wav,
                        output_wav_path=tmp_piece,
                        target_duration=None,
                        log_cb=log,
                        explicit_voice=v_id,
                        explicit_engine=v_type
                    )
                    with open(tmp_txt, "w", encoding="utf-8") as f:
                        f.write(expected_cache + "\n")

                if os.path.exists(tmp_piece) and os.path.getsize(tmp_piece) > 1000:
                    data, piece_sr = sf.read(tmp_piece)
                    if len(data.shape) > 1:
                        data = np.mean(data, axis=1)
                    data = trim_audio_silence(data, piece_sr, db_thresh=30)
                    
                    if piece_sr != OUTPUT_SR:
                        import math
                        import scipy.signal
                        g = math.gcd(int(OUTPUT_SR), int(piece_sr))
                        up = int(OUTPUT_SR) // g
                        down = int(piece_sr) // g
                        data = scipy.signal.resample_poly(data, up, down)

                    audio_pieces.append(data.astype(np.float32))
                    audio_pieces.append(silence_sentence)

                prog_var.config(value=int(10 + (idx + 1) / max(1, tot) * 80))

            active_engine_obj.unload_model()

            if narr_state["cancel_requested"]:
                log("[Narration] Generation was cancelled.")
                return

            if not audio_pieces:
                raise RuntimeError("Nepodarilo sa vytvoriť zvukovú stopu.")

            combined = np.concatenate(audio_pieces)
            max_v = np.max(np.abs(combined))
            if max_v > 0:
                combined = (combined / max_v) * 0.92

            master_wav = os.path.join(temp_narr_dir, "master_narration.wav")
            sf.write(master_wav, combined, OUTPUT_SR)

            ts = time.strftime("%Y%m%d_%H%M%S")
            final_mp3 = os.path.join(temp_narr_dir, f"narration_{lang_code}_{ts}.mp3")
            convert_to_mp3_160k(master_wav, final_mp3)

            narr_state["last_mp3"] = final_mp3
            prog_var.config(value=100)
            narr_play_out_btn.config(state="normal")
            narr_save_btn.config(state="normal")

            dur = get_audio_duration(final_mp3)
            log(f"[Done] MP3 audio ready ({dur:.1f} s).")
            messagebox.showinfo("Done", f"Narration successfully generated!\n\nDuration: {dur:.1f} s\nFile: {final_mp3}")

        except Exception as e:
            log(f"[ERROR] {str(e)}")
            messagebox.showerror("Error", f"Error during speech generation:\n{str(e)}")
        finally:
            narr_gen_btn.config(state="normal")
            narr_cancel_btn.config(state="disabled")

    def play_narr_output():
        if narr_state["last_mp3"] and os.path.exists(narr_state["last_mp3"]):
            data, sr = sf.read(narr_state["last_mp3"])
            sd.stop()
            sd.play(data, sr)

    def save_narr_output():
        if not (narr_state["last_mp3"] and os.path.exists(narr_state["last_mp3"])):
            return
        dest = filedialog.asksaveasfilename(
            defaultextension=".mp3",
            filetypes=[("MP3 Audio (160 kbps)", "*.mp3")]
        )
        if dest:
            shutil.copy2(narr_state["last_mp3"], dest)
            messagebox.showinfo("Saved", f"MP3 file was saved:\n{dest}")

    # Priradenie tlačidiel
    def reset_project_memory():
        v = video_entry.get().strip()
        if not v:
            messagebox.showinfo("Info", "Zadajte najprv video súbor.")
            return
        proj_dir = os.path.splitext(v)[0] + "_ai_dub"
        proj_f = os.path.join(proj_dir, "project.json")
        if not os.path.exists(proj_f):
            messagebox.showinfo("Info", f"No saved project exists yet for video '{os.path.basename(v)}'.")
            return
        if messagebox.askyesno("Clear Project Cache", f"Are you sure you want to clear saved analysis and settings for video:\n{os.path.basename(v)}?\n\n(On next run, character analysis and voices will be re-analyzed from scratch)"):
            try:
                os.remove(proj_f)
                dub_state["data"] = load_or_init_project(v, proj_f)
                refresh_speaker_table()
                refresh_subtitle_table()
                log(f"[Project] Project cache {os.path.basename(proj_f)} was cleared.")
                messagebox.showinfo("Done", "Project cache was successfully cleared.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete project file: {e}")

    reset_proj_btn.config(command=reset_project_memory)
    analyze_btn.config(command=lambda: threading.Thread(target=task_analyze, daemon=True).start())
    dub_btn.config(command=lambda: threading.Thread(target=task_dub, daemon=True).start())
    batch_dub_btn.config(command=lambda: threading.Thread(target=task_batch_dub, daemon=True).start())
    cancel_btn.config(command=lambda: dub_state.update({"cancel_requested": True}))
    export_work_btn.config(command=handle_export_work)
    import_work_btn.config(command=handle_import_work)

    narr_gen_btn.config(command=lambda: threading.Thread(target=task_generate_narration, daemon=True).start())
    narr_cancel_btn.config(command=lambda: narr_state.update({"cancel_requested": True}))
    narr_play_out_btn.config(command=play_narr_output)
    narr_save_btn.config(command=save_narr_output)

    # Inicializácia roletiek a počiatočného stavu
    update_tab1_voices_dropdown()
    update_spk_voices_dropdown()
    update_narr_voices_dropdown()
    parse_and_update_tab2_phrases(force_default=False)

    root.after(120, process_logs)
    return root

