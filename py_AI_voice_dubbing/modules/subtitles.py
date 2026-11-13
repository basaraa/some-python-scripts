from modules.config import *
from modules.audio_utils import format_timestamp_srt, parse_timestamp_srt

# ==============================================================================
# 4. SUBTITLE PARSING & TEXT UTILITIES
# ==============================================================================

def parse_srt_file(srt_path):
    with open(srt_path, 'r', encoding='utf-8-sig') as f:
        content = f.read().strip()
    return parse_srt_content(content)

def parse_srt_content(raw_srt_text):
    blocks = re.split(r'\n\s*\n', raw_srt_text.strip())
    subtitles = []
    for block in blocks:
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if len(lines) >= 2:
            time_line = lines[1] if '-->' in lines[1] else lines[0]
            text_lines = lines[2:] if '-->' in lines[1] else lines[1:]
            if '-->' in time_line:
                times = time_line.split('-->')
                start_sec = parse_timestamp_srt(times[0])
                end_sec = parse_timestamp_srt(times[1])
                subtitles.append({
                    "id": len(subtitles) + 1,
                    "start": start_sec,
                    "end": end_sec,
                    "text": " ".join(text_lines),
                    "speaker": "Speaker_01",
                    "assigned_voice": ""
                })
    return subtitles

def extract_clean_text_from_srt(srt_path_or_content):
    """
    Extracts clean dialogue text from SRT content or file path,
    removing sequence numbers, timestamps, and formatting tags (HTML/ASS),
    and intelligently merging incomplete sentence fragments split across subtitle cues.
    """
    if os.path.exists(srt_path_or_content):
        with open(srt_path_or_content, 'r', encoding='utf-8-sig', errors='replace') as f:
            raw_text = f.read()
    else:
        raw_text = srt_path_or_content

    raw_text = raw_text.replace('\r\n', '\n').replace('\r', '\n')
    blocks = re.split(r'\n\s*\n', raw_text.strip())
    raw_lines = []
    for block in blocks:
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if not lines:
            continue
        text_lines = []
        for l in lines:
            if l.isdigit():
                continue
            if '-->' in l:
                continue
            l_clean = re.sub(r'<[^>]+>', '', l)
            l_clean = re.sub(r'\{[^}]+\}', '', l_clean).strip()
            if l_clean:
                text_lines.append(l_clean)
        if text_lines:
            raw_lines.append(" ".join(text_lines))

    # Merge across subtitle boundaries when sentences are continued
    sentence_terminators = ('.', '!', '?', '. ', '! ', '? ', '…')
    merged_lines = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        if not merged_lines:
            merged_lines.append(line)
        else:
            prev = merged_lines[-1].strip()
            clean_prev = prev.rstrip('"\')]}»”’')
            ends_with_terminator = clean_prev.endswith(sentence_terminators)
            starts_with_lowercase = len(line) > 0 and line[0].islower()
            ends_with_continuation = clean_prev.endswith((',', ';', ':', '-', '–', '—'))

            if (not ends_with_terminator) or starts_with_lowercase or ends_with_continuation:
                if prev.endswith('-') and len(prev) > 1 and prev[-2].isalnum() and line[0].isalnum():
                    merged_lines[-1] = prev[:-1] + line
                else:
                    merged_lines[-1] = prev + " " + line
            else:
                merged_lines.append(line)

    return "\n".join(merged_lines)

def export_working_file(segments, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for seg in segments:
            v_tag = f" | {seg['assigned_voice']}" if seg.get("assigned_voice") else ""
            f.write(f"[{seg['speaker']}{v_tag}]\n")
            f.write(f"{format_timestamp_srt(seg['start'])} --> {format_timestamp_srt(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")

def import_working_file(working_path):
    with open(working_path, 'r', encoding='utf-8-sig') as f:
        content = f.read().strip()
    blocks = re.split(r'\n\s*\n', content)
    items = []
    for b in blocks:
        lines = [l.strip() for l in b.split('\n') if l.strip()]
        if len(lines) >= 3 and lines[0].startswith('[') and '-->' in lines[1]:
            header = lines[0].replace('[', '').replace(']', '').strip()
            spk = header
            assigned_voice = ""
            if '|' in header:
                parts = header.split('|', 1)
                spk = parts[0].strip()
                assigned_voice = parts[1].strip()

            times = lines[1].split('-->')
            items.append({
                "id": len(items) + 1,
                "speaker": spk,
                "assigned_voice": assigned_voice,
                "start": parse_timestamp_srt(times[0]),
                "end": parse_timestamp_srt(times[1]),
                "text": " ".join(lines[2:])
            })
    return items

def split_text_into_sentences_list(raw_text):
    """
    Splits text by punctuation (. ! ? 。 ！？ and line breaks) into discrete sentences for Tab 2 phrase table.
    """
    cleaned = raw_text.replace("\r\n", "\n")
    cleaned = re.sub(r'[„“”"«»「」]', '', cleaned)
    lines = [l.strip() for l in cleaned.split("\n") if l.strip()]
    sentences = []
    for line in lines:
        parts = re.findall(r'[^.!?。！？]+(?:[.!?。！？]+|$)', line)
        for p in parts:
            p_clean = p.strip()
            if p_clean and not re.fullmatch(r'^[.,!?:;\- 。！？、 ]+$', p_clean):
                if p_clean[-1] not in ".!?。！？":
                    p_clean = p_clean.rstrip(',:; -–—') + "."
                sentences.append(p_clean)
    return sentences

def sync_subtitles_with_disk(old_subs, new_subs, speakers):
    """
    Intelligently synchronizes subtitle lines loaded from .srt file with existing project state.
    Preserves assigned characters, voices, and audio_chunk_keys for both unchanged and modified lines.

    3-Phase 1-to-1 Matching:
      1. Exact text match + start time proximity < 10s (unchanged lines retain exact voice and cache)
      2. Timestamp slot alignment < 0.4s (handles edited text in the same scene slot)
      3. Proximity fallback < 1.5s (fallback for timing shifts)
    """
    if not old_subs:
        for s in new_subs:
            spk = s.get("speaker", "Speaker_01")
            spk_data = speakers.get(spk, {})
            st_v = spk_data.get("supertonic_voice", "M1")
            if not s.get("assigned_voice"):
                s["assigned_voice"] = f"⚡ Supertonic [{st_v}]"
        return new_subs

    def _norm(t):
        return " ".join(t.replace("\r", " ").replace("\n", " ").split()).lower()

    available = list(old_subs)  # 1-to-1 consumption pool
    synced = []

    for idx, new_s in enumerate(new_subs):
        ns_text = _norm(new_s.get("text", ""))
        ns_start = new_s["start"]
        matched_old = None

        # --- Fáza 1: rovnaký text + čas blízko (nezmenený riadok) ---
        best1_diff = 10.0
        best1_cand = None
        for cand in available:
            if _norm(cand.get("text", "")) == ns_text:
                diff = abs(ns_start - cand["start"])
                if diff < best1_diff:
                    best1_diff = diff
                    best1_cand = cand
        if best1_cand is not None:
            matched_old = best1_cand
            available.remove(best1_cand)

        # --- Fáza 2: rovnaký časový slot (editovaný text) ---
        if matched_old is None:
            best2_diff = 0.4
            best2_cand = None
            for cand in available:
                diff = abs(ns_start - cand["start"])
                if diff < best2_diff:
                    best2_diff = diff
                    best2_cand = cand
            if best2_cand is not None:
                matched_old = best2_cand
                available.remove(best2_cand)

        # --- Fáza 3: fallback – časový prekryv < 1.5 s ---
        if matched_old is None:
            best3_diff = 1.5
            best3_cand = None
            for cand in available:
                diff = abs(ns_start - cand["start"])
                if diff < best3_diff:
                    best3_diff = diff
                    best3_cand = cand
            if best3_cand is not None:
                matched_old = best3_cand
                available.remove(best3_cand)

        if matched_old:
            new_s["speaker"] = matched_old.get("speaker", "Speaker_01")
            new_s["assigned_voice"] = matched_old.get("assigned_voice", "")
            # Zachovaj audio_chunk_key pre cache reuse v process_dub
            if matched_old.get("audio_chunk_key"):
                new_s["audio_chunk_key"] = matched_old["audio_chunk_key"]
        else:
            new_s["speaker"] = "Speaker_01"

        if not new_s.get("assigned_voice"):
            spk_data = speakers.get(new_s["speaker"], {})
            st_v = spk_data.get("supertonic_voice", "M1")
            new_s["assigned_voice"] = f"⚡ Supertonic [{st_v}]"

        new_s["id"] = idx + 1
        synced.append(new_s)

    return synced

def is_batch_video_up_to_date(vid_path, sub_path, target_lang):
    """
    Checks whether a video has already been successfully dubbed and exported,
    and ensures the .srt subtitle file has not been modified since the export.
    """
    base_dir = os.path.splitext(vid_path)[0] + "_ai_dub"
    proj_file = os.path.join(base_dir, "project.json")
    if not os.path.exists(proj_file):
        return False

    try:
        with open(proj_file, "r", encoding="utf-8") as f:
            p_data = json.load(f)
    except Exception:
        return False

    if not p_data.get("steps_completed", {}).get("final_export", False):
        return False

    out_dir = os.path.join(base_dir, "output")
    if not os.path.exists(out_dir):
        return False

    dubbed_videos = [f for f in os.listdir(out_dir) if f.lower().endswith(('.mkv', '.mp4')) and '_dubbed_' in f.lower()]
    if not dubbed_videos:
        return False

    # Ak bol .srt súbor upravený po uložení project.json, video nie je aktuálne!
    if os.path.exists(sub_path):
        if os.path.getmtime(sub_path) > os.path.getmtime(proj_file):
            return False

    return True

def split_text_into_natural_sentences(raw_text):
    """
    Splits continuous text into natural sentence paragraphs for TTS processing.
    """
    cleaned = raw_text.replace("\r\n", "\n")
    cleaned = re.sub(r'[„“”"«»]', '', cleaned)
    cleaned = re.sub(r'[–—―]', ' - ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)

    paragraphs = [p.strip() for p in cleaned.split("\n") if p.strip()]
    final_chunks = []

    for p in paragraphs:
        sentences = re.findall(r'[^.!?]+(?:[.!?]+|$)', p)
        for s in sentences:
            s_clean = s.strip()
            if not s_clean or re.fullmatch(r'^[.,!?:;\- ]+$', s_clean):
                continue

            s_clean = re.sub(r'\.+', '.', s_clean)
            if len(s_clean) > 140:
                parts = re.split(r'([,;:\-]+)', s_clean)
                cur = ""
                for part in parts:
                    if len(cur) + len(part) <= 140:
                        cur += part
                    else:
                        c_strip = cur.strip()
                        if c_strip and not re.fullmatch(r'^[.,!?:;\- ]+$', c_strip):
                            final_chunks.append(c_strip)
                        cur = part
                c_strip = cur.strip()
                if c_strip and not re.fullmatch(r'^[.,!?:;\- ]+$', c_strip):
                    final_chunks.append(c_strip)
            else:
                final_chunks.append(s_clean)

        final_chunks.append("<PARAGRAPH_BREAK>")

    if final_chunks and final_chunks[-1] == "<PARAGRAPH_BREAK>":
        final_chunks.pop()

    sanitized_chunks = []
    for c in final_chunks:
        if c == "<PARAGRAPH_BREAK>":
            sanitized_chunks.append(c)
        else:
            c_final = c.strip()
            if c_final and c_final[-1] not in ".!?":
                c_final += "."
            sanitized_chunks.append(c_final)

    return sanitized_chunks


# ==============================================================================
# 7. PROJECT STATE & STORAGE MANAGER
# ==============================================================================

def create_project_structure(video_path):
    base_dir = os.path.splitext(video_path)[0] + "_ai_dub"
    folders = {
        "root": base_dir,
        "audio": os.path.join(base_dir, "audio"),
        "separation": os.path.join(base_dir, "separation"),
        "speakers": os.path.join(base_dir, "speakers"),
        "generated": os.path.join(base_dir, "generated"),
        "subtitles": os.path.join(base_dir, "subtitles"),
        "output": os.path.join(base_dir, "output")
    }
    for p in folders.values():
        os.makedirs(p, exist_ok=True)
    project_file = os.path.join(base_dir, "project.json")
    return folders, project_file

def load_or_init_project(video_path, project_file):
    if os.path.exists(project_file):
        with open(project_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "video_path": video_path,
        "mode": "EXISTING_SRT",
        "tts_engine": "Supertonic (Lokálne)",
        "target_lang": "cs",
        "use_separation": True,
        "speakers": {},
        "subtitles": [],
        "steps_completed": {}
    }

def save_project(project_data, project_file):
    with open(project_file, 'w', encoding='utf-8') as f:
        json.dump(project_data, f, indent=4, ensure_ascii=False)


def find_matching_subtitle(video_path, target_lang="cs"):
    """
    Finds a matching subtitle (.srt or .ass) for a video file.
    Priority:
      1. Exact language match (e.g. video.cs.srt, video.cz.srt, video.sk.srt, video.ja.srt)
      2. Exact base stem (e.g. video.srt, video.ass)
      3. Language/General aliases (e.g. video.en.srt, video.default.srt)
      4. Single subtitle file in directory if only one exists
    """
    if not video_path or not os.path.exists(video_path):
        return None

    folder = os.path.dirname(os.path.abspath(video_path))
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    sub_exts = ('.srt', '.ass')

    lang_code = target_lang.split()[0].lower() if target_lang else "cs"
    aliases = [lang_code]
    if lang_code in ("cs", "sk"):
        aliases.extend(["cz", "cze", "ces", "sk", "slo", "slk"])
    elif lang_code == "ja":
        aliases.extend(["jp", "jpn"])
    elif lang_code == "en":
        aliases.extend(["eng", "default"])
    aliases.extend(["default", "en", "eng"])

    # 1. Check language-specific suffixes
    for ext in sub_exts:
        for al in aliases:
            p = os.path.join(folder, f"{base_name}.{al}{ext}")
            if os.path.exists(p):
                return p
            p_dash = os.path.join(folder, f"{base_name}-{al}{ext}")
            if os.path.exists(p_dash):
                return p_dash

    # 2. Check exact base name (video.srt, video.ass)
    for ext in sub_exts:
        p = os.path.join(folder, f"{base_name}{ext}")
        if os.path.exists(p):
            return p

    # 3. Fuzzy prefix match in folder
    prefix = base_name.lower() + "."
    try:
        all_subs = [f for f in os.listdir(folder) if f.lower().endswith(sub_exts)]
        for f in all_subs:
            if f.lower().startswith(prefix):
                return os.path.join(folder, f)
        # If there is only exactly 1 subtitle file in the entire video directory, use it!
        if len(all_subs) == 1:
            return os.path.join(folder, all_subs[0])
    except Exception:
        pass

    return None

