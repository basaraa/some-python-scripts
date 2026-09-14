import os
import json
import shutil
import subprocess
import threading
import time
import base64
import re
import tempfile
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = ImageTk = None

VIDEO_EXTENSIONS = {".mkv", ".avi", ".mp4", ".mov", ".m4v", ".ts", ".m2ts", ".webm"}

# Normalization table for language codes (ISO 639-1, 639-2/B, 639-2/T and abbreviations)
LANG_MAP = {
    "cs": "cze", "ces": "cze", "cze": "cze", "cz": "cze", "czech": "cze",
    "sk": "slo", "slk": "slo", "slo": "slo", "slovak": "slo",
    "en": "eng", "eng": "eng", "english": "eng",
    "zh": "chi", "zho": "chi", "chi": "chi", "chinese": "chi",
    "de": "ger", "deu": "ger", "ger": "ger", "german": "ger",
    "fr": "fre", "fra": "fre", "fre": "fre", "french": "fre",
    "es": "spa", "spa": "spa", "spanish": "spa",
    "it": "ita", "ita": "ita", "italian": "ita",
    "ru": "rus", "rus": "rus", "russian": "rus",
    "pl": "pol", "pol": "pol", "polish": "pol",
    "hu": "hun", "hun": "hun", "hungarian": "hun",
    "uk": "ukr", "ukr": "ukr", "ukrainian": "ukr",
    "ja": "jpn", "jpn": "jpn", "japanese": "jpn",
    "ko": "kor", "kor": "kor", "korean": "kor",
    "pt": "por", "por": "por", "portuguese": "por",
    "nl": "dut", "nld": "dut", "dut": "dut", "dutch": "dut",
    "sv": "swe", "swe": "swe", "swedish": "swe",
    "da": "dan", "dan": "dan", "danish": "dan",
    "fi": "fin", "fin": "fin", "finnish": "fin",
    "no": "nor", "nor": "nor", "nob": "nor", "nno": "nor", "norwegian": "nor",
    "tr": "tur", "tur": "tur", "turkish": "tur",
    "ro": "rum", "ron": "rum", "rum": "rum", "romanian": "rum",
    "hr": "hrv", "hrv": "hrv", "croatian": "hrv",
    "sr": "srp", "srp": "srp", "serbian": "srp",
    "bg": "bul", "bul": "bul", "bulgarian": "bul",
    "el": "gre", "ell": "gre", "gre": "gre", "greek": "gre",
    "ar": "ara", "ara": "ara", "arabic": "ara",
}


def norm_lang(code):
    if not code:
        return "und"
    c = str(code).strip().lower().split("-")[0].split("_")[0]
    return LANG_MAP.get(c, c or "und")


def tool_exists(name):
    return shutil.which(name) is not None


def get_video_files(folder="."):
    return sorted(f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS)


def ffprobe_json(file):
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", file]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        raise RuntimeError(r.stderr.strip() or f"ffprobe failed: {file}")
    return json.loads(r.stdout)


def parse_fps(value):
    try:
        if "/" in str(value):
            a, b = str(value).split("/", 1)
            return float(a) / float(b) if float(b) else 0.0
        return float(value)
    except Exception:
        return 0.0


def get_media_info(file):
    data = ffprobe_json(file)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if not video:
        raise RuntimeError("Video stream not found")
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0)

    # Detection of HDR vs SDR
    color_trc = (video.get("color_transfer") or video.get("color_trc") or "").lower()
    color_primaries = (video.get("color_primaries") or "").lower()
    side_data = video.get("side_data_list", [])
    has_dovi = any("dovi" in str(sd.get("side_data_type", "")).lower() for sd in side_data)
    has_hdr_meta = any("mastering display" in str(sd.get("side_data_type", "")).lower() or 
                       "content light level" in str(sd.get("side_data_type", "")).lower() for sd in side_data)
    
    is_hdr = (
        color_trc in ("smpte2084", "arib-std-b67") or 
        has_dovi or 
        has_hdr_meta or 
        (color_primaries == "bt2020" and "10" in str(video.get("pix_fmt", "")) and color_trc not in ("bt709", "smpte170m", "bt470bg"))
    )

    return {
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "duration": duration,
        "fps": parse_fps(video.get("r_frame_rate", "0/1")),
        "video_codec": video.get("codec_name", ""),
        "pix_fmt": video.get("pix_fmt", ""),
        "audio_codec": audio.get("codec_name", "") if audio else "",
        "is_hdr": is_hdr,
        "hdr_type": "HDR" if is_hdr else "SDR",
    }


def supports_nvenc():
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        return "hevc_nvenc" in r.stdout
    except Exception:
        return False


def format_time(seconds, ms=True):
    seconds = max(0.0, float(seconds or 0))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if ms:
        milli = int(round((seconds - int(seconds)) * 1000))
        if milli >= 1000:
            s += 1
            milli = 0
        if s >= 60:
            s = 0
            m += 1
        if m >= 60:
            m = 0
            h += 1
        return f"{h:02d}:{m:02d}:{s:02d}.{milli:03d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_time(text):
    text = str(text).strip().replace(",", ".")
    if not text:
        return 0.0
    if ":" in text:
        p = text.split(":")
        if len(p) == 3:
            return float(p[0]) * 3600 + float(p[1]) * 60 + float(p[2])
        if len(p) == 2:
            return float(p[0]) * 60 + float(p[1])
    return float(text)


def even(n):
    n = int(n)
    return n if n % 2 == 0 else n - 1


def clamp_crop(info, vals):
    w, h = info["width"], info["height"]
    unit = vals.get("mode", "pixels")
    def cv(v, base):
        return round(base * float(v) / 100.0) if unit == "percent" else int(round(float(v)))
    left = max(0, cv(vals.get("left", 0), w))
    right = max(0, cv(vals.get("right", 0), w))
    top = max(0, cv(vals.get("top", 0), h))
    bottom = max(0, cv(vals.get("bottom", 0), h))
    if left + right >= w - 2:
        right = max(0, w - left - 2)
    if top + bottom >= h - 2:
        bottom = max(0, h - top - 2)
    x, y = left, top
    cw, ch = even(w - left - right), even(h - top - bottom)
    if cw < 2: cw = 2
    if ch < 2: ch = 2
    return x, y, cw, ch


def get_crop_for_file(file, info, options):
    if file in options.get("crop_overrides", {}):
        v = options["crop_overrides"][file]
        x, y, w, h = clamp_crop(info, v)
        return {"enabled": True, "x": x, "y": y, "w": w, "h": h}
    if options.get("crop_enabled"):
        x, y, w, h = clamp_crop(info, options["crop_global"])
        return {"enabled": True, "x": x, "y": y, "w": w, "h": h}
    return {"enabled": False, "x": 0, "y": 0, "w": info["width"], "h": info["height"]}


def get_cut_for_file(file, options):
    if file in options.get("cut_overrides", {}):
        return options["cut_overrides"][file]
    if options.get("cut_enabled"):
        return options["cut_global"]
    return {"enabled": False, "start": 0.0, "end": 0.0}


def get_target_resolution(info, options):
    return tuple(options.get("resolution_profiles", {}).get(
        f"{info['width']}x{info['height']}", (info["width"], info["height"])
    ))


def quality_encoder_args(options):
    q = int(options.get("quality", 27))
    encoder = options.get("encoder", "auto")
    if encoder in ("auto", "nvenc") and supports_nvenc():
        return ["-c:v", "hevc_nvenc", "-preset", options.get("nvenc_preset", "p5"),
                "-rc", "vbr", "-cq", str(q), "-b:v", "0"]
    return ["-c:v", "libx265", "-preset", "medium", "-crf", str(q)]


# ---------------- SUBTITLE HELPERS (WEBVTT -> SRT) ----------------
def vtt_text_to_srt(vtt_content):
    lines = vtt_content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out = []
    idx = 1
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "-->" in line:
            break
        i += 1

    def fmt_ts(ts):
        ts = ts.replace(".", ",")
        p = ts.split(":")
        if len(p) == 2:
            return f"00:{p[0].zfill(2)}:{p[1]}"
        elif len(p) == 3:
            return f"{p[0].zfill(2)}:{p[1].zfill(2)}:{p[2]}"
        return ts

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if "-->" not in line and i + 1 < len(lines) and "-->" in lines[i + 1]:
            i += 1
            line = lines[i].strip()

        if "-->" in line:
            parts = line.split("-->")
            if len(parts) == 2:
                start_str = parts[0].strip().split()[0]
                end_str = parts[1].strip().split()[0]
                timing = f"{fmt_ts(start_str)} --> {fmt_ts(end_str)}"
                i += 1
                text_lines = []
                while i < len(lines) and lines[i].strip():
                    t_line = re.sub(r"<[^>]+>", "", lines[i])
                    text_lines.append(t_line)
                    i += 1
                if text_lines:
                    out.append(f"{idx}\n{timing}\n" + "\n".join(text_lines) + "\n")
                    idx += 1
                continue
        i += 1
    return "\n".join(out)


def parse_mkv_subtitles_ebml(mkv_path):
    tracks = {}
    timecode_scale = 1000000
    cluster_time = 0
    try:
        with open(mkv_path, "rb") as f:
            f_size = os.path.getsize(mkv_path)

            def read_id():
                b = f.read(1)
                if not b: return None
                b0 = b[0]
                if b0 == 0: return None
                mask = 0x80
                l = 1
                while not (b0 & mask):
                    mask >>= 1
                    l += 1
                if l > 4: return None
                val = b0
                if l > 1:
                    rest = f.read(l - 1)
                    if len(rest) != l - 1: return None
                    for r in rest: val = (val << 8) | r
                return val

            def read_len():
                b = f.read(1)
                if not b: return None
                b0 = b[0]
                if b0 == 0: return None
                mask = 0x80
                l = 1
                while not (b0 & mask):
                    mask >>= 1
                    l += 1
                val = b0 & (mask - 1)
                if l > 1:
                    rest = f.read(l - 1)
                    if len(rest) != l - 1: return None
                    for r in rest: val = (val << 8) | r
                if val == (1 << (7 * l)) - 1:
                    return -1
                return val

            curr_track_num = None
            while f.tell() < f_size:
                eid = read_id()
                if eid is None: break
                elen = read_len()
                if elen is None: break

                if eid in (0x1A45DFA3, 0x18538067, 0x1549A966, 0x1654AE6B, 0xAE, 0x1F43B675, 0xA0):
                    continue

                if elen < 0:
                    break

                data_pos = f.tell()
                if eid == 0x2AD7B1:
                    raw = f.read(elen)
                    timecode_scale = int.from_bytes(raw, "big") if raw else 1000000
                elif eid == 0xD7:
                    raw = f.read(elen)
                    curr_track_num = int.from_bytes(raw, "big") if raw else 0
                    if curr_track_num not in tracks:
                        tracks[curr_track_num] = {'codec': '', 'lang': 'und', 'title': '', 'cues': []}
                elif eid == 0x86:
                    raw = f.read(elen).decode("utf-8", errors="replace").strip()
                    if curr_track_num in tracks:
                        tracks[curr_track_num]['codec'] = raw
                elif eid in (0x22B59C, 0x22B59D):
                    raw = f.read(elen).decode("utf-8", errors="replace").strip().lower()
                    if curr_track_num in tracks:
                        tracks[curr_track_num]['lang'] = norm_lang(raw)
                elif eid == 0x536E:
                    raw = f.read(elen).decode("utf-8", errors="replace").strip()
                    if curr_track_num in tracks:
                        tracks[curr_track_num]['title'] = raw
                elif eid == 0xE7:
                    raw = f.read(elen)
                    cluster_time = int.from_bytes(raw, "big") if raw else 0
                elif eid in (0xA1, 0xA3):
                    raw = f.read(elen)
                    if len(raw) > 3:
                        b0 = raw[0]
                        if b0 != 0:
                            mask = 0x80
                            vl = 1
                            while not (b0 & mask):
                                mask >>= 1
                                vl += 1
                            t_num = b0 & (mask - 1)
                            for k in range(1, vl):
                                t_num = (t_num << 8) | raw[k]
                            if t_num in tracks and ("WEBVTT" in tracks[t_num]['codec'].upper() or "UTF8" in tracks[t_num]['codec'].upper() or not tracks[t_num]['codec']):
                                if len(raw) >= vl + 3:
                                    rel_time = int.from_bytes(raw[vl:vl+2], "big", signed=True)
                                    payload = raw[vl+3:].decode("utf-8", errors="replace").strip()
                                    if payload:
                                        t_scale_ms = timecode_scale / 1000000.0
                                        start_ms = int((cluster_time + rel_time) * t_scale_ms)
                                        tracks[t_num]['cues'].append((start_ms, payload))
                else:
                    f.seek(data_pos + elen)
    except Exception:
        pass
    return tracks


def cues_to_srt(cues):
    cues.sort(key=lambda x: x[0])
    srt_blocks = []

    def fmt_time(ms):
        ms = max(0, ms)
        h = ms // 3600000
        ms %= 3600000
        m = ms // 60000
        ms %= 60000
        s = ms // 1000
        ms %= 1000
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    for i, (start_ms, text) in enumerate(cues, 1):
        if "-->" in text:
            vtt_block = vtt_text_to_srt(f"WEBVTT\n\n{text}")
            if vtt_block.strip():
                srt_blocks.append(vtt_block.strip())
                continue
        dur = min(4000, max(1200, cues[i][0] - start_ms)) if i < len(cues) else 3000
        clean_text = re.sub(r"<[^>]+>", "", text).strip()
        srt_blocks.append(f"{i}\n{fmt_time(start_ms)} --> {fmt_time(start_ms + dur)}\n{clean_text}\n")
    return "\n\n".join(srt_blocks)


def extract_webvtt_as_srt(mkv_file, stream_index, stream_lang="und", sub_track_index=None):
    """Safely extracts WebVTT track to SRT without language mismatch risk."""
    target_lang = norm_lang(stream_lang)

    # Method 1: mkvextract
    if tool_exists("mkvextract"):
        try_ids = [stream_index] if sub_track_index is None else [stream_index, sub_track_index]
        for tid in try_ids:
            try:
                with tempfile.TemporaryDirectory() as td:
                    tmp_vtt = os.path.join(td, "sub.vtt")
                    subprocess.run(["mkvextract", "tracks", mkv_file, f"{tid}:{tmp_vtt}"],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                    if os.path.exists(tmp_vtt) and os.path.getsize(tmp_vtt) > 0:
                        with open(tmp_vtt, "r", encoding="utf-8", errors="replace") as vf:
                            res = vtt_text_to_srt(vf.read())
                            if res.strip():
                                return res
            except Exception:
                pass

    # Method 2: Python EBML parser fallback
    tracks = parse_mkv_subtitles_ebml(mkv_file)
    
    # 1. Search for exact match by normalized language
    matched = [t for t in tracks.values() if t.get('cues') and norm_lang(t.get('lang')) == target_lang]
    if matched:
        return cues_to_srt(matched[0]['cues'])
    
    # 2. If there is only one subtitle track in the file, use it
    cues_tracks = [t for t in tracks.values() if t.get('cues')]
    if len(cues_tracks) == 1 and (target_lang == "und" or norm_lang(cues_tracks[0].get('lang')) in (target_lang, "und")):
        return cues_to_srt(cues_tracks[0]['cues'])

    return ""


def detect_unknown_subtitle_codecs(file):
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "verbose", "-analyzeduration", "200M",
             "-probesize", "200M", "-i", file, "-map", "0:s?", "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace"
        )
        text = r.stderr
    except Exception:
        return {}

    result = {}
    pending_codec = None
    for line in text.splitlines():
        m = re.search(r"Unknown/unsupported AVCodecID\s+([^\s]+)", line)
        if m:
            pending_codec = m.group(1).strip()
            continue
        m = re.search(r"Could not find codec parameters for stream\s+(\d+)", line)
        if m and pending_codec:
            idx = int(m.group(1))
            mapping = {
                "S_TEXT/WEBVTT": "webvtt",
                "S_TEXT/UTF8": "text",
                "S_TEXT/ASCII": "text",
                "S_TEXT/SSA": "ssa",
                "S_TEXT/ASS": "ass",
                "S_VOBSUB": "dvd_subtitle",
                "S_HDMV/PGS": "hdmv_pgs_subtitle",
                "S_DVBSUB": "dvb_subtitle",
            }
            if pending_codec in mapping:
                result[idx] = mapping[pending_codec]
            pending_codec = None
    return result


def subtitle_streams(file):
    data = ffprobe_json(file)
    out = []
    for s in data.get("streams", []):
        if s.get("codec_type") != "subtitle":
            continue
        codec = (s.get("codec_name") or "none").lower()
        tags = s.get("tags", {}) or {}
        raw_lang = tags.get("language") or "und"
        out.append({
            "index": s.get("index"),
            "codec": codec,
            "language": norm_lang(raw_lang),
            "orig_lang": raw_lang,
            "title": tags.get("title", ""),
            "default": int((s.get("disposition", {}) or {}).get("default", 0) or 0),
        })
    return out


def subtitle_languages(file):
    return sorted({x["language"] for x in subtitle_streams(file)})


def build_video_filter(file, info, options):
    filters = []
    cut = get_cut_for_file(file, options)
    crop = get_crop_for_file(file, info, options)
    target_w, target_h = get_target_resolution(info, options)
    
    if cut["enabled"]:
        filters += [f"trim=start={cut['start']:.6f}:end={cut['end']:.6f}", "setpts=PTS-STARTPTS"]
    if crop["enabled"]:
        filters.append(f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']}")
    profile_is_original = (target_w, target_h) == (info["width"], info["height"])
    if (target_w, target_h) != (crop["w"], crop["h"]) and not (crop["enabled"] and profile_is_original):
        filters.append(f"scale={target_w}:{target_h}:flags=lanczos")
        
    # HDR -> SDR Tonemapping (BT.709)
    if options.get("remove_hdr") and info.get("is_hdr"):
        filters.append("libplacebo=tonemapping=bt.2390:colorspace=bt709:color_primaries=bt709:color_trc=bt709:format=yuv420p")
        
    return ",".join(filters) if filters else None


def build_ffmpeg_command(file, options, info, extra_converted_srts=None):
    base = os.path.splitext(file)[0]
    ext = options.get("container_format", ".mkv")
    output = f"{base}_processed{ext}"
    if os.path.abspath(output) == os.path.abspath(file):
        output = f"{base}_processed.mkv"

    extra_srts = extra_converted_srts or []
    cmd = ["ffmpeg", "-hide_banner", "-analyzeduration", "200M", "-probesize", "200M", "-i", file]

    for srt_path, _ in extra_srts:
        cmd += ["-i", srt_path]

    external_srt = base + ".srt" if options.get("subtitle_insert") else None
    if external_srt and os.path.exists(external_srt):
        cmd += ["-i", external_srt]

    vf = build_video_filter(file, info, options)
    crop = get_crop_for_file(file, info, options)
    cut = get_cut_for_file(file, options)
    target = get_target_resolution(info, options)
    hdr_convert = options.get("remove_hdr") and info.get("is_hdr")
    video_encode = options.get("video_convert") or crop["enabled"] or cut["enabled"] or target != (info["width"], info["height"]) or hdr_convert

    cmd += ["-map", "0:v:0"]
    if video_encode:
        if vf:
            cmd += ["-vf", vf]
        cmd += quality_encoder_args(options)
        cmd += ["-pix_fmt", "yuv420p"]
        if options.get("remove_hdr") and info.get("is_hdr"):
            cmd += ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
    else:
        cmd += ["-c:v", "copy"]

    tracks = str(options.get("audio_tracks_to_keep", "")).strip()
    if tracks:
        for t in tracks.split(","):
            t = t.strip()
            if t.isdigit():
                cmd += ["-map", f"0:a:{int(t)-1}?"]
    else:
        cmd += ["-map", "0:a?"]

    if options.get("audio_convert") and info.get("audio_codec", "").lower() == "eac3":
        cmd += ["-c:a", "aac", "-b:a", "160k"]
    else:
        cmd += ["-c:a", "copy"]

    keep_subs = options.get("subtitle_keep", "all")
    subtitle_list = subtitle_streams(file)
    converted_indices = {s_info["index"] for _, s_info in extra_srts}

    if keep_subs == "all":
        selected_subs = [s for s in subtitle_list if s["index"] not in converted_indices and s["codec"] != "none"]
    elif keep_subs == "selected":
        selected_languages = {norm_lang(l) for l in options.get("subtitle_languages", [])}
        selected_subs = [s for s in subtitle_list if norm_lang(s["language"]) in selected_languages and s["index"] not in converted_indices and s["codec"] != "none"]
    else:
        selected_subs = []

    out_sub_idx = 0
    for s in selected_subs:
        cmd += ["-map", f"0:{s['index']}"]
        if ext.lower() in (".mkv", ".webm"):
            cmd += [f"-c:s:{out_sub_idx}", "copy"]
        else:
            cmd += [f"-c:s:{out_sub_idx}", "mov_text"]
        out_sub_idx += 1

    curr_input = 1
    for srt_path, s_info in extra_srts:
        cmd += ["-map", f"{curr_input}:0"]
        if ext.lower() in (".mkv", ".webm"):
            cmd += [f"-c:s:{out_sub_idx}", "subrip"]
        else:
            cmd += [f"-c:s:{out_sub_idx}", "mov_text"]
        cmd += [f"-metadata:s:s:{out_sub_idx}", f"language={norm_lang(s_info.get('language', 'und'))}"]
        if s_info.get("title"):
            cmd += [f"-metadata:s:s:{out_sub_idx}", f"title={s_info.get('title')}"]
        out_sub_idx += 1
        curr_input += 1

    if external_srt and os.path.exists(external_srt):
        cmd += ["-map", f"{curr_input}:0"]
        if ext.lower() in (".mkv", ".webm"):
            cmd += [f"-c:s:{out_sub_idx}", "subrip"]
        else:
            cmd += [f"-c:s:{out_sub_idx}", "mov_text"]
        out_sub_idx += 1
        curr_input += 1

    if ext.lower() in (".mkv", ".webm"):
        cmd += ["-map", "0:t?", "-c:t", "copy"]

    cmd += ["-map_metadata", "0", "-y", output]
    return cmd


def extract_frame_png(file, seconds, width=900):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{max(0, seconds):.3f}",
           "-i", file, "-frames:v", "1", "-vf", f"scale={width}:-2", "-f", "image2pipe", "-vcodec", "png", "pipe:1"]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode or not r.stdout:
        raise RuntimeError(r.stderr.decode("utf-8", errors="replace").strip() or "Preview failed.")
    return base64.b64encode(r.stdout).decode("ascii")


class VideoBatchProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Batch Processor — Crop / Cut / Resolution / HDR")
        self.root.geometry("1350x900")
        self.root.minsize(1100, 760)
        self.files, self.info = [], {}
        self.crop_overrides, self.cut_overrides = {}, {}
        self.subtitle_language_vars = {}
        self.preview_photo = None
        self.preview_seconds = 0.0
        self.preview_file = None
        self.preview_job = None
        self.preview_request_id = 0
        self.crop_drag_start_pos = None
        self.playing = False
        self.play_stop = threading.Event()
        self.play_thread = None
        self.play_fps = 25.0
        self.cut_slider_dragging = False
        self.crop_slider_dragging = False
        self.slider_programmatic = False
        self.resolution_vars = {}

        self.vars = {
            "audio_convert": tk.BooleanVar(value=False),
            "video_convert": tk.BooleanVar(value=False),
            "remove_hdr": tk.BooleanVar(value=False),
            "subtitle_insert": tk.BooleanVar(value=False),
            "subtitle_mode": tk.StringVar(value="add"),
            "audio_tracks_to_keep": tk.StringVar(value=""),
            "export_audio": tk.BooleanVar(value=False),
            "container_format": tk.StringVar(value=".mkv"),
            "extract_subs": tk.BooleanVar(value=False),
            "crop_enabled": tk.BooleanVar(value=False),
            "cut_enabled": tk.BooleanVar(value=False),
            "crop_mode": tk.StringVar(value="pixels"),
            "crop_left": tk.StringVar(value="0"),
            "crop_right": tk.StringVar(value="0"),
            "crop_top": tk.StringVar(value="0"),
            "crop_bottom": tk.StringVar(value="0"),
            "cut_start": tk.StringVar(value="00:00:00"),
            "cut_end": tk.StringVar(value="00:00:00"),
            "quality": tk.IntVar(value=27),
            "encoder": tk.StringVar(value="auto"),
            "nvenc_preset": tk.StringVar(value="p5"),
            "subtitle_keep_mode": tk.StringVar(value="all"),
        }
        self.build_ui()
        self.refresh_files()

    def selected_file(self):
        sel = self.file_list.curselection()
        if not sel:
            return None
        idx = sel[0] // 2
        return self.files[idx] if idx < len(self.files) else None

    def selected_info(self):
        f = self.selected_file()
        return self.info.get(f) if f else None

    def refresh_files(self):
        self.files = get_video_files()
        self.info = {}
        for f in self.files:
            try: self.info[f] = get_media_info(f)
            except Exception: pass
        self.refresh_list()
        self.build_resolution_profiles()
        self.update_subtitle_list()
        if self.files:
            self.file_list.selection_set(0)
            self.on_file_selected()
        self.update_status()

    def refresh_list(self):
        self.file_list.delete(0, "end")
        for f in self.files:
            i = self.info.get(f, {})
            self.file_list.insert("end", f"▶ {os.path.basename(f)}")
            
            if i:
                flags = []
                if f in self.crop_overrides: flags.append("crop")
                if f in self.cut_overrides: flags.append("cut")
                suffix = f"  [{', '.join(flags)}]" if flags else ""
                hdr_tag = i.get("hdr_type", "SDR")
                self.file_list.insert("end", f"   └ {i['width']}×{i['height']} | {format_time(i['duration'], False)} | {hdr_tag}{suffix}")
            else:
                self.file_list.insert("end", "   └ loading info...")

    def current_options_preview(self):
        return {"crop_overrides": self.crop_overrides, "cut_overrides": self.cut_overrides,
                "crop_enabled": self.vars["crop_enabled"].get(), "cut_enabled": self.vars["cut_enabled"].get(),
                "crop_global": self.read_crop_fields(False),
                "cut_global": {"enabled": self.vars["cut_enabled"].get(), "start": 0, "end": 0}}

    def add_files(self):
        paths = filedialog.askopenfilenames(filetypes=[("Video", "*.mkv *.mp4 *.avi *.mov *.m4v *.ts *.m2ts *.webm"), ("All", "*.*")])
        if not paths: return
        self.files = list(paths); self.info = {}
        for f in self.files:
            try: self.info[f] = get_media_info(f)
            except Exception as e: messagebox.showerror("FFprobe", f"{f}\n\n{e}")
        self.refresh_list(); self.build_resolution_profiles(); self.update_subtitle_list()
        if self.files: self.file_list.selection_set(0); self.on_file_selected()

    def on_file_selected(self, event=None):
        if self.playing:
            self.stop_playback()
        f = self.selected_file()
        if not f or f not in self.info: return
        self.preview_file = f
        info = self.info[f]
        if f in self.crop_overrides:
            self.set_crop_fields(self.crop_overrides[f])
        else:
            self.set_crop_fields({"mode": self.vars["crop_mode"].get(), "left": 0, "right": 0, "top": 0, "bottom": 0})
        if f in self.cut_overrides:
            c = self.cut_overrides[f]
            self.vars["cut_start"].set(format_time(c["start"], False)); self.vars["cut_end"].set(format_time(c["end"], False))
        else:
            self.vars["cut_start"].set("00:00:00"); self.vars["cut_end"].set(format_time(info["duration"], False))
        self.cut_scale.configure(to=max(.01, info["duration"]))
        self.crop_scale.configure(to=max(.01, info["duration"]))
        self.cut_scale.set(0); self.crop_scale.set(0)
        self.preview_seconds = 0
        self.load_preview(0)
        self.update_cut_labels()
        self.update_crop_result()

    # ---------------- UI ----------------
    def build_ui(self):
        top = ttk.Frame(self.root, padding=8); top.pack(fill="x")
        ttk.Button(top, text="Refresh videos", command=self.refresh_files).pack(side="left")
        ttk.Button(top, text="Add videos...", command=self.add_files).pack(side="left", padx=6)
        self.status_var = tk.StringVar(); ttk.Label(top, textvariable=self.status_var).pack(side="right")
        paned = ttk.PanedWindow(self.root, orient="horizontal"); paned.pack(fill="both", expand=True, padx=8, pady=(0,8))
        left = ttk.Frame(paned, padding=5); right = ttk.Frame(paned, padding=5); paned.add(left, weight=2); paned.add(right, weight=5)
        ttk.Label(left, text="Videos", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        lf=ttk.Frame(left); lf.pack(fill="both", expand=True)
        self.file_list=tk.Listbox(lf, exportselection=False); sb=ttk.Scrollbar(lf, command=self.file_list.yview); self.file_list.configure(yscrollcommand=sb.set)
        self.file_list.pack(side="left", fill="both", expand=True); sb.pack(side="right", fill="y"); self.file_list.bind("<<ListboxSelect>>", self.on_file_selected)
        self.notebook=ttk.Notebook(right); self.notebook.pack(fill="both", expand=True)
        self.build_operations_tab(); self.build_crop_tab(); self.build_cut_tab(); self.build_resolution_tab(); self.build_subtitle_tab()
        bottom=ttk.Frame(self.root,padding=8); bottom.pack(fill="x")
        self.run_btn=ttk.Button(bottom,text="START PROCESSING",command=self.start_processing); self.run_btn.pack(side="right",ipadx=20,ipady=6)
        ttk.Label(bottom,text="Operations are optional and can be combined.").pack(side="left")

    def build_operations_tab(self):
        tab=ttk.Frame(self.notebook,padding=12); self.notebook.add(tab,text="Operations")
        ttk.Label(tab,text="Operations to perform",font=("TkDefaultFont",11,"bold")).pack(anchor="w")
        ttk.Checkbutton(tab,text="Convert audio to AAC (EAC3 → AAC)",variable=self.vars["audio_convert"]).pack(anchor="w",pady=3)
        ttk.Checkbutton(tab,text="Convert video to HEVC",variable=self.vars["video_convert"]).pack(anchor="w",pady=3)
        ttk.Checkbutton(tab,text="Remove HDR / Convert to SDR (Tonemapping)",variable=self.vars["remove_hdr"]).pack(anchor="w",pady=3)
        ttk.Checkbutton(tab,text="Use CROP",variable=self.vars["crop_enabled"]).pack(anchor="w",pady=3)
        ttk.Checkbutton(tab,text="Use CUT",variable=self.vars["cut_enabled"]).pack(anchor="w",pady=3)
        ttk.Checkbutton(tab,text="Insert external SRT",variable=self.vars["subtitle_insert"]).pack(anchor="w",pady=3)
        ttk.Checkbutton(tab,text="Export audio to .m4a",variable=self.vars["export_audio"]).pack(anchor="w",pady=3)
        ttk.Checkbutton(tab,text="Extract subtitles from MKV",variable=self.vars["extract_subs"]).pack(anchor="w",pady=3)
        row=ttk.Frame(tab); row.pack(anchor="w",pady=7); ttk.Label(row,text="Audio tracks (empty = all):").pack(side="left"); ttk.Entry(row,textvariable=self.vars["audio_tracks_to_keep"],width=18).pack(side="left",padx=6)
        row=ttk.Frame(tab); row.pack(anchor="w",pady=5); ttk.Label(row,text="Container:").pack(side="left"); ttk.OptionMenu(row,self.vars["container_format"],".mkv",".mkv",".mp4",".mov").pack(side="left",padx=6)
        ttk.Separator(tab).pack(fill="x",pady=15)
        ttk.Label(tab,text="HEVC Quality",font=("TkDefaultFont",11,"bold")).pack(anchor="w")
        q=ttk.Frame(tab); q.pack(fill="x",pady=8); ttk.Label(q,text="CQ / CRF (lower = higher quality):").pack(side="left")
        ttk.Scale(q,from_=20,to=32,variable=self.vars["quality"],orient="horizontal",length=300).pack(side="left",padx=8); ttk.Label(q,textvariable=self.vars["quality"],width=4).pack(side="left")
        er=ttk.Frame(tab); er.pack(anchor="w",pady=4); ttk.Label(er,text="Encoder:").pack(side="left"); ttk.OptionMenu(er,self.vars["encoder"],"auto","auto","nvenc","cpu (x265)").pack(side="left",padx=6); ttk.Label(er,text="NVENC preset:").pack(side="left",padx=(20,0)); ttk.OptionMenu(er,self.vars["nvenc_preset"],"p5","p1","p2","p3","p4","p5","p6","p7").pack(side="left",padx=6)
        ttk.Label(tab,text="Recommendation for smaller files: CPU/x265 + slow preset. NVENC is faster, but usually less efficient at the same visual quality.",wraplength=850,foreground="#555").pack(anchor="w",pady=15)

    def build_crop_tab(self):
        tab=ttk.Frame(self.notebook,padding=8); self.notebook.add(tab,text="CROP")
        top=ttk.Frame(tab); top.pack(fill="x")
        ttk.Label(top,text="Crop margins:").pack(side="left")
        ttk.OptionMenu(top,self.vars["crop_mode"],"pixels","pixels","percent").pack(side="left",padx=6)
        ttk.Label(top,text="(0 = none)").pack(side="left")
        fields=ttk.Frame(tab); fields.pack(fill="x",pady=7)
        for label,key in [("Left","crop_left"),("Right","crop_right"),("Top","crop_top"),("Bottom","crop_bottom")]:
            ttk.Label(fields,text=label).pack(side="left",padx=(4,2)); ttk.Entry(fields,textvariable=self.vars[key],width=8).pack(side="left",padx=(0,10))
        ttk.Button(fields,text="Apply globally to all",command=self.apply_global_crop).pack(side="left",padx=5)
        ttk.Button(fields,text="Save for this video only",command=self.save_crop_override).pack(side="left",padx=5)
        ttk.Button(fields,text="Clear crop for video",command=self.clear_crop_override).pack(side="left",padx=5)
        self.crop_result_var=tk.StringVar(); ttk.Label(tab,textvariable=self.crop_result_var,font=("TkDefaultFont",10,"bold")).pack(anchor="w",pady=3)
        ttk.Label(tab,text="Move the slider to any frame. The rectangle shows the cropped area.",foreground="#555").pack(anchor="w")
        self.crop_scale=ttk.Scale(tab,from_=0,to=1,orient="horizontal",command=self.on_crop_slider); self.crop_scale.pack(fill="x",pady=5)
        self.crop_scale.bind("<Button-1>", self.on_crop_scale_click)
        self.crop_time_label=ttk.Label(tab,text="00:00:00.000"); self.crop_time_label.pack(anchor="w")
        self.crop_canvas=tk.Canvas(tab,background="black",highlightthickness=1); self.crop_canvas.pack(fill="both",expand=True,pady=5)
        ttk.Button(tab,text="Refresh preview",command=lambda:self.load_preview(self.preview_seconds)).pack(anchor="w")
        for v in (self.vars["crop_left"],self.vars["crop_right"],self.vars["crop_top"],self.vars["crop_bottom"],self.vars["crop_mode"]): v.trace_add("write",lambda *a:self.update_crop_result())

    def build_cut_tab(self):
        tab=ttk.Frame(self.notebook,padding=8); self.notebook.add(tab,text="CUT")
        row=ttk.Frame(tab); row.pack(fill="x")
        ttk.Label(row,text="Start:").pack(side="left"); ttk.Entry(row,textvariable=self.vars["cut_start"],width=13).pack(side="left",padx=5); ttk.Label(row,text="End:").pack(side="left",padx=(12,0)); ttk.Entry(row,textvariable=self.vars["cut_end"],width=13).pack(side="left",padx=5)
        ttk.Button(row,text="Set globally",command=self.apply_global_cut).pack(side="left",padx=8); ttk.Button(row,text="Save for video",command=self.save_cut_override).pack(side="left")
        ttk.Label(tab,text="Time format is always HH:MM:SS. During playback, you can set Start/End directly at the current frame.",foreground="#555").pack(anchor="w",pady=5)
        self.cut_range_var=tk.StringVar(); ttk.Label(tab,textvariable=self.cut_range_var,font=("TkDefaultFont",10,"bold")).pack(anchor="w")
        self.cut_scale=ttk.Scale(tab,from_=0,to=1,orient="horizontal",command=self.on_cut_slider); self.cut_scale.pack(fill="x",pady=5)
        self.cut_scale.bind("<Button-1>", self.on_cut_scale_click)
        marker_row=ttk.Frame(tab); marker_row.pack(fill="x")
        self.cut_start_marker=ttk.Label(marker_row,text="Start: 00:00:00"); self.cut_start_marker.pack(side="left")
        self.cut_current_marker=ttk.Label(marker_row,text="Current: 00:00:00"); self.cut_current_marker.pack(side="left",expand=True)
        self.cut_end_marker=ttk.Label(marker_row,text="End: 00:00:00"); self.cut_end_marker.pack(side="right")
        self.cut_time_label=ttk.Label(tab,text="00:00:00.000"); self.cut_time_label.pack(anchor="w")
        self.cut_canvas=tk.Canvas(tab,background="black",highlightthickness=1); self.cut_canvas.pack(fill="both",expand=True,pady=5)
        buttons=ttk.Frame(tab); buttons.pack(fill="x",pady=5)
        self.play_btn=ttk.Button(buttons,text="▶ Play in window",command=self.toggle_play); self.play_btn.pack(side="left")
        ttk.Button(buttons,text="⏮ Frame to Start",command=self.preview_cut_start).pack(side="left",padx=5)
        ttk.Button(buttons,text="Frame to End ⏭",command=self.preview_cut_end).pack(side="left",padx=5)
        ttk.Button(buttons,text="Set START = current",command=self.set_cut_start_current).pack(side="left",padx=12)
        ttk.Button(buttons,text="Set END = current",command=self.set_cut_end_current).pack(side="left",padx=5)
        ttk.Button(buttons,text="Clear cut for video",command=self.clear_cut_override).pack(side="right")
        self.vars["cut_start"].trace_add("write",lambda *a:self.update_cut_labels()); self.vars["cut_end"].trace_add("write",lambda *a:self.update_cut_labels())

    def build_resolution_tab(self):
        tab=ttk.Frame(self.notebook,padding=12); self.notebook.add(tab,text="Resolution")
        ttk.Label(tab,text="Output resolution is always applied. Default = original resolution. Configured in batch by source size.",wraplength=900).pack(anchor="w")
        self.resolution_frame=ttk.Frame(tab); self.resolution_frame.pack(fill="both",expand=True,pady=8)
        self.build_resolution_profiles()

    def build_subtitle_tab(self):
        tab=ttk.Frame(self.notebook,padding=10); self.notebook.add(tab,text="Subtitles")
        ttk.Label(tab,text="Subtitle retention",font=("TkDefaultFont",11,"bold")).pack(anchor="w")
        ttk.Radiobutton(tab,text="All (default)",variable=self.vars["subtitle_keep_mode"],value="all").pack(anchor="w",pady=3)
        ttk.Radiobutton(tab,text="None",variable=self.vars["subtitle_keep_mode"],value="none").pack(anchor="w",pady=3)
        ttk.Radiobutton(tab,text="Selected languages (globally for all videos)",variable=self.vars["subtitle_keep_mode"],value="selected").pack(anchor="w",pady=3)
        ttk.Label(tab,text="Selection matches normalized language (e.g. cze/cs, slo/sk). WebVTT is converted without track mismatch.",foreground="#555",wraplength=850).pack(anchor="w",pady=5)
        self.subtitle_frame=ttk.Frame(tab); self.subtitle_frame.pack(fill="both",expand=True,pady=8)
        ttk.Button(tab,text="Refresh subtitle list",command=self.update_subtitle_list).pack(anchor="w")

    # ---------------- settings ----------------
    def set_crop_fields(self,v):
        self.vars["crop_mode"].set(v.get("mode","pixels")); self.vars["crop_left"].set(str(v.get("left",0))); self.vars["crop_right"].set(str(v.get("right",0))); self.vars["crop_top"].set(str(v.get("top",0))); self.vars["crop_bottom"].set(str(v.get("bottom",0)))

    def read_crop_fields(self, validate=True):
        v={"mode":self.vars["crop_mode"].get(),"left":float(self.vars["crop_left"].get()),"right":float(self.vars["crop_right"].get()),"top":float(self.vars["crop_top"].get()),"bottom":float(self.vars["crop_bottom"].get())}
        if validate and any(x<0 for x in (v["left"],v["right"],v["top"],v["bottom"])): raise ValueError("Crop values cannot be negative.")
        return v

    def apply_global_crop(self):
        try:
            self.read_crop_fields(); self.vars["crop_enabled"].set(True); self.update_crop_result(); self.refresh_list()
            messagebox.showinfo("Crop","This crop is set globally for all videos.")
        except Exception as e: messagebox.showerror("Crop",str(e))

    def save_crop_override(self):
        f=self.selected_file()
        if not f:return
        try:
            v=self.read_crop_fields(); self.crop_overrides[f]=v; self.refresh_list(); self.update_crop_result()
        except Exception as e: messagebox.showerror("Crop",str(e))

    def clear_crop_override(self):
        f=self.selected_file()
        if f: self.crop_overrides.pop(f,None); self.refresh_list(); self.on_file_selected()

    def update_crop_result(self):
        f=self.selected_file()
        if not f or f not in self.info:return
        try:
            v=self.read_crop_fields(False); x,y,w,h=clamp_crop(self.info[f],v); orig=self.info[f]
            self.crop_result_var.set(f"Original: {orig['width']}×{orig['height']}   →   cropped: {w}×{h}   |   cropped: left {x}px, right {orig['width']-x-w}px, top {y}px, bottom {orig['height']-y-h}px")
            self.draw_crop_overlay()
        except Exception: self.crop_result_var.set("Invalid crop values")

    # ---------------- cut ----------------
    def validate_cut(self):
        f=self.selected_file(); dur=self.info[f]["duration"] if f else 0
        s,e=parse_time(self.vars["cut_start"].get()),parse_time(self.vars["cut_end"].get())
        if s<0 or e<=s or e>dur+0.05: raise ValueError(f"Cut must be 0 ≤ Start < End ≤ {format_time(dur)}")
        return s,e

    def update_cut_labels(self):
        try:
            s=parse_time(self.vars["cut_start"].get()); e=parse_time(self.vars["cut_end"].get())
            self.cut_range_var.set(f"CUT: {format_time(s,False)}  →  {format_time(e,False)}   |   length: {format_time(max(0,e-s),False)}")
            if hasattr(self,"cut_start_marker"):
                self.cut_start_marker.configure(text=f"Start: {format_time(s,False)}")
                self.cut_end_marker.configure(text=f"End: {format_time(e,False)}")
                self.cut_current_marker.configure(text=f"Current: {format_time(self.preview_seconds,False)}")
        except Exception:
            self.cut_range_var.set("CUT: invalid time")

    def apply_global_cut(self):
        try:s,e=self.validate_cut(); self.vars["cut_enabled"].set(True); self.cut_range_var.set(f"GLOBAL CUT: {format_time(s,False)} → {format_time(e,False)} | length {format_time(e-s,False)}")
        except Exception as ex: messagebox.showerror("Cut",str(ex))

    def save_cut_override(self):
        f=self.selected_file()
        if not f:return
        try:s,e=self.validate_cut(); self.cut_overrides[f]={"enabled":True,"start":s,"end":e}; self.refresh_list(); self.update_cut_labels()
        except Exception as ex: messagebox.showerror("Cut",str(ex))

    def clear_cut_override(self):
        f=self.selected_file()
        if f:self.cut_overrides.pop(f,None); self.refresh_list(); self.on_file_selected()

    def preview_cut_start(self):
        try:t=parse_time(self.vars["cut_start"].get()); self.cut_scale.set(t); self.load_preview(t)
        except Exception as e:messagebox.showerror("Cut",str(e))

    def preview_cut_end(self):
        try:t=parse_time(self.vars["cut_end"].get()); self.cut_scale.set(t); self.load_preview(t)
        except Exception as e:messagebox.showerror("Cut",str(e))

    def _slider_click_time(self, widget, event, duration):
        try:
            width=max(1, widget.winfo_width())
            ratio=max(0.0, min(1.0, event.x / float(width)))
            return ratio * max(0.0, duration)
        except Exception:
            return None

    def on_cut_scale_click(self, event):
        f=self.selected_file()
        if not f or f not in self.info:
            return
        t=self._slider_click_time(self.cut_scale, event, self.info[f]["duration"])
        if t is None:
            return
        self.slider_programmatic=False
        self.cut_scale.set(t)
        self.on_cut_slider(t)
        return "break"

    def on_crop_scale_click(self, event):
        f=self.selected_file()
        if not f or f not in self.info:
            return
        t=self._slider_click_time(self.crop_scale, event, self.info[f]["duration"])
        if t is None:
            return
        self.slider_programmatic=False
        self.crop_scale.set(t)
        self.on_crop_slider(t)
        return "break"

    def on_cut_slider(self,v):
        t=float(v)
        self.preview_seconds=t
        self.cut_time_label.configure(text=format_time(t))
        self.update_cut_range_visual()
        if hasattr(self,"cut_current_marker"):
            self.cut_current_marker.configure(text=f"Current: {format_time(t,False)}")
        if not self.slider_programmatic and not self.playing:
            self.schedule_preview(t)

    def on_crop_slider(self,v):
        t=float(v)
        self.preview_seconds=t
        self.crop_time_label.configure(text=format_time(t))
        if not self.slider_programmatic and not self.playing:
            self.schedule_preview(t)

    def schedule_preview(self,t):
        if self.preview_job:
            try: self.root.after_cancel(self.preview_job)
            except Exception: pass
        self.preview_request_id += 1
        request_id = self.preview_request_id
        self.preview_job = self.root.after(140, self.load_preview, t, request_id)

    def set_cut_start_current(self):
        self.vars["cut_start"].set(format_time(self.preview_seconds,False)); self.update_cut_labels()

    def set_cut_end_current(self):
        self.vars["cut_end"].set(format_time(self.preview_seconds,False)); self.update_cut_labels()

    # ---------------- integrated playback ----------------
    def toggle_play(self):
        if self.playing:
            self.stop_playback(); return
        f=self.selected_file()
        if not f:return
        try:
            s=parse_time(self.vars["cut_start"].get()); e=parse_time(self.vars["cut_end"].get())
            if e<=s: raise ValueError("End must be greater than Start.")
            current=float(self.preview_seconds)
            if current < s or current >= e-0.02:
                play_from=s
            else:
                play_from=current
        except Exception as ex: messagebox.showerror("Playback",str(ex)); return
        self.playing=True; self.play_btn.configure(text="■ Stop"); self.play_stop.clear()
        self.play_thread=threading.Thread(target=self.playback_worker,args=(f,play_from,e),daemon=True); self.play_thread.start()

    def stop_playback(self):
        self.play_stop.set(); self.playing=False; self.play_btn.configure(text="▶ Play in window")

    def playback_worker(self,file,start,end):
        info=self.info[file]; fps=max(8,min(info.get("fps") or 25,60)); out_w=900
        out_h=max(2,even(round(info["height"]*out_w/info["width"])))
        cmd=["ffmpeg","-hide_banner","-loglevel","error","-ss",f"{start:.3f}","-i",file,"-t",f"{end-start:.3f}","-vf",f"scale={out_w}:{out_h}","-pix_fmt","rgb24","-f","rawvideo","pipe:1"]
        try:
            size=out_w*out_h*3
            p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=size)
            frame_time=1.0/fps; t=start; next_tick=time.monotonic()
            while not self.play_stop.is_set():
                data=p.stdout.read(size)
                if len(data)<size:
                    break
                current=t; t+=frame_time
                self.root.after(0,self.show_playback_frame,data,out_w,out_h,current)
                next_tick += frame_time
                delay=next_tick-time.monotonic()
                if delay>0:
                    time.sleep(delay)
                elif delay < -0.25:
                    next_tick=time.monotonic()
            try:
                p.terminate(); p.wait(timeout=0.5)
            except Exception:
                try:p.kill()
                except Exception:pass
        finally:
            self.root.after(0,self.stop_playback)

    def show_playback_frame(self,data,w,h,t):
        try:
            if Image is None or ImageTk is None:
                raise RuntimeError("Playback requires Pillow: pip install pillow")
            image=Image.frombytes("RGB",(w,h),data)
            self.preview_photo=ImageTk.PhotoImage(image=image)
            self.preview_seconds=t
            self.slider_programmatic=True
            try:
                self.cut_scale.set(t)
                self.crop_scale.set(t)
            finally:
                self.slider_programmatic=False
            self.cut_time_label.configure(text=format_time(t))
            self.crop_time_label.configure(text=format_time(t))
            self.update_cut_range_visual()
            self.draw_preview(self.preview_photo)
            self.draw_crop_overlay()
        except Exception as e:
            self.preview_error(str(e))

    # ---------------- preview ----------------
    def load_preview(self,seconds=0, request_id=None):
        f=self.selected_file()
        if not f:return
        dur=self.info[f]["duration"]; seconds=max(0,min(float(seconds),max(0,dur-.001))); self.preview_seconds=seconds; self.preview_file=f
        if request_id is None:
            self.preview_request_id += 1
            request_id = self.preview_request_id
        def worker():
            try:
                enc=extract_frame_png(f,seconds,900); self.root.after(0,self.show_preview,enc,seconds,f,request_id)
            except Exception as e:self.root.after(0,lambda:self.preview_error(str(e)))
        threading.Thread(target=worker,daemon=True).start()

    def show_preview(self,enc,seconds,f,request_id=None):
        if f!=self.selected_file():return
        if request_id is not None and request_id != self.preview_request_id:return
        try:self.preview_photo=tk.PhotoImage(data=enc)
        except Exception as e:self.preview_error(str(e));return
        self.preview_seconds=seconds; self.preview_file=f; self.draw_preview(self.preview_photo); self.cut_time_label.configure(text=format_time(seconds)); self.crop_time_label.configure(text=format_time(seconds)); self.cut_scale.set(seconds); self.crop_scale.set(seconds); self.draw_crop_overlay()

    def draw_preview(self,photo):
        for canvas in (self.cut_canvas,self.crop_canvas):
            canvas.delete("all"); canvas.update_idletasks(); cw=max(100,canvas.winfo_width()); ch=max(100,canvas.winfo_height()); pw,ph=photo.width(),photo.height(); x=max(0,(cw-pw)//2); y=max(0,(ch-ph)//2); canvas.create_image(x,y,anchor="nw",image=photo); canvas.image_ref=photo
        self.preview_box=self.get_preview_box(self.crop_canvas,photo)

    def get_preview_box(self,canvas,photo):
        cw=max(100,canvas.winfo_width()); ch=max(100,canvas.winfo_height()); return (max(0,(cw-photo.width())//2),max(0,(ch-photo.height())//2),max(0,(cw-photo.width())//2)+photo.width(),max(0,(ch-photo.height())//2)+photo.height())

    def draw_crop_overlay(self):
        if not hasattr(self,"crop_canvas") or not self.preview_photo:return
        self.draw_preview(self.preview_photo)
        f=self.selected_file(); info=self.info.get(f) if f else None
        if not info:return
        try:x,y,w,h=clamp_crop(info,self.read_crop_fields(False))
        except:return
        l,t,r,b=self.preview_box; sx=(r-l)/info["width"]; sy=(b-t)/info["height"]
        cx=l+x*sx; cy=t+y*sy; cr=cx+w*sx; cb=cy+h*sy
        self.crop_canvas.create_rectangle(cx,cy,cr,cb,outline="red",width=3)
        self.crop_canvas.create_text(cx+6,cy+6,text=f"{w}×{h}",anchor="nw",fill="yellow",font=("TkDefaultFont",10,"bold"))

    def update_cut_range_visual(self):
        self.update_cut_labels()

    def preview_error(self,text): self.cut_time_label.configure(text=f"Preview: {text}")

    # ---------------- resolution ----------------
    def build_resolution_profiles(self):
        old=self.resolution_vars; self.resolution_vars={}
        for w,h in sorted({(self.info[f]["width"],self.info[f]["height"]) for f in self.files if f in self.info}):
            key=f"{w}x{h}"; self.resolution_vars[key]=old.get(key,(tk.StringVar(value=str(w)),tk.StringVar(value=str(h))))
        if hasattr(self,"resolution_frame"):self.rebuild_resolution_widgets()

    def rebuild_resolution_widgets(self):
        for c in self.resolution_frame.winfo_children():c.destroy()
        h=ttk.Frame(self.resolution_frame);h.pack(fill="x"); ttk.Label(h,text="Original",width=18).pack(side="left");ttk.Label(h,text="Output W",width=15).pack(side="left");ttk.Label(h,text="Output H",width=15).pack(side="left")
        for key in sorted(self.resolution_vars,key=lambda k:tuple(map(int,k.split("x")))):
            wv,hv=self.resolution_vars[key]; r=ttk.Frame(self.resolution_frame);r.pack(fill="x",pady=3);ttk.Label(r,text=key,width=18).pack(side="left");ttk.Entry(r,textvariable=wv,width=12).pack(side="left",padx=5);ttk.Entry(r,textvariable=hv,width=12).pack(side="left",padx=5)
        ttk.Label(self.resolution_frame,text="E.g. 1920×1080 → 1920×1079 and 1280×720 → 1280×719 are two separate batch profiles.",foreground="#555").pack(anchor="w",pady=12)

    def get_resolution_profiles(self):
        out={}
        for key,(wv,hv) in self.resolution_vars.items():
            w,h=int(wv.get()),int(hv.get())
            if w<2 or h<2:raise ValueError(f"Invalid resolution: {key} → {w}x{h}")
            out[key]=(w,h)
        return out

    # ---------------- subtitles ----------------
    def collect_all_subtitle_languages(self):
        langs = set()
        for f in self.files:
            try:
                langs.update(subtitle_languages(f))
            except Exception:
                pass
        return sorted(langs)

    def update_subtitle_list(self):
        if not hasattr(self, "subtitle_frame"):
            return
        for c in self.subtitle_frame.winfo_children():
            c.destroy()
        langs = self.collect_all_subtitle_languages()
        if not langs:
            ttk.Label(self.subtitle_frame, text="No supported subtitle tracks found in the available videos.").pack(anchor="w")
            return

        ttk.Label(self.subtitle_frame, text="Selection is global based on normalized language (e.g. cze/cs, slo/sk).", foreground="#555").pack(anchor="w", pady=(0,8))
        for lang in langs:
            if lang not in self.subtitle_language_vars:
                self.subtitle_language_vars[lang] = tk.BooleanVar(value=False)
            var = self.subtitle_language_vars[lang]
            count = sum(1 for f in self.files for s in subtitle_streams(f) if s["language"] == lang)
            ttk.Checkbutton(self.subtitle_frame, text=f"{lang}  ({count} streams in batch)", variable=var).pack(anchor="w", pady=2)
        ttk.Button(self.subtitle_frame, text="Select all languages", command=lambda:self.set_all_subtitle_languages(True)).pack(anchor="w", pady=(10,2))
        ttk.Button(self.subtitle_frame, text="Deselect all languages", command=lambda:self.set_all_subtitle_languages(False)).pack(anchor="w")

    def set_all_subtitle_languages(self, value):
        for var in getattr(self, "subtitle_language_vars", {}).values():
            var.set(value)

    def get_selected_subtitle_languages(self):
        return {norm_lang(lang) for lang, var in getattr(self, "subtitle_language_vars", {}).items() if var.get()}

    # ---------------- processing ----------------
    def collect_options(self):
        profiles=self.get_resolution_profiles(); opts={k:v.get() for k,v in self.vars.items() if isinstance(v,tk.Variable)}
        opts["resolution_profiles"]=profiles; opts["crop_overrides"]=dict(self.crop_overrides); opts["cut_overrides"]=dict(self.cut_overrides)
        opts["subtitle_languages"]=sorted(self.get_selected_subtitle_languages()); opts["subtitle_keep"]=opts.pop("subtitle_keep_mode")
        opts["crop_global"]=self.read_crop_fields(True) if opts["crop_enabled"] else {"mode":"pixels","left":0,"right":0,"top":0,"bottom":0}
        if opts["cut_enabled"]:
            s=parse_time(opts["cut_start"]);e=parse_time(opts["cut_end"])
            if e<=s:raise ValueError("Global Cut: End must be greater than Start.")
            opts["cut_global"]={"enabled":True,"start":s,"end":e}
        else:opts["cut_global"]={"enabled":False,"start":0,"end":0}
        return opts

    def start_processing(self):
        if not self.files:messagebox.showwarning("Videos","No videos selected.");return
        try:opts=self.collect_options()
        except Exception as e:messagebox.showerror("Settings",str(e));return
        self.run_btn.configure(state="disabled");threading.Thread(target=self.process_worker,args=(opts,),daemon=True).start()

    def process_worker(self,opts):
        errors=[];done=0
        for n,f in enumerate(self.files,1):
            temp_files_to_cleanup = []
            try:
                self.root.after(0,lambda n=n,f=f:self.status_var.set(f"Processing {n}/{len(self.files)}: {os.path.basename(f)}"))
                info=self.info[f]
                unknown_decoders = detect_unknown_subtitle_codecs(f)
                all_subs = subtitle_streams(f)

                # Prepare clean .srt for WebVTT / S_TEXT/WEBVTT tracks
                converted_srts = []
                sub_track_idx = 0
                for s in all_subs:
                    is_vtt = (s["codec"] == "webvtt") or (s["codec"] == "none") or (unknown_decoders.get(s["index"]) == "webvtt")
                    if is_vtt and f.lower().endswith(".mkv"):
                        srt_text = extract_webvtt_as_srt(f, s["index"], s["language"], sub_track_index=sub_track_idx)
                        if srt_text:
                            tfd, tpath = tempfile.mkstemp(suffix=f"_{s['language']}.srt")
                            os.close(tfd)
                            with open(tpath, "w", encoding="utf-8") as srt_file:
                                srt_file.write(srt_text)
                            temp_files_to_cleanup.append(tpath)
                            converted_srts.append((tpath, s))
                    sub_track_idx += 1

                # 1. Subtitle extraction to separate files
                if opts["extract_subs"] and f.lower().endswith(".mkv"):
                    for s in all_subs:
                        out_ext = {"subrip": "srt", "ass": "ass", "ssa": "ssa", "hdmv_pgs_subtitle": "sup"}.get(s["codec"], "srt")
                        out = f"{os.path.splitext(f)[0]}.{s['language']}.{out_ext}"
                        
                        matching_conv = next((tpath for tpath, s_info in converted_srts if s_info["index"] == s["index"]), None)
                        if matching_conv:
                            shutil.copyfile(matching_conv, f"{os.path.splitext(f)[0]}.{s['language']}.srt")
                        elif s["codec"] != "none":
                            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", f, "-map", f"0:{s['index']}", "-c:s", "copy", out]
                            subprocess.run(cmd, check=True)

                # 2. Audio export
                if opts["export_audio"]:
                    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-i",f,"-vn","-map","0:a:0?","-c:a","copy","-y",os.path.splitext(f)[0]+"_audio.m4a"],check=True)

                # 3. Video processing
                crop = get_crop_for_file(f, info, opts)
                cut = get_cut_for_file(f, opts)
                target = get_target_resolution(info, opts)
                subtitle_change = (opts["subtitle_keep"] == "none") or (opts["subtitle_keep"] == "selected")
                has_vtt_subs = len(converted_srts) > 0 if opts["subtitle_keep"] != "none" else False
                hdr_convert = opts.get("remove_hdr") and info.get("is_hdr")

                has = (opts["video_convert"] or opts["audio_convert"] or opts["subtitle_insert"] or
                       opts["audio_tracks_to_keep"] or crop["enabled"] or cut["enabled"] or
                       target != (info["width"], info["height"]) or subtitle_change or has_vtt_subs or hdr_convert)

                if has:
                    srts_for_video = []
                    if opts["subtitle_keep"] == "all":
                        srts_for_video = converted_srts
                    elif opts["subtitle_keep"] == "selected":
                        sel_langs = {norm_lang(l) for l in opts.get("subtitle_languages", [])}
                        srts_for_video = [(p, s_inf) for p, s_inf in converted_srts if norm_lang(s_inf["language"]) in sel_langs]

                    cmd = build_ffmpeg_command(f, opts, info, extra_converted_srts=srts_for_video)
                    subprocess.run(cmd, check=True)
                    done += 1
            except Exception as e:
                errors.append(f"{os.path.basename(f)}: {e}")
            finally:
                for tmp in temp_files_to_cleanup:
                    try:
                        if os.path.exists(tmp): os.remove(tmp)
                    except Exception: pass

        def finish():
            self.run_btn.configure(state="normal");self.update_status()
            if errors:messagebox.showerror("Completed with errors",f"Processed: {done}\nErrors: {len(errors)}\n\n"+"\n".join(errors))
            else:messagebox.showinfo("Done",f"Processing completed.\nProcessed videos: {done}")
        self.root.after(0,finish)

    def update_status(self):self.status_var.set(f"{len(self.files)} videos | NVENC: {'yes' if supports_nvenc() else 'no'}")


def run_gui():
    if not tool_exists("ffmpeg") or not tool_exists("ffprobe"):
        r=tk.Tk();r.withdraw();messagebox.showerror("FFmpeg missing","ffmpeg and ffprobe are required in PATH.");r.destroy();return
    root=tk.Tk()
    try:ttk.Style().theme_use("clam")
    except Exception:pass
    VideoBatchProcessor(root);root.mainloop()


if __name__=="__main__":run_gui()