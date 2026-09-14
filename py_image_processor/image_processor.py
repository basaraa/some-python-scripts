import os
import datetime
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter, ImageOps

SUPPORTED_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.gif', '.ico')

class BatchImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Batch Image Converter & Editor")
        self.root.geometry("1340x860")
        self.root.minsize(1050, 650)

        self.current_folder = os.getcwd()
        self.active_file = None

        # Data structure for loaded folder items:
        # filepath -> {
        #   'orig_img': Image/None (lazy-loaded),
        #   'mod_img': Image/None,
        #   'pipeline': list of action dicts (e.g. {'type': 'filter', 'name': 'Sharpen', 'tag': 'Sharpen'}),
        #   'target_format': str/None,
        #   'ops': list[str],
        #   'checked': bool
        # }
        self.images_data = {}

        # Interactive crop variables
        self.crop_start_x = None
        self.crop_start_y = None
        self.crop_rect_id = None
        self.selected_crop_relative = None

        self.display_scale = 1.0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0

        self._build_ui()
        self.toggle_resize_inputs()

        # Keyboard shortcuts
        self.root.bind("<Control-z>", lambda e: self.undo_last())
        self.root.bind("<Control-Z>", lambda e: self.undo_last())

        self.load_folder(self.current_folder)

    def _build_ui(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # ==========================================================
        # 1. LEFT PANEL: FILE EXPLORER & MULTI-SELECTION
        # ==========================================================
        left_frame = ttk.Frame(main_paned, width=330)
        main_paned.add(left_frame, weight=0)

        folder_box = ttk.LabelFrame(left_frame, text="Working Directory", padding=6)
        folder_box.pack(fill=tk.X, padx=4, pady=4)

        ttk.Button(folder_box, text="📁 Change Directory", command=self.change_folder).pack(fill=tk.X)
        self.lbl_folder_path = ttk.Label(folder_box, text=self.current_folder, font=("", 8), wraplength=300)
        self.lbl_folder_path.pack(anchor="w", pady=(4, 0))

        selection_btn_frame = ttk.Frame(left_frame)
        selection_btn_frame.pack(fill=tk.X, padx=4, pady=2)

        ttk.Button(selection_btn_frame, text="Mark [✓]", command=lambda: self.set_selection_checks(True)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(selection_btn_frame, text="Unmark [ ]", command=lambda: self.set_selection_checks(False)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(selection_btn_frame, text="All [✓]", command=lambda: self.set_all_checks(True)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(selection_btn_frame, text="All [ ]", command=lambda: self.set_all_checks(False)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(selection_btn_frame, text="🔄", width=3, command=lambda: self.load_folder(self.current_folder)).pack(side=tk.LEFT, padx=1)

        tree_frame = ttk.Frame(left_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        columns = ("name", "status")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", selectmode="extended")
        self.tree.heading("name", text="File Name")
        self.tree.heading("status", text="Status / Pending Ops")
        self.tree.column("name", width=150, anchor="w")
        self.tree.column("status", width=150, anchor="w")

        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<<TreeviewSelect>>", self.on_file_select)
        self.tree.bind("<Double-1>", self.on_file_double_click)
        self.tree.bind("<space>", lambda e: self.toggle_selected_checks())

        # Save and Revert / Undo Controls
        save_action_box = ttk.LabelFrame(left_frame, text="Export & Undo", padding=6)
        save_action_box.pack(fill=tk.X, padx=4, pady=4)

        ttk.Button(save_action_box, text="💾 SAVE ALL MODIFIED (with Timestamp)", command=self.save_all_modified).pack(fill=tk.X, pady=2)
        ttk.Button(save_action_box, text="↩️ Undo Last Step (Ctrl+Z)", command=self.undo_last).pack(fill=tk.X, pady=2)
        ttk.Button(save_action_box, text="🔄 Revert Selected to Original", command=self.revert_selected).pack(fill=tk.X, pady=2)
        ttk.Button(save_action_box, text="⚠️ Revert All to Original", command=self.revert_all).pack(fill=tk.X, pady=2)

        # ==========================================================
        # 2. MIDDLE PANEL: TOOLS & TRANSFORMATIONS
        # ==========================================================
        tools_outer = ttk.Frame(main_paned, width=320)
        main_paned.add(tools_outer, weight=0)

        tools_canvas = tk.Canvas(tools_outer, borderwidth=0, highlightthickness=0, width=300)
        tools_scroll = ttk.Scrollbar(tools_outer, orient=tk.VERTICAL, command=tools_canvas.yview)
        tools_frame = ttk.Frame(tools_canvas)

        tools_frame.bind("<Configure>", lambda e: tools_canvas.configure(scrollregion=tools_canvas.bbox("all")))
        tools_canvas.create_window((0, 0), window=tools_frame, anchor="nw")
        tools_canvas.configure(yscrollcommand=tools_scroll.set)

        tools_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tools_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Target Scope Selector
        scope_group = ttk.LabelFrame(tools_frame, text="Apply Operations To:", padding=6)
        scope_group.pack(fill=tk.X, padx=4, pady=4)

        self.apply_mode_var = tk.StringVar(value="selected")
        ttk.Radiobutton(scope_group, text="Selected in list (Highlighted)", variable=self.apply_mode_var, value="selected").pack(anchor="w")
        ttk.Radiobutton(scope_group, text="Checked items [✓]", variable=self.apply_mode_var, value="checked").pack(anchor="w")
        ttk.Radiobutton(scope_group, text="Active (Previewed) image only", variable=self.apply_mode_var, value="active").pack(anchor="w")

        # 1. Format Conversion
        conv_group = ttk.LabelFrame(tools_frame, text="1. Target Format & Quality", padding=6)
        conv_group.pack(fill=tk.X, padx=4, pady=4)

        ttk.Label(conv_group, text="Format:").pack(anchor="w")
        self.format_var = tk.StringVar(value="JPEG")
        self.cb_format = ttk.Combobox(conv_group, textvariable=self.format_var, values=["JPEG", "PNG", "WEBP", "BMP", "TIFF", "GIF", "ICO"], state="readonly")
        self.cb_format.pack(fill=tk.X, pady=2)

        ttk.Label(conv_group, text="Quality (for JPG / WEBP):").pack(anchor="w", pady=(4, 0))
        self.quality_var = tk.IntVar(value=90)
        ttk.Scale(conv_group, from_=1, to=100, variable=self.quality_var, orient=tk.HORIZONTAL).pack(fill=tk.X)

        ttk.Button(conv_group, text="Set Target Format", command=self.apply_format_change).pack(fill=tk.X, pady=4)

        # 2. Crop
        crop_group = ttk.LabelFrame(tools_frame, text="2. Interactive Crop", padding=6)
        crop_group.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(crop_group, text="Drag a box on preview canvas:", font=("", 8)).pack(anchor="w")
        self.btn_crop = ttk.Button(crop_group, text="✂️ Crop to Selection", state=tk.DISABLED, command=self.apply_crop)
        self.btn_crop.pack(fill=tk.X, pady=2)

        # 3. Resize (Mutually exclusive Pixels vs Percentage)
        resize_group = ttk.LabelFrame(tools_frame, text="3. Resize", padding=6)
        resize_group.pack(fill=tk.X, padx=4, pady=4)

        self.resize_mode_var = tk.StringVar(value="pixels")

        self.rb_pixels = ttk.Radiobutton(
            resize_group, text="By Pixels (W x H):", variable=self.resize_mode_var,
            value="pixels", command=self.toggle_resize_inputs
        )
        self.rb_pixels.pack(anchor="w")

        dim_f = ttk.Frame(resize_group)
        dim_f.pack(fill=tk.X, pady=2, padx=(16, 0))
        ttk.Label(dim_f, text="W:").grid(row=0, column=0)
        self.w_var = tk.StringVar()
        self.ent_w = ttk.Entry(dim_f, textvariable=self.w_var, width=6)
        self.ent_w.grid(row=0, column=1, padx=2)

        ttk.Label(dim_f, text="H:").grid(row=0, column=2)
        self.h_var = tk.StringVar()
        self.ent_h = ttk.Entry(dim_f, textvariable=self.h_var, width=6)
        self.ent_h.grid(row=0, column=3, padx=2)

        self.aspect_var = tk.BooleanVar(value=True)
        self.chk_aspect = ttk.Checkbutton(resize_group, text="Keep Aspect Ratio", variable=self.aspect_var)
        self.chk_aspect.pack(anchor="w", padx=(16, 0))

        self.rb_percent = ttk.Radiobutton(
            resize_group, text="By Percentage (%):", variable=self.resize_mode_var,
            value="percent", command=self.toggle_resize_inputs
        )
        self.rb_percent.pack(anchor="w", pady=(6, 0))

        pct_f = ttk.Frame(resize_group)
        pct_f.pack(fill=tk.X, pady=2, padx=(16, 0))
        ttk.Label(pct_f, text="Scale:").grid(row=0, column=0)
        self.pct_var = tk.StringVar(value="50")
        self.ent_pct = ttk.Entry(pct_f, textvariable=self.pct_var, width=6)
        self.ent_pct.grid(row=0, column=1, padx=2)
        ttk.Label(pct_f, text="%").grid(row=0, column=2)

        ttk.Button(resize_group, text="Apply Resize", command=self.apply_resize).pack(fill=tk.X, pady=(6, 2))

        # 4. Rotation & Flip
        rot_group = ttk.LabelFrame(tools_frame, text="4. Rotate & Flip", padding=6)
        rot_group.pack(fill=tk.X, padx=4, pady=4)

        r_btn_f = ttk.Frame(rot_group)
        r_btn_f.pack(fill=tk.X, pady=1)
        ttk.Button(r_btn_f, text="⟲ 90° CCW", command=lambda: self.apply_rotate(90)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(r_btn_f, text="⟳ 90° CW", command=lambda: self.apply_rotate(-90)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(r_btn_f, text="⇄ Flip H", command=self.apply_flip_h).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        # 5. Independent Toggleable Filters
        filt_group = ttk.LabelFrame(tools_frame, text="5. Filters (Click to Toggle On/Off)", padding=6)
        filt_group.pack(fill=tk.X, padx=4, pady=4)

        ttk.Button(filt_group, text="Grayscale (B&W)", command=lambda: self.toggle_filter("Grayscale")).pack(fill=tk.X, pady=1)
        ttk.Button(filt_group, text="Sharpen", command=lambda: self.toggle_filter("Sharpen")).pack(fill=tk.X, pady=1)
        ttk.Button(filt_group, text="Blur", command=lambda: self.toggle_filter("Blur")).pack(fill=tk.X, pady=1)

        # ==========================================================
        # 3. RIGHT PANEL: CANVAS PREVIEW
        # ==========================================================
        preview_frame = ttk.Frame(main_paned)
        main_paned.add(preview_frame, weight=1)

        self.canvas = tk.Canvas(preview_frame, bg="#1a1a1a", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_crop_start)
        self.canvas.bind("<B1-Motion>", self.on_crop_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_crop_end)
        self.canvas.bind("<Configure>", lambda e: self.redraw_preview())

        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor="w", padding=4)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ==========================================================
    # UI STATE TOGGLE
    # ==========================================================
    def toggle_resize_inputs(self):
        mode = self.resize_mode_var.get()
        if mode == "pixels":
            self.ent_w.config(state=tk.NORMAL)
            self.ent_h.config(state=tk.NORMAL)
            self.chk_aspect.config(state=tk.NORMAL)
            self.ent_pct.config(state=tk.DISABLED)
        else:
            self.ent_w.config(state=tk.DISABLED)
            self.ent_h.config(state=tk.DISABLED)
            self.chk_aspect.config(state=tk.DISABLED)
            self.ent_pct.config(state=tk.NORMAL)

    def _sync_dimensions_fields(self):
        if self.active_file and self.active_file in self.images_data:
            img = self.images_data[self.active_file]['mod_img']
            if img:
                self.w_var.set(str(img.width))
                self.h_var.set(str(img.height))

    # ==========================================================
    # FILE LIST & SELECTION LOGIC
    # ==========================================================
    def change_folder(self):
        folder = filedialog.askdirectory(initialdir=self.current_folder, title="Select Image Directory")
        if folder:
            self.load_folder(folder)

    def load_folder(self, folder_path):
        self.current_folder = folder_path
        self.lbl_folder_path.config(text=folder_path)
        self.tree.delete(*self.tree.get_children())
        self.images_data.clear()
        self.active_file = None
        self.canvas.delete("all")

        try:
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(SUPPORTED_EXTS)]
            files.sort()
            for filename in files:
                full_path = os.path.join(folder_path, filename)
                self.images_data[full_path] = {
                    'orig_img': None,
                    'mod_img': None,
                    'pipeline': [],
                    'target_format': None,
                    'ops': [],
                    'checked': False
                }
                self.tree.insert("", tk.END, iid=full_path, values=(f"[ ] {filename}", "Original"))

            self.status_bar.config(text=f"Loaded {len(files)} images from: {folder_path}")
            if files:
                first_item = os.path.join(folder_path, files[0])
                self.tree.selection_set(first_item)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load directory:\n{e}")

    def on_file_select(self, event):
        selected = self.tree.selection()
        if not selected:
            return
        filepath = selected[0]
        self.active_file = filepath
        self._ensure_loaded(filepath)
        self._sync_dimensions_fields()
        self.redraw_preview()

    def on_file_double_click(self, event):
        item_id = self.tree.focus()
        if not item_id or item_id not in self.images_data:
            return
        self.images_data[item_id]['checked'] = not self.images_data[item_id]['checked']
        self._update_tree_row(item_id)

    def toggle_selected_checks(self):
        selected = self.tree.selection()
        for filepath in selected:
            if filepath in self.images_data:
                self.images_data[filepath]['checked'] = not self.images_data[filepath]['checked']
                self._update_tree_row(filepath)

    def set_selection_checks(self, check_state: bool):
        selected = self.tree.selection()
        for filepath in selected:
            if filepath in self.images_data:
                self.images_data[filepath]['checked'] = check_state
                self._update_tree_row(filepath)

    def set_all_checks(self, check_state: bool):
        for filepath, data in self.images_data.items():
            data['checked'] = check_state
            self._update_tree_row(filepath)

    def _update_tree_row(self, filepath):
        if not self.tree.exists(filepath):
            return
        data = self.images_data[filepath]
        fname = os.path.basename(filepath)
        chk_icon = "[✓]" if data['checked'] else "[ ]"

        if data['ops']:
            status_text = f"● [{', '.join(data['ops'])}]"
        else:
            status_text = "Original"

        self.tree.item(filepath, values=(f"{chk_icon} {fname}", status_text))

    def _ensure_loaded(self, filepath):
        if self.images_data[filepath]['orig_img'] is None:
            img = Image.open(filepath)
            self.images_data[filepath]['orig_img'] = img
            self.rebuild_image(filepath)

    def _get_working_image(self, filepath):
        self._ensure_loaded(filepath)
        return self.images_data[filepath]['mod_img']

    def _get_target_files(self):
        mode = self.apply_mode_var.get()
        if mode == "selected":
            selected = list(self.tree.selection())
            return selected if selected else ([self.active_file] if self.active_file else [])
        elif mode == "checked":
            return [fp for fp, d in self.images_data.items() if d['checked']]
        elif mode == "active":
            return [self.active_file] if self.active_file else []
        return []

    # ==========================================================
    # NON-DESTRUCTIVE PIPELINE ENGINE (REBUILDS ACCURATELY)
    # ==========================================================
    def rebuild_image(self, filepath):
        """Re-applies the action pipeline on the original image."""
        data = self.images_data[filepath]
        if data['orig_img'] is None:
            return

        img = data['orig_img'].copy()
        
        for action in data['pipeline']:
            act_type = action['type']
            
            if act_type == 'crop':
                rx1, ry1, rx2, ry2 = action['data']
                w, h = img.size
                crop_box = (int(rx1 * w), int(ry1 * h), int(rx2 * w), int(ry2 * h))
                img = img.crop(crop_box)
                
            elif act_type == 'resize':
                r_mode = action['data']['mode']
                if r_mode == 'pixels':
                    target_w = action['data']['w']
                    target_h = action['data']['h']
                    if action['data']['aspect']:
                        orig_w, orig_h = img.size
                        if target_w != orig_w:
                            target_h = max(1, int(orig_h * (target_w / orig_w)))
                        elif target_h != orig_h:
                            target_w = max(1, int(orig_w * (target_h / orig_h)))
                    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                elif r_mode == 'percent':
                    pct = action['data']['pct']
                    w = max(1, int(img.width * (pct / 100.0)))
                    h = max(1, int(img.height * (pct / 100.0)))
                    img = img.resize((w, h), Image.Resampling.LANCZOS)
                    
            elif act_type == 'rotate':
                deg = action['data']
                img = img.rotate(deg, expand=True)
                
            elif act_type == 'flip_h':
                img = ImageOps.mirror(img)
                
            elif act_type == 'filter':
                fname = action['name']
                if fname == 'Grayscale':
                    img = ImageOps.grayscale(img)
                elif fname == 'Sharpen':
                    img = img.filter(ImageFilter.SHARPEN)
                elif fname == 'Blur':
                    img = img.filter(ImageFilter.GaussianBlur(radius=2))

        data['mod_img'] = img
        data['ops'] = [a['tag'] for a in data['pipeline']]
        if data['target_format']:
            data['ops'].append(f"->{data['target_format']}")
            
        self._update_tree_row(filepath)
        return img

    # ==========================================================
    # TRUE INDEPENDENT FILTER TOGGLE & UNDO
    # ==========================================================
    def toggle_filter(self, filter_name):
        """Toggles only this specific filter on/off without affecting other transforms."""
        targets = self._get_target_files()
        if not targets:
            messagebox.showwarning("Warning", "No target files selected.")
            return

        sample_fp = targets[0]
        self._ensure_loaded(sample_fp)
        has_filter = any(a['type'] == 'filter' and a['name'] == filter_name for a in self.images_data[sample_fp]['pipeline'])

        for filepath in targets:
            self._ensure_loaded(filepath)
            data = self.images_data[filepath]
            
            if has_filter:
                # Remove ONLY this filter from pipeline
                data['pipeline'] = [a for a in data['pipeline'] if not (a['type'] == 'filter' and a['name'] == filter_name)]
            else:
                # Add filter to pipeline
                data['pipeline'].append({'type': 'filter', 'name': filter_name, 'tag': filter_name})
                
            self.rebuild_image(filepath)

        self._sync_dimensions_fields()
        self.redraw_preview()
        action_str = "Removed" if has_filter else "Applied"
        self.status_bar.config(text=f"{action_str} filter '{filter_name}' on {len(targets)} file(s).")

    def undo_last(self):
        """Reverts only the single most recent operation from the pipeline."""
        targets = self._get_target_files()
        if not targets:
            return

        undone = 0
        for filepath in targets:
            data = self.images_data[filepath]
            if data['pipeline']:
                data['pipeline'].pop()
                self.rebuild_image(filepath)
                undone += 1

        if undone > 0:
            self._sync_dimensions_fields()
            self.redraw_preview()
            self.status_bar.config(text=f"Undid last action for {undone} file(s).")

    # ==========================================================
    # TRANSFORMATIONS (PIPELINE APPENDERS)
    # ==========================================================
    def apply_format_change(self):
        fmt = self.format_var.get()
        targets = self._get_target_files()
        if not targets:
            messagebox.showwarning("Warning", "No target files selected.")
            return

        for filepath in targets:
            self.images_data[filepath]['target_format'] = fmt
            self.rebuild_image(filepath)

        self.status_bar.config(text=f"Target format set to {fmt} for {len(targets)} file(s).")

    def apply_resize(self):
        mode = self.resize_mode_var.get()
        targets = self._get_target_files()
        if not targets:
            return

        if mode == "pixels":
            try:
                target_w = int(self.w_var.get())
                target_h = int(self.h_var.get())
                if target_w <= 0 or target_h <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "Please provide valid positive integers for width and height.")
                return

            for filepath in targets:
                self._ensure_loaded(filepath)
                self.images_data[filepath]['pipeline'].append({
                    'type': 'resize',
                    'data': {'mode': 'pixels', 'w': target_w, 'h': target_h, 'aspect': self.aspect_var.get()},
                    'tag': f"{target_w}x{target_h}"
                })
                self.rebuild_image(filepath)

        elif mode == "percent":
            try:
                pct = float(self.pct_var.get())
                if pct <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "Please provide a valid positive percentage (e.g. 50 or 150).")
                return

            tag_label = f"{int(pct) if pct.is_integer() else pct}%"
            for filepath in targets:
                self._ensure_loaded(filepath)
                self.images_data[filepath]['pipeline'].append({
                    'type': 'resize',
                    'data': {'mode': 'percent', 'pct': pct},
                    'tag': tag_label
                })
                self.rebuild_image(filepath)

        self._sync_dimensions_fields()
        self.redraw_preview()
        self.status_bar.config(text=f"Applied Resize to {len(targets)} file(s).")

    def apply_rotate(self, deg):
        targets = self._get_target_files()
        for filepath in targets:
            self._ensure_loaded(filepath)
            self.images_data[filepath]['pipeline'].append({
                'type': 'rotate',
                'data': deg,
                'tag': f"Rot{deg}°"
            })
            self.rebuild_image(filepath)
            
        self._sync_dimensions_fields()
        self.redraw_preview()
        self.status_bar.config(text=f"Rotated {deg}° on {len(targets)} file(s).")

    def apply_flip_h(self):
        targets = self._get_target_files()
        for filepath in targets:
            self._ensure_loaded(filepath)
            self.images_data[filepath]['pipeline'].append({
                'type': 'flip_h',
                'data': None,
                'tag': "FlipH"
            })
            self.rebuild_image(filepath)
            
        self.redraw_preview()
        self.status_bar.config(text=f"Flipped horizontally {len(targets)} file(s).")

    # ==========================================================
    # CROP HANDLING
    # ==========================================================
    def on_crop_start(self, event):
        if not self.active_file:
            return
        self.crop_start_x = event.x
        self.crop_start_y = event.y
        if self.crop_rect_id:
            self.canvas.delete(self.crop_rect_id)
            self.crop_rect_id = None
            self.btn_crop.config(state=tk.DISABLED)

    def on_crop_drag(self, event):
        if not self.active_file or self.crop_start_x is None:
            return
        if self.crop_rect_id:
            self.canvas.coords(self.crop_rect_id, self.crop_start_x, self.crop_start_y, event.x, event.y)
        else:
            self.crop_rect_id = self.canvas.create_rectangle(
                self.crop_start_x, self.crop_start_y, event.x, event.y,
                outline="#00ff66", width=2, dash=(4, 4)
            )

    def on_crop_end(self, event):
        if not self.active_file or self.crop_start_x is None:
            return

        active_img = self._get_working_image(self.active_file)
        display_w = active_img.width * self.display_scale
        display_h = active_img.height * self.display_scale

        x1 = min(self.crop_start_x, event.x) - self.canvas_offset_x
        y1 = min(self.crop_start_y, event.y) - self.canvas_offset_y
        x2 = max(self.crop_start_x, event.x) - self.canvas_offset_x
        y2 = max(self.crop_start_y, event.y) - self.canvas_offset_y

        rx1 = max(0.0, min(1.0, x1 / display_w))
        ry1 = max(0.0, min(1.0, y1 / display_h))
        rx2 = max(0.0, min(1.0, x2 / display_w))
        ry2 = max(0.0, min(1.0, y2 / display_h))

        if (rx2 - rx1) > 0.02 and (ry2 - ry1) > 0.02:
            self.selected_crop_relative = (rx1, ry1, rx2, ry2)
            self.btn_crop.config(state=tk.NORMAL)
        else:
            self.selected_crop_relative = None
            self.btn_crop.config(state=tk.DISABLED)

    def apply_crop(self):
        if not self.selected_crop_relative:
            return

        rx1, ry1, rx2, ry2 = self.selected_crop_relative
        targets = self._get_target_files()
        
        for filepath in targets:
            self._ensure_loaded(filepath)
            self.images_data[filepath]['pipeline'].append({
                'type': 'crop',
                'data': (rx1, ry1, rx2, ry2),
                'tag': 'Crop'
            })
            self.rebuild_image(filepath)

        self.selected_crop_relative = None
        self.btn_crop.config(state=tk.DISABLED)
        self._sync_dimensions_fields()
        self.redraw_preview()
        self.status_bar.config(text=f"Applied Crop on {len(targets)} file(s).")

    # ==========================================================
    # EXPORT & REVERT
    # ==========================================================
    def save_all_modified(self):
        modified_files = [fp for fp, d in self.images_data.items() if d['ops'] or d['target_format']]
        if not modified_files:
            messagebox.showinfo("Information", "No modified files with pending changes found.")
            return

        out_dir = filedialog.askdirectory(
            initialdir=self.current_folder,
            title="Select Destination Directory"
        )
        if not out_dir:
            return

        ext_map = {
            "JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp",
            "BMP": ".bmp", "TIFF": ".tiff", "GIF": ".gif", "ICO": ".ico"
        }

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_count = 0

        for filepath in modified_files:
            data = self.images_data[filepath]
            img = data['mod_img']
            if img is None:
                continue

            target_fmt = data['target_format'] or self.format_var.get()
            ext = ext_map.get(target_fmt, ".jpg")

            base_name = os.path.splitext(os.path.basename(filepath))[0]
            out_filename = f"{base_name}_{timestamp}{ext}"
            out_filepath = os.path.join(out_dir, out_filename)

            img_to_save = img.copy()
            if target_fmt in ["JPEG", "BMP"] and img_to_save.mode in ("RGBA", "LA", "P"):
                bg = Image.new("RGB", img_to_save.size, (255, 255, 255))
                if img_to_save.mode != "RGBA":
                    img_to_save = img_to_save.convert("RGBA")
                bg.paste(img_to_save, mask=img_to_save.split()[3])
                img_to_save = bg
            elif target_fmt == "JPEG" and img_to_save.mode != "RGB":
                img_to_save = img_to_save.convert("RGB")

            if target_fmt in ["JPEG", "WEBP"]:
                img_to_save.save(out_filepath, format=target_fmt, quality=self.quality_var.get())
            elif target_fmt == "ICO":
                img_to_save.save(out_filepath, format=target_fmt, sizes=[(256, 256)])
            else:
                img_to_save.save(out_filepath, format=target_fmt)

            data['pipeline'].clear()
            data['target_format'] = None
            self.rebuild_image(filepath)
            saved_count += 1

        messagebox.showinfo("Export Successful", f"Saved {saved_count} file(s) to:\n{out_dir}\nTimestamp: {timestamp}")
        self.status_bar.config(text=f"Export complete: {saved_count} file(s) saved.")

    def revert_selected(self):
        targets = self._get_target_files()
        if not targets:
            return
        for filepath in targets:
            data = self.images_data[filepath]
            data['pipeline'].clear()
            data['target_format'] = None
            self.rebuild_image(filepath)

        self._sync_dimensions_fields()
        self.redraw_preview()
        self.status_bar.config(text=f"Reverted {len(targets)} file(s) to original state.")

    def revert_all(self):
        for filepath, data in self.images_data.items():
            data['pipeline'].clear()
            data['target_format'] = None
            self.rebuild_image(filepath)

        self._sync_dimensions_fields()
        self.redraw_preview()
        self.status_bar.config(text="All image modifications have been reverted.")

    # ==========================================================
    # CANVAS PREVIEW & ON-CANVAS BADGE
    # ==========================================================
    def redraw_preview(self):
        if not self.active_file or self.images_data[self.active_file]['mod_img'] is None:
            self.canvas.delete("all")
            return

        img = self.images_data[self.active_file]['mod_img']
        c_w = max(self.canvas.winfo_width(), 100)
        c_h = max(self.canvas.winfo_height(), 100)

        scale = min(c_w / img.width, c_h / img.height, 1.0)
        self.display_scale = scale

        pw = max(1, int(img.width * scale))
        ph = max(1, int(img.height * scale))

        preview_img = img.resize((pw, ph), Image.Resampling.LANCZOS)
        self.tk_preview = ImageTk.PhotoImage(preview_img)

        self.canvas.delete("all")
        self.canvas_offset_x = (c_w - pw) // 2
        self.canvas_offset_y = (c_h - ph) // 2

        self.canvas.create_image(self.canvas_offset_x, self.canvas_offset_y, anchor="nw", image=self.tk_preview)

        gcd = math.gcd(img.width, img.height)
        aspect_ratio = f"{img.width // gcd}:{img.height // gcd}" if gcd > 0 else "N/A"
        badge_text = f" {img.width} × {img.height} px  ({aspect_ratio}) "

        self.canvas.create_rectangle(10, 10, 10 + len(badge_text) * 8, 32, fill="#111111", outline="#00ff66")
        self.canvas.create_text(16, 21, anchor="w", text=badge_text, fill="#ffffff", font=("Segoe UI", 9, "bold"))

if __name__ == "__main__":
    root = tk.Tk()
    app = BatchImageEditorApp(root)
    root.mainloop()