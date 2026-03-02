"""
Deepfake Training Orchestrator — Control Tower Dashboard
Run from project root: python -m ui.dashboard
"""

import ast
import os
import queue
import signal
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

# ─────────────────────────── Global State ───────────────────────────
is_tool_running = False
is_exiting = False
process = None
active_thread_count = 0
thread_lock = threading.Lock()
log_queue = queue.Queue()
shutdown_timer_id = None

# ─────────────────────────── Constants ──────────────────────────────
FOLDER_MAP = {
    "Preprocessor":      "utils/preprocessor/",
    "Metadata":          "utils/metadata/",
    "Feature Extractor": "utils/featureExtractor/",
    "Training":          "notebooks/",
    "Detection":         "models/",
}

BUTTON_WIDTH = 160   # px approximation for wrap calculation
WINDOW_WIDTH = 900

COLORS = {
    "bg":        "#0d0f1a",
    "panel":     "#131625",
    "border":    "#1e2235",
    "accent":    "#00e5ff",
    "accent2":   "#7c3aed",
    "text":      "#e2e8f0",
    "muted":     "#64748b",
    "btn":       "#1e2235",
    "btn_hover": "#252a42",
    "CRITICAL":  "#ef4444",
    "ERROR":     "#f97316",
    "INFO":      "#22c55e",
    "WARNING":   "#eab308",
}

FONT_TITLE  = ("Courier New", 11, "bold")
FONT_MONO   = ("Courier New", 9)
FONT_LABEL  = ("Courier New", 9)
FONT_BUTTON = ("Courier New", 9, "bold")


# ═══════════════════════════════════════════════════════════════════
#  BOOTSTRAPPER SANITY CHECK
# ═══════════════════════════════════════════════════════════════════
def sanity_check():
    if not os.path.isdir("utils"):
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Launch Error",
            "Missing /utils directory.\n\n"
            "Please launch from the project root:\n"
            "  python -m ui.dashboard"
        )
        root.destroy()
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
#  DYNAMIC TOOL SCANNER
# ═══════════════════════════════════════════════════════════════════
def scan_tools(panel_name: str) -> list[str]:
    folder = FOLDER_MAP[panel_name]
    if not os.path.isdir(folder):
        return []

    if panel_name == "Detection":
        exts = {".joblib", ".cbm"}
        return [
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1] in exts
        ]
    else:
        return [
            f for f in os.listdir(folder)
            if f.endswith(".py") and f != "__init__.py"
        ]


# ═══════════════════════════════════════════════════════════════════
#  AST INSPECTOR
# ═══════════════════════════════════════════════════════════════════
def extract_arguments(filepath: str) -> list[dict]:
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            source = fh.read()
        tree = ast.parse(source)
    except Exception:
        return []

    def collect_from_body(stmts) -> list[dict]:
        args = []
        for node in ast.walk(ast.Module(body=stmts, type_ignores=[])):
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
            ):
                call = node.value
                func = call.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "add_argument"
                ):
                    arg = parse_add_argument(call)
                    if arg:
                        args.append(arg)
        return args

    # Pass 1 — def main()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            result = collect_from_body(node.body)
            if result:
                return result

    # Pass 2 — if __name__ == "__main__"
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
            ):
                result = collect_from_body(node.body)
                if result:
                    return result

    return []


def parse_add_argument(call: ast.Call) -> dict | None:
    """Extract flag, help, default, required from add_argument call."""
    if not call.args:
        return None
    first = call.args[0]
    if not isinstance(first, ast.Constant) or not str(first.value).startswith("--"):
        return None

    flag = str(first.value)
    info = {"flag": flag, "help": "", "default": None, "required": False}

    for kw in call.keywords:
        if kw.arg == "help" and isinstance(kw.value, ast.Constant):
            info["help"] = str(kw.value.value)
        elif kw.arg == "default" and isinstance(kw.value, ast.Constant):
            info["default"] = kw.value.value
        elif kw.arg == "required" and isinstance(kw.value, ast.Constant):
            info["required"] = bool(kw.value.value)

    return info


# ═══════════════════════════════════════════════════════════════════
#  PARAMETER MODAL
# ═══════════════════════════════════════════════════════════════════
def open_parameter_modal(root: tk.Tk, filepath: str):
    global is_tool_running, is_exiting

    if is_tool_running or is_exiting:
        messagebox.showwarning(
            "⚠️  Task Running",
            "Please wait for the current task to finish."
        )
        return

    args = extract_arguments(filepath)

    modal = tk.Toplevel(root)
    modal.title(f"⚡ {os.path.basename(filepath)}")
    modal.configure(bg=COLORS["bg"])
    modal.grab_set()
    modal.resizable(False, False)

    # Title bar
    tk.Label(
        modal,
        text=f"{os.path.basename(filepath)}",
        font=("Courier New", 12, "bold"),
        bg=COLORS["bg"], fg=COLORS["accent"],
        anchor="w"
    ).pack(fill="x", padx=12, pady=(12, 4))

    tk.Frame(modal, bg=COLORS["border"], height=1).pack(fill="x", padx=12, pady=(0, 8))

    entries: dict[str, tk.StringVar] = {}

    if args:
        form_frame = tk.Frame(modal, bg=COLORS["bg"])
        form_frame.pack(fill="both", expand=True, padx=12, pady=4)

        for i, arg in enumerate(args):
            flag   = arg["flag"]
            label  = flag.lstrip("-")
            req    = arg["required"]
            default = arg["default"]
            help_t  = arg["help"]

            # Label
            display = f"{'* ' if req else '  '}{label}{' (Required)' if req else ''}"
            lbl_color = COLORS["accent"] if req else COLORS["text"]
            tk.Label(
                form_frame, text=display,
                font=FONT_LABEL, bg=COLORS["bg"], fg=lbl_color,
                anchor="w"
            ).grid(row=i * 2, column=0, columnspan=4, sticky="w", pady=(6, 0))

            if help_t:
                tk.Label(
                    form_frame, text=f"  {help_t}",
                    font=("Courier New", 9), bg=COLORS["bg"], fg=COLORS["muted"],
                    anchor="w"
                ).grid(row=i * 2, column=0, columnspan=4, sticky="w")

            var = tk.StringVar(value=str(default) if default is not None else "")
            entries[flag] = var

            entry = tk.Entry(
                form_frame, textvariable=var,
                font=FONT_MONO,
                bg=COLORS["btn"], fg=COLORS["text"],
                insertbackground=COLORS["accent"],
                relief="flat", bd=4,
                width=34
            )
            entry.grid(row=i * 2 + 1, column=0, sticky="ew", pady=2)

            def _browse_file(v=var):
                path = filedialog.askopenfilename()
                if path:
                    try:
                        rel = os.path.relpath(path)
                    except ValueError:
                        rel = path
                    v.set(rel)

            def _browse_dir(v=var):
                path = filedialog.askdirectory()
                if path:
                    try:
                        rel = os.path.relpath(path)
                    except ValueError:
                        rel = path
                    v.set(rel)

            _make_icon_btn(form_frame, "Browse Files 📄", _browse_file).grid(
                row=i * 2 + 1, column=1, padx=(4, 0), pady=2)
            _make_icon_btn(form_frame, "Browse Folders 📁", _browse_dir).grid(
                row=i * 2 + 1, column=2, padx=(2, 0), pady=2)

        form_frame.columnconfigure(0, weight=1)
    else:
        tk.Label(
            modal, text="  No configurable parameters found.",
            font=FONT_LABEL, bg=COLORS["bg"], fg=COLORS["muted"]
        ).pack(pady=12)

    # ── Action buttons
    btn_frame = tk.Frame(modal, bg=COLORS["bg"])
    btn_frame.pack(fill="x", padx=12, pady=12)

    def on_run():
        cmd = build_command(filepath, args, entries)
        modal.destroy()
        launch_task(root, filepath, cmd)

    def on_cancel():
        modal.destroy()

    _make_action_btn(btn_frame, "Run Task", on_run, COLORS["accent2"]).pack(
        side="left", padx=(0, 6))
    _make_action_btn(btn_frame, "Cancel", on_cancel, COLORS["btn"]).pack(
        side="left")


def _make_icon_btn(parent, text, cmd):
    btn = tk.Button(
        parent, text=text, command=cmd,
        font=("Courier New", 11),
        bg=COLORS["btn"], fg=COLORS["text"],
        activebackground=COLORS["btn_hover"],
        relief="flat", bd=0, padx=4, pady=2, cursor="hand2"
    )
    return btn


def _make_action_btn(parent, text, cmd, bg):
    btn = tk.Button(
        parent, text=text, command=cmd,
        font=FONT_BUTTON,
        bg=bg, fg="#ffffff" if bg != COLORS["btn"] else COLORS["text"],
        activebackground=COLORS["btn_hover"],
        relief="flat", bd=0, padx=12, pady=6, cursor="hand2"
    )
    return btn


# ═══════════════════════════════════════════════════════════════════
#  COMMAND BUILDER
# ═══════════════════════════════════════════════════════════════════
def build_command(filepath: str, args: list[dict], entries: dict[str, tk.StringVar]) -> list[str]:
    module = filepath.replace(os.sep, ".").replace("/", ".").removesuffix(".py")
    cmd = [sys.executable, "-u", "-m", module]

    for arg in args:
        flag = arg["flag"]
        val  = entries.get(flag, tk.StringVar()).get().strip()
        if val:
            cmd += [flag, val]

    return cmd


# ═══════════════════════════════════════════════════════════════════
#  TASK EXECUTION ENGINE
# ═══════════════════════════════════════════════════════════════════
def launch_task(root: tk.Tk, filepath: str, cmd: list[str]):
    global is_tool_running, process, active_thread_count

    is_tool_running = True
    active_thread_count = 2

    while not log_queue.empty():
        log_queue.get_nowait()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    process = proc

    log_win = open_log_viewer(root, os.path.basename(filepath), proc)

    def stream_reader(stream, default_tag):
        global active_thread_count, is_tool_running
        try:
            for line in stream:
                tag = classify_line(line, default_tag)
                log_queue.put((tag, line))
        finally:
            with thread_lock:
                active_thread_count -= 1
                if active_thread_count == 0:
                    is_tool_running = False
                    log_queue.put(("INFO", "\n─── Task finished ───\n"))

    threading.Thread(
        target=stream_reader, args=(proc.stdout, "WARNING"), daemon=True
    ).start()
    threading.Thread(
        target=stream_reader, args=(proc.stderr, "WARNING"), daemon=True
    ).start()


def classify_line(line: str, default_tag: str) -> str:
    for kw in ("CRITICAL", "ERROR", "INFO"):
        if kw in line:
            return kw
    return default_tag


# ═══════════════════════════════════════════════════════════════════
#  LOG VIEWER
# ═══════════════════════════════════════════════════════════════════
def open_log_viewer(root: tk.Tk, title: str, proc: subprocess.Popen) -> tk.Toplevel:
    win = tk.Toplevel(root)
    win.title(f"Log — {title}")
    win.configure(bg=COLORS["bg"])
    win.geometry("800x480")

    # Header
    header = tk.Frame(win, bg=COLORS["panel"], pady=6)
    header.pack(fill="x")
    tk.Label(
        header, text=f"  ◈ {title}",
        font=FONT_TITLE, bg=COLORS["panel"], fg=COLORS["accent"]
    ).pack(side="left", padx=8)

    # Text area
    text = scrolledtext.ScrolledText(
        win,
        font=FONT_MONO,
        bg="#080a12", fg=COLORS["text"],
        insertbackground=COLORS["accent"],
        relief="flat", bd=0,
        state="disabled",
        wrap="word"
    )
    text.pack(fill="both", expand=True, padx=6, pady=6)

    for tag, color in [
        ("CRITICAL", COLORS["CRITICAL"]),
        ("ERROR",    COLORS["ERROR"]),
        ("INFO",     COLORS["INFO"]),
        ("WARNING",  COLORS["WARNING"]),
    ]:
        text.tag_config(tag, foreground=color)

    MAX_LINES = 5000
    user_scrolled = [False]

    def on_scroll(*_):
        pos = text.yview()
        user_scrolled[0] = pos[1] < 0.95

    text.bind("<MouseWheel>", on_scroll)
    text.bind("<ButtonPress-4>", on_scroll)
    text.bind("<ButtonPress-5>", on_scroll)

    # Bottom toolbar
    toolbar = tk.Frame(win, bg=COLORS["panel"], pady=4)
    toolbar.pack(fill="x")

    stop_btn = tk.Button(
        toolbar,
        text="Stop Task",
        font=FONT_BUTTON,
        bg="#7f1d1d", fg="#fca5a5",
        activebackground="#991b1b",
        relief="flat", bd=0, padx=10, pady=4,
        cursor="hand2"
    )
    stop_btn.pack(side="right", padx=8)

    def on_stop():
        if not messagebox.askyesno("Stop Task", "Are you sure you want to stop the running task?"):
            return
        stop_btn.config(text="Stopping…", state="disabled")
        try:
            os.kill(proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass

    stop_btn.config(command=on_stop)

    # Queue drainer
    def drain_queue():
        try:
            for _ in range(200):
                tag, line = log_queue.get_nowait()
                text.config(state="normal")

                text.insert("end", line, tag)

                # Trim buffer
                line_count = int(text.index("end-1c").split(".")[0])
                if line_count > MAX_LINES:
                    text.delete("1.0", f"{line_count - MAX_LINES}.0")

                if not user_scrolled[0]:
                    text.see("end")

                text.config(state="disabled")
        except queue.Empty:
            pass
        win.after(100, drain_queue)

    win.after(100, drain_queue)
    return win


# ═══════════════════════════════════════════════════════════════════
#  GRACEFUL SHUTDOWN
# ═══════════════════════════════════════════════════════════════════
def on_exit(root: tk.Tk):
    global is_exiting

    if is_exiting:
        return
    is_exiting = True
    root.config(cursor="watch")

    def _kill_and_exit():
        global process, active_thread_count
        if process:
            try:
                os.kill(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass

        deadline = 20
        elapsed = 0

        def monitor():
            nonlocal elapsed
            with thread_lock:
                done = active_thread_count == 0

            if done or elapsed >= deadline * 1000:
                if process and process.poll() is None:
                    process.kill()
                root.destroy()
                return

            elapsed += 500
            root.after(500, monitor)

        root.after(500, monitor)

    _kill_and_exit()


# ═══════════════════════════════════════════════════════════════════
#  MAIN DASHBOARD UI
# ═══════════════════════════════════════════════════════════════════
def build_panel(root: tk.Tk, parent: tk.Frame, name: str, grid_opts: dict):
    tools = scan_tools(name)
    is_detection = (name == "Detection")

    frame = tk.LabelFrame(
        parent,
        text=f"  ◈ {name}  ",
        font=("Courier New", 11, "bold"),
        bg=COLORS["panel"],
        fg=COLORS["accent"],
        bd=1,
        relief="flat",
        highlightbackground=COLORS["border"],
        highlightthickness=1,
        labelanchor="nw",
        padx=6, pady=6,
    )
    frame.grid(**grid_opts, padx=6, pady=6, sticky="nsew")

    if not tools:
        tk.Label(
            frame,
            text="No usable scripts found",
            font=FONT_LABEL,
            bg=COLORS["panel"], fg=COLORS["muted"],
            anchor="center"
        ).pack(expand=True)
        return

    # Wrapping button grid
    cols = max(1, WINDOW_WIDTH // (BUTTON_WIDTH + 20))
    folder = FOLDER_MAP[name]

    for idx, filename in enumerate(sorted(tools)):
        row = idx // cols
        col = idx % cols

        if is_detection:
            lbl = tk.Label(
                frame, text=filename,
                font=FONT_MONO,
                bg=COLORS["panel"], fg=COLORS["muted"],
                relief="flat", padx=6, pady=4,
                bd=1
            )
            lbl.grid(row=row, column=col, padx=3, pady=3, sticky="w")
        else:
            filepath = os.path.join(folder, filename)

            btn = tk.Button(
                frame,
                text=filename,
                font=FONT_BUTTON,
                bg=COLORS["btn"], fg=COLORS["text"],
                activebackground=COLORS["btn_hover"],
                activeforeground=COLORS["accent"],
                relief="flat", bd=0,
                padx=8, pady=5,
                cursor="hand2",
                command=lambda fp=filepath: open_parameter_modal(root, fp),
                wraplength=BUTTON_WIDTH,
                justify="left",
                anchor="w"
            )
            btn.grid(row=row, column=col, padx=3, pady=3, sticky="ew")
            _bind_hover(btn)


def _bind_hover(btn: tk.Button):
    btn.bind("<Enter>", lambda _: btn.config(bg=COLORS["btn_hover"]))
    btn.bind("<Leave>", lambda _: btn.config(bg=COLORS["btn"]))


def build_dashboard():
    sanity_check()

    root = tk.Tk()
    root.title("Control Tower — Deepfake Training Orchestrator")
    root.configure(bg=COLORS["bg"])
    root.minsize(WINDOW_WIDTH, 520)
    root.protocol("WM_DELETE_WINDOW", lambda: on_exit(root))

    # ── Top bar
    topbar = tk.Frame(root, bg=COLORS["panel"], pady=8)
    topbar.pack(fill="x")

    tk.Label(
        topbar,
        text="CONTROL TOWER",
        font=("Courier New", 14, "bold"),
        bg=COLORS["panel"], fg=COLORS["accent"],
    ).pack(side="left", padx=16)

    tk.Label(
        topbar,
        text="Deepfake Training Orchestrator",
        font=("Courier New", 8),
        bg=COLORS["panel"], fg=COLORS["muted"],
    ).pack(side="left")

    tk.Frame(root, bg=COLORS["border"], height=1).pack(fill="x")

    # ── Panel grid
    grid_frame = tk.Frame(root, bg=COLORS["bg"])
    grid_frame.pack(fill="both", expand=True, padx=6, pady=6)

    for c in range(3):
        grid_frame.columnconfigure(c, weight=1)
    for r in range(2):
        grid_frame.rowconfigure(r, weight=1)

    panel_layout = [
        ("Preprocessor",     {"row": 0, "column": 0}),
        ("Metadata",         {"row": 0, "column": 1}),
        ("Feature Extractor",{"row": 0, "column": 2}),
        ("Training",         {"row": 1, "column": 0}),
        ("Detection",        {"row": 1, "column": 1, "columnspan": 2}),
    ]

    for name, opts in panel_layout:
        build_panel(root, grid_frame, name, opts)

    # ── Status bar
    statusbar = tk.Frame(root, bg=COLORS["panel"], pady=4)
    statusbar.pack(fill="x", side="bottom")
    tk.Label(
        statusbar,
        text="  ● IDLE — no task running",
        font=("Courier New", 9),
        bg=COLORS["panel"], fg=COLORS["muted"]
    ).pack(side="left", padx=8)

    def _ctrl_c(*_):
        on_exit(root)

    import signal as _sig
    try:
        _sig.signal(_sig.SIGINT, _ctrl_c)
    except (OSError, ValueError):
        pass  # Not main thread

    root.mainloop()


if __name__ == "__main__":
    build_dashboard()