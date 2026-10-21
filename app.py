"""Giao diện nhận diện hoa: chọn file ảnh, camera laptop hoặc camera điện thoại (IP Webcam).

    python app.py
"""
import sys
import threading
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk

import common as C

# ------------------------------------------------------------------ cấu hình
THRESHOLD = 0.60        # dưới ngưỡng này hiện "Không chắc chắn" (có thanh kéo chỉnh trên giao diện)
CAMERA_INDEX = 0        # camera laptop; nếu không mở được thử 1, 2...
PHONE_CAMERA_URL = "192.168.2.4:8080"   # địa chỉ hiện trong app IP Webcam, ví dụ "192.168.1.5:8080"

CONNECT_TIMEOUT = 8     # chờ tối đa 8 giây để camera gửi khung hình đầu tiên
PREDICT_EVERY = 0.3     # camera dự đoán mỗi 0.3 giây
SMOOTHING = 5           # trung bình N lần dự đoán gần nhất cho kết quả đỡ nhảy
VIEW = (560, 420)       # kích thước khung hiển thị ảnh
ROI_RATIO = 0.75        # cạnh khung vuông / cạnh ngắn của khung hình
UI_REFRESH_MS = 15      # chu kỳ vòng lặp camera
MAX_MISSES = 60         # ~3 giây liên tục không có hình thì coi là mất kết nối
MISS_SLEEP = 0.05

COLOR_SURE = (0, 200, 0)
COLOR_UNSURE = (255, 165, 0)
TOP_K = 3


# -------------------------------------------------------------------- camera
def normalize_phone_url(raw: str) -> str:
    """'192.168.1.5:8080' -> 'http://192.168.1.5:8080/video' (cho phép gõ thiếu http:// và /video)."""
    url = raw.strip()
    if not url.startswith("http"):
        url = "http://" + url
    if urlparse(url).path in ("", "/"):
        url = url.rstrip("/") + "/video"
    return url


def open_laptop_capture(index: int):
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def open_phone_capture(url: str):
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG)


class FrameGrabber(threading.Thread):
    """Đọc camera ở luồng nền và luôn giữ khung hình mới nhất.

    Nhờ vậy giao diện không bị đơ khi mạng Wi-Fi chậm và không bị trễ do dồn khung hình cũ.
    `open_capture` là hàm không tham số trả về cv2.VideoCapture.
    """

    def __init__(self, open_capture):
        super().__init__(daemon=True)
        self.open_capture = open_capture
        self.frame = None        # khung hình BGR mới nhất
        self.failed = False
        self.running = True

    def run(self):
        cap = self.open_capture()
        if not cap.isOpened():
            self.failed = True
            cap.release()
            return
        misses = 0
        while self.running:
            ok, frame = cap.read()
            if ok:
                self.frame, misses = frame, 0
                continue
            misses += 1
            if misses > MAX_MISSES:
                self.failed = True
                break
            time.sleep(MISS_SLEEP)
        cap.release()


class CameraSession:
    """Toàn bộ trạng thái của một lần mở camera. Mở camera mới = tạo đối tượng mới."""

    def __init__(self, label: str, open_capture, mirror: bool):
        self.label = label                       # "laptop" / "điện thoại"
        self.mirror = mirror
        self.connected = False
        self.started = time.monotonic()
        self.last_raw = None                     # khung BGR đã xử lý gần nhất (để biết khi nào có khung mới)
        self.last_frame = None                   # khung RGB gần nhất (dùng cho nút chụp)
        self.last_pred = 0.0
        self.recent = deque(maxlen=SMOOTHING)
        self.grabber = FrameGrabber(open_capture)
        self.grabber.start()

    def has_failed(self) -> bool:
        g = self.grabber
        timed_out = g.frame is None and time.monotonic() - self.started > CONNECT_TIMEOUT
        return g.failed or timed_out

    def poll(self):
        """Trả về khung RGB mới (đã lật gương nếu cần), hoặc None nếu chưa có khung mới."""
        frame = self.grabber.frame
        if frame is None or frame is self.last_raw:
            return None
        self.last_raw = frame
        self.connected = True
        if self.mirror:
            frame = cv2.flip(frame, 1)
        self.last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.last_frame

    def due_for_prediction(self) -> bool:
        return time.monotonic() - self.last_pred >= PREDICT_EVERY

    def add_prediction(self, probs):
        self.last_pred = time.monotonic()
        self.recent.append(probs)

    def mean_probs(self):
        return np.mean(self.recent, axis=0) if self.recent else None

    def stop(self):
        self.grabber.running = False             # luồng nền tự giải phóng camera khi kết thúc


# ------------------------------------------------------------ bảng kết quả
class ResultPanel(ttk.Frame):
    """Cột bên phải: kết quả chính, top-K, thanh chỉnh ngưỡng tin cậy."""

    def __init__(self, parent, class_names):
        super().__init__(parent)
        self.class_names = class_names
        self.threshold = tk.DoubleVar(value=THRESHOLD)
        self._build_result()
        self._build_top_rows()
        self._build_threshold()

    def _build_result(self):
        ttk.Label(self, text="Kết quả nhận diện", style="Head.TLabel").pack(anchor="w")
        self.result_var, self.conf_var = tk.StringVar(value="—"), tk.StringVar()
        self.result_lbl = ttk.Label(self, textvariable=self.result_var, style="Result.TLabel", wraplength=340)
        self.result_lbl.pack(anchor="w", pady=(8, 0))
        ttk.Label(self, textvariable=self.conf_var, style="Sub.TLabel", wraplength=340).pack(anchor="w", pady=(0, 14))

    def _build_top_rows(self):
        ttk.Label(self, text=f"Top {TOP_K} dự đoán", style="Head.TLabel").pack(anchor="w")
        self.rows = []
        for _ in range(TOP_K):
            r = ttk.Frame(self)
            r.pack(fill="x", pady=4)
            name, pct = ttk.Label(r, width=24), ttk.Label(r, width=7, anchor="e")
            bar = ttk.Progressbar(r, maximum=100)
            name.grid(row=0, column=0, sticky="w")
            pct.grid(row=0, column=1, sticky="e")
            bar.grid(row=1, column=0, columnspan=2, sticky="we")
            r.columnconfigure(0, weight=1)
            self.rows.append((name, bar, pct))

    def _build_threshold(self):
        ttk.Label(self, text="Ngưỡng tin cậy", style="Head.TLabel").pack(anchor="w", pady=(24, 0))
        t = ttk.Frame(self)
        t.pack(fill="x")
        ttk.Scale(t, from_=0.3, to=0.95, variable=self.threshold, length=240).pack(side="left")
        lbl = ttk.Label(t, text=f"{THRESHOLD:.0%}", width=6)
        lbl.pack(side="left", padx=8)
        self.threshold.trace_add("write", lambda *_: lbl.configure(text=f"{self.threshold.get():.0%}"))

    def name_of(self, index: int) -> str:
        return C.display_name(self.class_names[index])

    def is_confident(self, probs) -> bool:
        return probs is not None and probs.max() >= self.threshold.get()

    def clear(self):
        self.result_var.set("—")
        self.conf_var.set("")
        self.result_lbl.configure(style="Result.TLabel")
        for name, bar, pct in self.rows:
            name.configure(text="")
            pct.configure(text="")
            bar["value"] = 0

    def show(self, probs):
        top = np.argsort(probs)[::-1][:TOP_K]
        best, thr = top[0], self.threshold.get()
        if self.is_confident(probs):
            self.result_var.set(self.name_of(best))
            self.result_lbl.configure(style="Good.Result.TLabel")
            self.conf_var.set(f"Độ tin cậy: {probs[best]:.1%}")
        else:
            self.result_var.set("Không chắc chắn")
            self.result_lbl.configure(style="Warn.Result.TLabel")
            self.conf_var.set(f"Gần nhất: {self.name_of(best)} ({probs[best]:.1%}), thấp hơn ngưỡng {thr:.0%}")
        for (name, bar, pct), i in zip(self.rows, top):
            name.configure(text=self.name_of(i))
            bar["value"] = probs[i] * 100
            pct.configure(text=f"{probs[i]:.1%}")


# ----------------------------------------------------------------- ứng dụng
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Phân loại hoa - MobileNetV2")
        self.geometry("1000x620")
        self.resizable(False, False)

        self.device = C.get_device()
        self.model, self.class_names = C.load_model(self.device)
        self.transform = C.eval_transform()

        self.cam = None            # CameraSession đang chạy (None nếu không dùng camera)
        self.job = None            # tác vụ after() của vòng lặp camera
        self.photo = None          # giữ tham chiếu để Tk không xóa ảnh

        self._style()
        self.menu = self._build_menu()
        self.work = self._build_work()
        self.show_menu()
        self.protocol("WM_DELETE_WINDOW", self.close)

    # ---------------------------------------------------------------- giao diện
    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("Title.TLabel", font=("Segoe UI", 28, "bold"))
        s.configure("Sub.TLabel", font=("Segoe UI", 10), foreground="#555555")
        s.configure("Head.TLabel", font=("Segoe UI", 12, "bold"))
        s.configure("Result.TLabel", font=("Segoe UI", 20, "bold"))
        s.configure("Good.Result.TLabel", font=("Segoe UI", 20, "bold"), foreground="#2e7d32")
        s.configure("Warn.Result.TLabel", font=("Segoe UI", 20, "bold"), foreground="#e65100")
        s.configure("Big.TButton", font=("Segoe UI", 14), padding=(40, 16))

    def _build_menu(self):
        f = ttk.Frame(self)
        box = ttk.Frame(f)
        box.place(relx=0.5, rely=0.45, anchor="center")
        ttk.Label(box, text="Phân loại hoa", style="Title.TLabel").pack(pady=(0, 4))
        ttk.Label(box, text=f"MobileNetV2 · chạy trên: {C.device_label(self.device)}",
                  style="Sub.TLabel").pack(pady=(0, 30))
        for text, command in (("Chọn ảnh để nhận diện", self.start_image),
                              ("Mở camera laptop", self.start_laptop),
                              ("Mở camera điện thoại", self.start_phone)):
            ttk.Button(box, text=text, style="Big.TButton", command=command).pack(fill="x", pady=8)
        return f

    def _build_work(self):
        f = ttk.Frame(self)
        self._build_preview(f).pack(side="left", padx=20, pady=20)
        self.panel = ResultPanel(f, self.class_names)
        self.panel.pack(side="left", fill="both", expand=True, padx=(0, 20), pady=20)
        return f

    def _build_preview(self, parent):
        left = ttk.Frame(parent)
        self.canvas = tk.Canvas(left, width=VIEW[0], height=VIEW[1], bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack()
        self.info = tk.StringVar()
        ttk.Label(left, textvariable=self.info, style="Sub.TLabel", wraplength=VIEW[0]).pack(anchor="w", pady=6)
        row = ttk.Frame(left)
        row.pack(fill="x")
        self.btn_action = ttk.Button(row)
        self.btn_action.pack(side="left")
        ttk.Button(row, text="Quay lại menu", command=self.show_menu).pack(side="right")
        return left

    def _switch(self, frame):
        for f in (self.menu, self.work):
            f.pack_forget()
        frame.pack(fill="both", expand=True)

    def show_menu(self):
        self.stop_camera()
        self._switch(self.menu)

    def _open_work(self, button_text, command):
        self.btn_action.configure(text=button_text, command=command)
        self.canvas.delete("all")
        self.info.set("")
        self.panel.clear()
        self._switch(self.work)

    def draw(self, img: Image.Image):
        img = img.copy()
        img.thumbnail(VIEW)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(VIEW[0] // 2, VIEW[1] // 2, image=self.photo)

    # --------------------------------------------------------------- dự đoán
    @torch.inference_mode()
    def predict(self, img: Image.Image) -> np.ndarray:
        x = self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)
        return torch.softmax(self.model(x), dim=1)[0].cpu().numpy()

    # ------------------------------------------------------------ chế độ ảnh
    def start_image(self):
        self.stop_camera()
        self._open_work("Chọn ảnh...", self.choose_image)
        self.choose_image()

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh hoa",
            initialdir=str(C.IMAGES_DIR if C.IMAGES_DIR.exists() else Path.home()),
            filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Tất cả", "*.*")])
        if not path:
            return
        try:
            with Image.open(path) as im:
                img = im.convert("RGB")
        except Exception as exc:
            messagebox.showerror("Lỗi", f"Không mở được ảnh:\n{exc}")
            return
        self.draw(img)
        self.info.set(path)
        self.panel.show(self.predict(img))

    # ----------------------------------------------------------- chế độ camera
    def start_laptop(self):
        self._start_camera("laptop", lambda: open_laptop_capture(CAMERA_INDEX), mirror=True)

    def start_phone(self):
        if not PHONE_CAMERA_URL.strip():
            messagebox.showinfo("Camera điện thoại",
                                'Chưa cấu hình địa chỉ camera.\nMở app.py, điền PHONE_CAMERA_URL '
                                '(ví dụ "192.168.1.5:8080") rồi chạy lại.')
            return
        url = normalize_phone_url(PHONE_CAMERA_URL)
        self._start_camera("điện thoại", lambda: open_phone_capture(url), mirror=False)

    def _start_camera(self, label, open_capture, mirror):
        self.stop_camera()
        self.cam = CameraSession(label, open_capture, mirror)
        self._open_work("Chụp & lưu ảnh", self.capture)
        self.info.set(f"Đang kết nối camera {label}...")
        self.camera_loop()

    def camera_loop(self):
        cam = self.cam
        if cam is None:
            return
        if cam.has_failed():
            message = self._camera_error(cam)
            self.show_menu()
            messagebox.showerror("Camera", message)
            return

        was_connected = cam.connected
        rgb = cam.poll()                       # None nếu chưa có khung hình mới
        if rgb is not None:
            if not was_connected:
                self.info.set("Đưa hoa vào giữa khung vuông. Xanh = đủ tin cậy, cam = chưa chắc.")
            box = self._roi_box(rgb)
            self._maybe_predict(cam, rgb, box)
            self._draw_overlay(cam, rgb, box)
        self.job = self.after(UI_REFRESH_MS, self.camera_loop)

    @staticmethod
    def _camera_error(cam: CameraSession) -> str:
        if cam.connected:
            return "Mất kết nối với camera."
        if cam.label == "điện thoại":
            return ("Không kết nối được camera điện thoại.\nKiểm tra: điện thoại và laptop cùng Wi-Fi, "
                    "đã bấm 'Start server' trong app IP Webcam, và PHONE_CAMERA_URL trong app.py đúng.")
        return ("Không mở được camera laptop.\nĐóng ứng dụng khác đang dùng camera (Zoom, Teams...) "
                "hoặc đổi CAMERA_INDEX trong app.py.")

    @staticmethod
    def _roi_box(rgb):
        """Khung vuông ở giữa ảnh: trả về (x0, y0, cạnh)."""
        h, w = rgb.shape[:2]
        side = int(min(h, w) * ROI_RATIO)
        return (w - side) // 2, (h - side) // 2, side

    def _maybe_predict(self, cam: CameraSession, rgb, box):
        if not cam.due_for_prediction():
            return
        x0, y0, side = box
        cam.add_prediction(self.predict(Image.fromarray(rgb[y0:y0 + side, x0:x0 + side])))
        self.panel.show(cam.mean_probs())

    def _draw_overlay(self, cam: CameraSession, rgb, box):
        x0, y0, side = box
        color = COLOR_SURE if self.panel.is_confident(cam.mean_probs()) else COLOR_UNSURE
        shown = rgb.copy()
        cv2.rectangle(shown, (x0, y0), (x0 + side, y0 + side), color, 3)
        self.draw(Image.fromarray(shown))

    def capture(self):
        cam = self.cam
        if cam is None or cam.last_frame is None:
            return
        probs = cam.mean_probs()
        label = self.class_names[int(probs.argmax())] if probs is not None else "unknown"
        out = C.ROOT / "captures"
        out.mkdir(exist_ok=True)
        path = out / f"capture_{time.strftime('%Y%m%d_%H%M%S')}_{label}.png"
        Image.fromarray(cam.last_frame).save(path)
        self.info.set(f"Đã lưu: {path}")

    def stop_camera(self):
        if self.job is not None:
            self.after_cancel(self.job)
            self.job = None
        if self.cam is not None:
            self.cam.stop()
            self.cam = None

    def close(self):
        self.stop_camera()
        self.destroy()


if __name__ == "__main__":
    try:
        App().mainloop()
    except FileNotFoundError as exc:      # chưa có model
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Thiếu mô hình", str(exc))
