import os
import threading
from tkinter import Tk, Entry, StringVar, ttk, messagebox, filedialog

from downloader import YouTubeDownloader

class AppUI(Tk):
    def __init__(self):
        super().__init__()
        self.title("Descărcare Video")
        self.geometry("600x400")

        self.url_entry = Entry(self, width=50)
        self.url_entry.pack(pady=10)

        self.format_var = StringVar(value="MP3")
        self.format_combobox = ttk.Combobox(self, textvariable=self.format_var, values=["MP3", "MP4"])
        self.format_combobox.pack(pady=10)

        self.save_path_var = StringVar()
        self.save_path_entry = Entry(self, textvariable=self.save_path_var, width=50)
        self.save_path_entry.pack(pady=10)
        self.browse_button = ttk.Button(self, text="Răsfoiește", command=self.browse_folder)
        self.browse_button.pack(pady=5)

        self.download_button = ttk.Button(self, text="Descarcă", command=self.start_download)
        self.download_button.pack(pady=10)

        self.progressbar = ttk.Progressbar(self, orient="horizontal", length=500, mode="determinate")
        self.progressbar.pack(pady=10)

        self.status_label = ttk.Label(self, text="", font=("Helvetica", 12))
        self.status_label.pack(pady=10)

    def browse_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.save_path_var.set(folder_selected)

    def start_download(self):
        url = self.url_entry.get()
        format_choice = self.format_var.get()
        save_path = self.save_path_var.get()

        if not url:
            messagebox.showerror("Eroare", "URL necompletat")
            return
        if not os.path.isdir(save_path):
            messagebox.showerror("Eroare", "Cale de salvare invalidă")
            return

        self.download_button.config(state="disabled")
        self.status_label.config(text="Descărcare în desfășurare...")

        downloader = YouTubeDownloader()
        thread = threading.Thread(target=self.download_video, args=(url, format_choice, save_path, downloader))
        thread.start()

    def download_video(self, url, format_choice, save_path, downloader):
        success, message = downloader.download(url, format_choice, save_path, self.update_progress)
        self.update_status(success, message)
        self.download_button.config(state="normal")

    def update_progress(self, percent, status_str):
        self.after(0, lambda: self.progressbar.configure(value=percent))
        self.after(0, lambda: self.status_label.config(text=status_str))

    def update_status(self, success, message):
        if success:
            messagebox.showinfo("Succes", message)
        else:
            messagebox.showerror("Eroare", message)
