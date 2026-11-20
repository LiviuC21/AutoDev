import os
import yt_dlp
from typing import Callable, Optional, Tuple

class YouTubeDownloader:
    def download(self, url: str, format_choice: str, save_path: str, progress_callback: Optional[Callable[[float, str], None]] = None) -> Tuple[bool, str]:
        ydl_opts = {
            'outtmpl': os.path.join(save_path, '%(title)s.%(ext)s'),
        }
        if format_choice == 'MP3':
            ydl_opts.update({
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
            })
        elif format_choice == 'MP4':
            ydl_opts.update({
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'merge_output_format': 'mp4',
            })

        def progress_hook(d):
            if d['status'] == 'downloading':
                if 'downloaded_bytes' in d and 'total_bytes' in d and d['total_bytes'] > 0:
                    percent = d['downloaded_bytes'] / d['total_bytes'] * 100
                elif 'downloaded_bytes' in d and 'total_bytes_estimate' in d and d['total_bytes_estimate'] > 0:
                    percent = d['downloaded_bytes'] / d['total_bytes_estimate'] * 100
                else:
                    percent = 0
                status_str = f"Descărcare {percent:.2f}%"
                if progress_callback:
                    progress_callback(percent, status_str)

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.add_progress_hook(progress_hook)
                ydl.download([url])
            return True, 'Descărcare completă!'
        except Exception as e:
            return False, str(e)
