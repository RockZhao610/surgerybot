import pathlib
import os
from typing import List
import numpy as np
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


class FileHandler:
    def save_mask(self, path, data):
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if cv2 is not None and isinstance(data, np.ndarray):
            cv2.imwrite(str(p), data)
        else:
            p.write_bytes(b"")

    def save_masks(self, directory: str, masks: List[np.ndarray], progress_callback=None):
        d = pathlib.Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        total = len(masks)
        for i, m in enumerate(masks):
            out = d / f"mask_{i:04d}.png"
            if cv2 is not None and m is not None:
                cv2.imwrite(str(out), m)
            else:
                out.touch()
            
            if progress_callback:
                progress_callback(int((i + 1) / total * 100))

    def save_path(self, path, points):
        p = pathlib.Path(path)
        p.write_text("")