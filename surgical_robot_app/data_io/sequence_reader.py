import pathlib
import os
from typing import List, Dict, Callable, Tuple, Optional
import numpy as np
from glob import glob

try:
    import pydicom  # type: ignore
except Exception:
    pydicom = None

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


class SequenceReader:
    def __init__(self):
        self._handlers: Dict[str, Callable[[str], Dict]] = {}

    def register_handler(self, key: str, loader: Callable[[str], Dict]) -> None:
        self._handlers[key.lower()] = loader

    def read_sequence(self, directory: str) -> List[str]:
        p = pathlib.Path(directory)
        return sorted([str(x) for x in p.glob("*") if x.is_file()])

    def read_files(self, paths: List[str]) -> List[Dict]:
        items: List[Dict] = []
        for path in paths:
            ext = os.path.splitext(path)[1].lower()
            handler = self._handlers.get(ext.replace(".", ""))
            if handler is not None:
                try:
                    item = handler(path)
                except Exception:
                    item = {"path": path, "format": "External", "data": None}
                items.append(item)
                continue
            if ext == ".dcm":
                fmt = "DCM"
                data = None
                if pydicom is not None:
                    try:
                        data = pydicom.dcmread(path)
                    except Exception:
                        data = None
                items.append({"path": path, "format": fmt, "data": data})
            elif ext == ".png":
                fmt = "PNG"
                img = None
                if cv2 is not None:
                    try:
                        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    except Exception:
                        img = None
                items.append({"path": path, "format": fmt, "data": img})
            else:
                items.append({"path": path, "format": "Unknown", "data": None})
        return items

    def read_with_external(self, func: Callable[[str], Dict], paths: List[str]) -> List[Dict]:
        return [func(p) for p in paths]

    def guess_format(self, image_dir: str) -> Optional[str]:
        dcm = glob(os.path.join(image_dir, "*.dcm"))
        png = glob(os.path.join(image_dir, "*.png"))
        if dcm:
            return "dicom"
        if png:
            return "png"
        return None

    def get_series_paths(self, image_dir: str, image_format: str) -> List[str]:
        if image_format.lower() == "dicom":
            return sorted(glob(os.path.join(image_dir, "*.dcm")))
        if image_format.lower() == "png":
            return sorted(glob(os.path.join(image_dir, "*.png")))
        return []

    def load_volume(self, image_dir: str, image_format: str) -> Tuple[np.ndarray, Dict]:
        metadata: Dict = {}
        slices: List[np.ndarray] = []
        if image_format.lower() == "dicom":
            dicom_files = self.get_series_paths(image_dir, "dicom")
            if not dicom_files:
                raise ValueError(f"No DICOM files in {image_dir}")
            for file in dicom_files:
                ds = pydicom.dcmread(file) if pydicom is not None else None
                img = ds.pixel_array if ds is not None else None
                if img is None:
                    continue
                if img.dtype != np.uint8:
                    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) if cv2 is not None else img.astype(np.uint8)
                slices.append(img)
                if not metadata and ds is not None:
                    px = getattr(ds, "PixelSpacing", [1.0, 1.0])
                    th = getattr(ds, "SliceThickness", 1.0)
                    metadata["PatientID"] = getattr(ds, "PatientID", "Unknown")
                    metadata["StudyDate"] = getattr(ds, "StudyDate", "Unknown")
                    metadata["Modality"] = getattr(ds, "Modality", "Unknown")
                    metadata["PixelSpacing"] = px
                    metadata["SliceThickness"] = th
                    metadata["Spacing"] = (float(px[0]), float(px[1]), float(th))
        elif image_format.lower() == "png":
            slice_files = self.get_series_paths(image_dir, "png")
            if not slice_files:
                raise ValueError(f"No PNG files in {image_dir}")
            for file in slice_files:
                img = cv2.imread(file, cv2.IMREAD_COLOR) if cv2 is not None else None
                if img is None:
                    continue
                slices.append(img)
            metadata["ImageFormat"] = "PNG"
            metadata["SliceCount"] = len(slices)
            metadata["Spacing"] = (0.68359, 0.68359, 1.0)
        else:
            raise ValueError(f"Unsupported format: {image_format}")
        volume = np.stack(slices, axis=0)
        return volume, metadata

    def load_volume_from_paths(self, paths: List[str], image_format: str) -> Tuple[np.ndarray, Dict]:
        metadata: Dict = {}
        slices: List[np.ndarray] = []
        if image_format.lower() == "dicom":
            files = sorted([p for p in paths if p.lower().endswith(".dcm")])
            if not files:
                raise ValueError("No DICOM files in paths")
            for file in files:
                ds = pydicom.dcmread(file) if pydicom is not None else None
                img = ds.pixel_array if ds is not None else None
                if img is None:
                    continue
                if img.dtype != np.uint8:
                    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) if cv2 is not None else img.astype(np.uint8)
                slices.append(img)
            if pydicom is not None and files:
                ds0 = pydicom.dcmread(files[0])
                px = getattr(ds0, "PixelSpacing", [1.0, 1.0])
                th = getattr(ds0, "SliceThickness", 1.0)
                metadata["Spacing"] = (float(px[0]), float(px[1]), float(th))
        elif image_format.lower() == "png":
            files = sorted([p for p in paths if p.lower().endswith(".png")])
            if not files:
                raise ValueError("No PNG files in paths")
            for file in files:
                img = cv2.imread(file, cv2.IMREAD_COLOR) if cv2 is not None else None
                if img is None:
                    continue
                slices.append(img)
            metadata["ImageFormat"] = "PNG"
            metadata["SliceCount"] = len(slices)
            metadata["Spacing"] = (0.68359, 0.68359, 1.0)
        else:
            raise ValueError(f"Unsupported format: {image_format}")
        volume = np.stack(slices, axis=0)
        return volume, metadata