"""Lightweight motion detection engine (CuPy GPU or CPU fallback)."""

from __future__ import annotations

import cv2
import logging
import numpy as np
from collections import deque

log = logging.getLogger("doorwatch.detector")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    cp = None
    CUPY_AVAILABLE = False

# Backward compatibility (face recognition removed).
FACE_REC_AVAILABLE = False


class PersonDetector:
    """
    Detects motion only; no person/object/face recognition.
    update() -> (confirmed_motion, boxes, has_motion)
    """

    def __init__(
        self,
        confidence: float = 0.0,  # legacy signature compatibility
        consecutive_frames: int = 2,
        known_faces_dir: str | None = None,  # legacy signature compatibility
        face_tolerance: float = 0.0,  # legacy signature compatibility
        min_contour_area: int = 2000,
        use_gpu: bool = True,
        max_active_ratio: float = 0.75,
        preprocess_grayscale: bool = True,
        filter_median: int = 3,
        filter_gaussian: int = 5,
        release_frames: int = 6,
        **_kwargs,
    ):
        self._min_contour_area = int(min_contour_area)
        self._consecutive_needed = max(1, int(consecutive_frames))
        self._release_frames = max(1, int(release_frames))

        self._history: deque[bool] = deque(maxlen=self._consecutive_needed)
        self._quiet_frames = 0
        self._max_active_ratio = float(np.clip(max_active_ratio, 0.05, 0.98))
        self._preprocess_grayscale = bool(preprocess_grayscale)
        self._filter_median = self._normalize_kernel(filter_median)
        self._filter_gaussian = self._normalize_kernel(filter_gaussian)

        self._use_cupy = False
        self._use_cuda = False
        self._gpu_frame = None
        self._gpu_mask = None

        self._cpu_bg_subtractor = None
        self._gpu_bg = None
        self._gpu_scale = 0.5
        self._gpu_diff_threshold = 10.0
        self._gpu_alpha_static = 0.04
        self._gpu_alpha_motion = 0.01
        self._gpu_min_pixels = max(64, int(self._min_contour_area * 0.05))

        self._bg_subtractor = None
        self._init_subtractor(use_gpu=use_gpu)

    @staticmethod
    def _normalize_kernel(size: int) -> int:
        k = int(size)
        if k <= 1:
            return 0
        if k % 2 == 0:
            k += 1
        return max(3, k)

    def _prepare_motion_frame(self, frame: np.ndarray) -> np.ndarray:
        proc = frame

        if self._preprocess_grayscale and proc.ndim == 3:
            proc = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

        if self._filter_median:
            proc = cv2.medianBlur(proc, self._filter_median)

        if self._filter_gaussian:
            proc = cv2.GaussianBlur(proc, (self._filter_gaussian, self._filter_gaussian), 0)

        return proc

    def _init_subtractor(self, use_gpu: bool) -> None:
        if use_gpu:
            # 1) Preferred: CuPy (does not require OpenCV CUDA build)
            if CUPY_AVAILABLE:
                try:
                    if cp.cuda.runtime.getDeviceCount() > 0:
                        self._use_cupy = True
                        self._bg_subtractor = "cupy"
                        log.info("Motion detection running on GPU (CuPy).")
                        return
                except Exception as exc:
                    log.warning("CuPy GPU unavailable, trying fallbacks: %s", exc)

            # 2) Optional: OpenCV CUDA MOG2 (if available)
            try:
                cuda_ok = (
                    hasattr(cv2, "cuda")
                    and cv2.cuda.getCudaEnabledDeviceCount() > 0
                    and hasattr(cv2, "cuda_GpuMat")
                    and hasattr(cv2.cuda, "createBackgroundSubtractorMOG2")
                )
                if cuda_ok:
                    self._bg_subtractor = cv2.cuda.createBackgroundSubtractorMOG2(
                        history=500, varThreshold=50, detectShadows=True
                    )
                    self._gpu_frame = cv2.cuda_GpuMat()
                    self._use_cuda = True
                    log.info("Motion detection running on GPU (OpenCV CUDA MOG2).")
                    return
            except Exception as exc:
                log.warning("Failed to initialize GPU motion detection, falling back to CPU: %s", exc)

        self._cpu_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        self._bg_subtractor = self._cpu_bg_subtractor
        self._use_cuda = False
        self._use_cupy = False
        log.info("Motion detection running on CPU.")

    def _detect_motion_gpu_cupy(
        self,
        frame: np.ndarray,
        min_contour_area: int | None = None,
    ) -> tuple[bool, list[tuple[int, int, int, int]]]:
        min_area = self._min_contour_area if min_contour_area is None else int(min_contour_area)
        min_pixels = max(32, int(min_area * 0.05))
        proc = self._prepare_motion_frame(frame)
        if proc.ndim == 3:
            proc = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(
            proc,
            None,
            fx=self._gpu_scale,
            fy=self._gpu_scale,
            interpolation=cv2.INTER_AREA,
        )
        gpu_frame = cp.asarray(small, dtype=cp.float32)

        if self._gpu_bg is None:
            self._gpu_bg = gpu_frame
            return False, []

        diff = cp.abs(gpu_frame - self._gpu_bg)
        motion_mask = diff >= self._gpu_diff_threshold
        active_pixels = int(cp.count_nonzero(motion_mask).item())
        total_pixels = int(motion_mask.size)
        active_ratio = (active_pixels / total_pixels) if total_pixels else 0.0

        # If almost the whole scene becomes "active", it is usually camera shake
        # or sudden lighting change. Ignore it for popup/rearm logic.
        if active_ratio >= self._max_active_ratio:
            fast_alpha = max(self._gpu_alpha_static, 0.20)
            self._gpu_bg = ((1.0 - fast_alpha) * self._gpu_bg) + (fast_alpha * gpu_frame)
            return False, []

        # Prevent sparse noisy pixels from becoming one huge box by cleaning
        # mask on CPU and extracting contour-based boxes.
        mask_u8 = cp.asnumpy(motion_mask.astype(cp.uint8)) * 255
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)

        sx = frame.shape[1] / small.shape[1]
        sy = frame.shape[0] / small.shape[0]
        area_scale = max(1e-6, sx * sy)
        min_area_small = max(6, int(min_area / area_scale))

        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects: list[tuple[int, int, int, int]] = []
        for contour in contours:
            if cv2.contourArea(contour) < min_area_small:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            rx = int(x * sx)
            ry = int(y * sy)
            rw = int(w * sx)
            rh = int(h * sy)
            if rw * rh >= min_area:
                rects.append((rx, ry, rw, rh))

        has_motion = len(rects) > 0 or active_pixels >= min_pixels

        alpha = self._gpu_alpha_motion if has_motion else self._gpu_alpha_static
        self._gpu_bg = ((1.0 - alpha) * self._gpu_bg) + (alpha * gpu_frame)
        return has_motion, rects

    def _detect_motion_cpu(
        self,
        frame: np.ndarray,
        min_contour_area: int | None = None,
    ) -> tuple[bool, list[tuple[int, int, int, int]]]:
        min_area = self._min_contour_area if min_contour_area is None else int(min_contour_area)
        proc = self._prepare_motion_frame(frame)
        fg_mask = self._cpu_bg_subtractor.apply(proc)
        fg_mask = cv2.threshold(fg_mask, 120, 255, cv2.THRESH_BINARY)[1]
        fg_mask = cv2.dilate(fg_mask, None, iterations=2)
        fg_mask = cv2.erode(fg_mask, None, iterations=1)
        active_ratio = cv2.countNonZero(fg_mask) / float(fg_mask.shape[0] * fg_mask.shape[1])

        if active_ratio >= self._max_active_ratio:
            return False, []

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects: list[tuple[int, int, int, int]] = []
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            rects.append((x, y, w, h))
        return len(rects) > 0, rects

    def detect_motion(
        self,
        frame: np.ndarray,
        min_contour_area: int | None = None,
    ) -> tuple[bool, list[tuple[int, int, int, int]]]:
        """
        Checks whether motion exists.
        Returns: (has_motion, boxes)
        """
        if self._use_cupy:
            try:
                return self._detect_motion_gpu_cupy(frame, min_contour_area=min_contour_area)
            except Exception as exc:
                log.warning("CuPy motion detection error, CPU fallback: %s", exc)
                self._use_cupy = False
                self._cpu_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=500, varThreshold=50, detectShadows=True
                )
                return self._detect_motion_cpu(frame, min_contour_area=min_contour_area)

        if self._use_cuda:
            proc = self._prepare_motion_frame(frame)
            self._gpu_frame.upload(proc)
            self._gpu_mask = self._bg_subtractor.apply(self._gpu_frame)
            fg_mask = self._gpu_mask.download()
            fg_mask = cv2.threshold(fg_mask, 120, 255, cv2.THRESH_BINARY)[1]
            fg_mask = cv2.dilate(fg_mask, None, iterations=2)
            fg_mask = cv2.erode(fg_mask, None, iterations=1)
            active_ratio = cv2.countNonZero(fg_mask) / float(fg_mask.shape[0] * fg_mask.shape[1])
            if active_ratio >= self._max_active_ratio:
                return False, []
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rects: list[tuple[int, int, int, int]] = []
            min_area = self._min_contour_area if min_contour_area is None else int(min_contour_area)
            for contour in contours:
                if cv2.contourArea(contour) < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                rects.append((x, y, w, h))
            return len(rects) > 0, rects

        return self._detect_motion_cpu(frame, min_contour_area=min_contour_area)

    def update(
        self,
        frame: np.ndarray,
        min_contour_area: int | None = None,
    ) -> tuple[bool, list[tuple[int, int, int, int]], bool]:
        has_motion, rects = self.detect_motion(frame, min_contour_area=min_contour_area)

        self._history.append(has_motion)

        if has_motion:
            self._quiet_frames = 0
        else:
            self._quiet_frames += 1
            if self._quiet_frames >= self._release_frames:
                self._history.clear()

        window_confirmed = (
            len(self._history) == self._consecutive_needed
            and all(self._history)
        )
        confirmed = window_confirmed

        return confirmed, rects, has_motion

    # Legacy API compatibility: face recognition is disabled in this mode.
    def identify_faces(self, frame: np.ndarray) -> list[str]:
        return []

    def get_last_recognitions(self) -> list[str]:
        return []

    def add_known_face_from_file(self, image_path: str, name: str) -> bool:
        return False

    def extract_primary_face(self, frame: np.ndarray) -> np.ndarray | None:
        return None

    def reset(self) -> None:
        self._history.clear()
        self._quiet_frames = 0
