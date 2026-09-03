"""Facial landmark extraction using MediaPipe Face Landmarker."""

from pathlib import Path


_LANDMARKER_CACHE = None


def _require_mediapipe():
    try:
        import mediapipe as mp

        return mp
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "mediapipe is not installed. Run: pip install -r requirements.txt"
        ) from exc


def _model_path():
    path = Path(__file__).resolve().parents[1] / "data" / "models" / "face_landmarker.task"
    if not path.exists():
        raise FileNotFoundError(
            "face_landmarker.task not found: %s. Download it from MediaPipe "
            "and place it in data/models/." % path
        )
    return str(path)


def _to_points(landmarks):
    return [{"x": p.x, "y": p.y, "z": p.z} for p in landmarks]


def _get_landmarker():
    """Return a process-wide IMAGE-mode Face Landmarker instance."""
    global _LANDMARKER_CACHE
    if _LANDMARKER_CACHE is not None:
        return _LANDMARKER_CACHE
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_model_path()),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
    )
    _LANDMARKER_CACHE = vision.FaceLandmarker.create_from_options(options)
    return _LANDMARKER_CACHE


def detect_frame(frame):
    """Detect 478 normalized landmarks from one BGR ndarray frame."""
    import cv2

    mp = _require_mediapipe()
    landmarker = _get_landmarker()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(image)
    if not result.face_landmarks:
        return []
    return _to_points(result.face_landmarks[0])


def detect_frame_detailed(frame):
    """Detect landmarks and ARKit-style blendshape scores from one BGR frame."""
    import cv2

    mp = _require_mediapipe()
    landmarker = _get_landmarker()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(image)
    if not result.face_landmarks:
        return {"landmarks": [], "blendshapes": {}}
    blendshapes = {}
    if result.face_blendshapes:
        for category in result.face_blendshapes[0]:
            blendshapes[category.category_name] = round(category.score, 4)
    return {
        "landmarks": _to_points(result.face_landmarks[0]),
        "blendshapes": blendshapes,
    }


def detect_image(image_path):
    """Detect 478 normalized landmarks from an image file.

    Returns a list of dicts: {"x": float, "y": float, "z": float}.
    """
    import cv2

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError("cannot read image: %s" % image_path)
    return detect_frame(frame)


def detect_video(video_path, sample_every_frames=15, max_frames=300):
    """Sample landmarks from a video at fixed frame intervals.

    Returns list of (timestamp_ms, landmarks) sorted by time.
    """
    return _detect_video_impl(video_path, sample_every_frames, max_frames)


def detect_video_detailed(video_path, sample_every_frames=15, max_frames=300):
    """Sample landmarks and blendshapes from a video at fixed frame intervals."""
    return _detect_video_impl(
        video_path, sample_every_frames, max_frames, detailed=True
    )


def detect_video_stats(video_path, sample_every_frames=15, max_frames=300):
    """Like detect_video, but also returns the number of sampled attempts."""
    frames, attempts = _detect_video_impl(
        video_path, sample_every_frames, max_frames, count_attempts=True
    )
    return frames, attempts


def detect_video_stats_detailed(video_path, sample_every_frames=15, max_frames=300):
    """Like detect_video_stats, but frames also carry blendshape scores."""
    frames, attempts = _detect_video_impl(
        video_path,
        sample_every_frames,
        max_frames,
        count_attempts=True,
        detailed=True,
    )
    return frames, attempts


def _detect_video_impl(
    video_path, sample_every_frames, max_frames, count_attempts=False, detailed=False
):
    import cv2

    mp = _require_mediapipe()
    landmarker = _get_landmarker()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frames = []
    attempts = 0
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or len(frames) >= max_frames:
                break
            if index % sample_every_frames == 0:
                attempts += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(image)
                if result.face_landmarks:
                    points = _to_points(result.face_landmarks[0])
                    ts_ms = int(index * 1000.0 / fps)
                    if detailed:
                        blendshapes = {}
                        if result.face_blendshapes:
                            for category in result.face_blendshapes[0]:
                                blendshapes[category.category_name] = round(
                                    category.score, 4
                                )
                        frames.append((ts_ms, points, blendshapes))
                    else:
                        frames.append((ts_ms, points))
            index += 1
    finally:
        cap.release()
    if count_attempts:
        return frames, attempts
    return frames
