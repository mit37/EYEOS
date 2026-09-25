"""The CSV row format shared by ``eval/generate_synthetic.py`` (writer) and
``eyeos.input_sources.synthetic.SyntheticInputSource`` (reader).

One row = one frame: a timestamp, a face count, a detection confidence, and (if
``face_count >= 1``) the 15 named landmark points as flat ``<point>_x/_y/_z`` columns.
A ``face_count`` of 0 leaves the point columns blank — that's how a "no face this
frame" tick round-trips through the CSV.
"""

from __future__ import annotations

from eyeos.landmarks import POINT_FIELDS, FaceLandmarks, Point3D

CSV_FIELDNAMES: list[str] = ["timestamp_s", "face_count", "detection_confidence"] + [
    f"{name}_{axis}" for name in POINT_FIELDS for axis in ("x", "y", "z")
]


def landmarks_to_row(timestamp_s: float, landmarks: FaceLandmarks | None) -> dict[str, str]:
    row: dict[str, str] = {
        "timestamp_s": repr(timestamp_s),
        "face_count": str(landmarks.face_count if landmarks else 0),
        "detection_confidence": repr(landmarks.detection_confidence if landmarks else 0.0),
    }
    for name in POINT_FIELDS:
        point = getattr(landmarks, name) if landmarks else None
        row[f"{name}_x"] = repr(point.x) if point else ""
        row[f"{name}_y"] = repr(point.y) if point else ""
        row[f"{name}_z"] = repr(point.z) if point else ""
    return row


def row_to_frame(row: dict[str, str]) -> tuple[float, FaceLandmarks | None]:
    timestamp_s = float(row["timestamp_s"])
    face_count = int(row["face_count"])
    detection_confidence = float(row["detection_confidence"])

    if face_count < 1 or row.get(f"{POINT_FIELDS[0]}_x", "") == "":
        return timestamp_s, None

    points = {
        name: Point3D(
            x=float(row[f"{name}_x"]),
            y=float(row[f"{name}_y"]),
            z=float(row[f"{name}_z"]) if row[f"{name}_z"] else 0.0,
        )
        for name in POINT_FIELDS
    }
    landmarks = FaceLandmarks(
        face_count=face_count,
        detection_confidence=detection_confidence,
        **points,
    )
    return timestamp_s, landmarks
