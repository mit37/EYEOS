"""eval/public/diagnostics.py and the ``diagnose`` subcommand: where the public-dataset
error comes from.

Every test builds fake participants whose truth is known: noise-free faces, faces whose
iris offsets shift with the (dataset's) head pose, faces with pure landmark noise, faces
with a different iris bias on each day, and EyeOS head poses that are a known multiple
of the dataset's. The faces are eval.generate_synthetic.make_landmarks, so no images,
OpenCV or MediaPipe are involved. The tests that write the dataset's .mat files need
scipy and skip without it (it is in the ``dev`` extra, so CI runs them).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import platform
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eval.generate_synthetic import make_landmarks
from eval.public import diagnostics, models, protocol
from eval.public import mpiifacegaze as mpii
from eyeos.calibration import Calibration, target_points
from eyeos.features import FeatureVector, extract_features
from eyeos.landmarks import FaceLandmarks, Point3D
from tests.fake_mpiifacegaze import FakePerson, offsets_for_target, write_calibration

W, H = 1280.0, 800.0


# ---------------------------------------------------------------------------
# fake participants with a known truth
# ---------------------------------------------------------------------------


def _rot_x(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """The rotation the documented convention reads back as (yaw, pitch, roll)."""
    return _rot_y(yaw) @ _rot_x(pitch) @ _rot_z(roll)


# Head frames whose axes are labelled the other way round (a half turn about one axis).
_AXES = {
    "none": np.eye(3),
    "x180": np.diag([1.0, -1.0, -1.0]),
    "y180": np.diag([-1.0, 1.0, -1.0]),
    "z180": np.diag([-1.0, -1.0, 1.0]),
}


def _rotate(face: FaceLandmarks, deg: float) -> FaceLandmarks:
    """Every landmark turned by ``deg`` about (0.5, 0.5) in the image: a head roll."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))

    def turn(p: Point3D) -> Point3D:
        dx, dy = p.x - 0.5, p.y - 0.5
        return Point3D(0.5 + c * dx - s * dy, 0.5 + s * dx + c * dy, p.z)

    changes = {
        f.name: turn(getattr(face, f.name))
        for f in dataclasses.fields(face)
        if isinstance(getattr(face, f.name), Point3D)
    }
    return dataclasses.replace(face, **changes)


def _face(
    left_x: float,
    left_y: float,
    right_x: float,
    right_y: float,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> FaceLandmarks:
    """make_landmarks with each eye's iris offset set on its own, and a nose placed so
    that eyeos.features reads exactly ``yaw_deg`` and ``pitch_deg`` (its face edges are
    0.2 either side of x = 0.5, forehead and chin 0.2 either side of y = 0.5)."""
    left = make_landmarks(left_x, left_y, nose_x=0.5 + yaw_deg * 0.2 / 60.0)
    right = make_landmarks(right_x, right_y)
    face = dataclasses.replace(
        left,
        right_iris_center=right.right_iris_center,
        nose_tip=Point3D(left.nose_tip.x, 0.5 + pitch_deg * 0.2 / 50.0),
    )
    return _rotate(face, roll_deg) if roll_deg else face


@dataclass(frozen=True)
class Spec:
    """One fake participant. Targets are uniform on a 1280x800 screen; the dataset's head
    yaw, pitch and roll are uniform in +-their range, independent of the target unless
    ``yaw_per_screen_deg`` makes the yaw follow the target's x too."""

    name: str = "p00"
    days: tuple[int, ...] = (150, 150, 150)  # frames per dayNN folder
    seed: int = 0
    noise: float = 0.0  # Gaussian sigma, independently per eye and axis: landmark noise
    yaw_range_deg: float = 0.0
    pitch_range_deg: float = 0.0
    roll_range_deg: float = 0.0
    shift_per_yaw_deg: float = 0.0  # iris x offset added per degree of the dataset's yaw
    shift_per_roll_deg: float = 0.0  # iris x offset added per degree of the dataset's roll
    shift_per_mm: float = 0.0  # iris x offset added per mm of the head's sideways position
    # The head turns toward where the eyes look: yaw changes by this many degrees from the
    # screen's left edge to its right (on top of the uniform +-yaw_range_deg).
    yaw_per_screen_deg: float = 0.0
    day_bias: tuple[float, ...] = ()  # iris offset (x and y) added on each day
    eyeos_scale: tuple[float, float] = (1.0, 1.0)  # EyeOS yaw, pitch = scale * dataset's
    roll_landmarks: bool = False  # turn the landmarks by the dataset's roll
    axes: str = "none"  # how the stored head frame's axes are labelled
    no_face: frozenset[int] = frozenset()  # frame indices without a face


@dataclass(frozen=True)
class Frame:
    image: str
    target: tuple[float, float]
    face: dict[str, float] | None  # _face's arguments; None = no face found
    rotation: np.ndarray  # the dataset's head rotation, head -> camera
    tvec: tuple[float, float, float]


def _frames(spec: Spec) -> list[Frame]:
    rng = random.Random(spec.seed)
    frames: list[Frame] = []
    for day, count in enumerate(spec.days):
        bias = spec.day_bias[day] if spec.day_bias else 0.0
        for _ in range(count):
            index = len(frames)
            x, y = rng.uniform(0, W - 1), rng.uniform(0, H - 1)
            yaw = rng.uniform(-spec.yaw_range_deg, spec.yaw_range_deg)
            yaw += spec.yaw_per_screen_deg * (x / W - 0.5)
            pitch = rng.uniform(-spec.pitch_range_deg, spec.pitch_range_deg)
            roll = rng.uniform(-spec.roll_range_deg, spec.roll_range_deg)
            noise = [rng.gauss(0.0, spec.noise) for _ in range(4)]
            tvec = (rng.uniform(-30, 30), rng.uniform(-30, 30), rng.uniform(450, 650))
            ox, oy = offsets_for_target(x, y, W, H)
            ox += spec.shift_per_yaw_deg * yaw + spec.shift_per_roll_deg * roll + bias
            ox += spec.shift_per_mm * tvec[0]
            oy += bias
            face = None
            if index not in spec.no_face:
                face = {
                    "left_x": ox + noise[0],
                    "left_y": oy + noise[1],
                    "right_x": ox + noise[2],
                    "right_y": oy + noise[3],
                    "yaw_deg": spec.eyeos_scale[0] * yaw,
                    "pitch_deg": spec.eyeos_scale[1] * pitch,
                    "roll_deg": roll if spec.roll_landmarks else 0.0,
                }
            rotation = _rotation(yaw, pitch, roll) @ _AXES[spec.axes]
            frames.append(Frame(f"day{day + 1:02d}/{index:04d}.jpg", (x, y), face, rotation, tvec))
    return frames


def _person(
    spec: Spec, with_days: bool = True, with_head: bool = True, with_3d: bool = False
) -> Any:
    frames = _frames(spec)
    data = protocol.PersonData(
        spec.name,
        (W, H),
        np.array([f.target for f in frames]),
        [None if f.face is None else _face(**f.face) for f in frames],
    )
    if with_3d:
        # The fake dataset's screen model (tests/fake_mpiifacegaze.py), for angular error.
        screen = FakePerson(spec.name)

        def to_camera_mm(points_px: np.ndarray) -> np.ndarray:
            return np.array([screen.gaze_target_mm(x, y) for x, y in points_px])

        data = dataclasses.replace(
            data,
            face_centers_mm=np.tile(screen.face_center, (len(frames), 1)),
            gaze_targets_mm=to_camera_mm(data.targets_px),
            to_camera_mm=to_camera_mm,
        )
    return diagnostics.DiagnosticPerson(
        data=data,
        days=[f.image.split("/")[0] for f in frames] if with_days else None,
        head_rotation=np.array([f.rotation for f in frames]) if with_head else None,
        head_translation_mm=np.array([f.tvec for f in frames]) if with_head else None,
    )


def _diagnose(*specs: Spec, **kwargs: Any) -> dict[str, Any]:
    result: dict[str, Any] = diagnostics.diagnose([_person(s) for s in specs], **kwargs)
    return result


def _one(spec: Spec, **kwargs: Any) -> dict[str, Any]:
    person: dict[str, Any] = _diagnose(spec, **kwargs)["persons"][spec.name]
    return person


# ---------------------------------------------------------------------------
# the fake itself
# ---------------------------------------------------------------------------


def test_the_fake_face_reads_back_the_offsets_and_pose_it_was_built_with():
    f = extract_features(_face(0.3, -0.4, -0.2, 0.5, yaw_deg=12.0, pitch_deg=-7.0))
    assert (f.left_iris_x, f.right_iris_x) == pytest.approx((0.3, -0.2))
    # make_landmarks draws y in lid half-heights, which are half the eye half-width.
    assert (f.left_iris_y, f.right_iris_y) == pytest.approx((-0.2, 0.25))
    assert (f.yaw_deg, f.pitch_deg, f.roll_deg) == pytest.approx((12.0, -7.0, 0.0))

    rolled = extract_features(_face(0.3, -0.4, -0.2, 0.5, roll_deg=8.0))
    assert rolled.roll_deg == pytest.approx(8.0)
    assert rolled.left_iris_x == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# a. feature sanity
# ---------------------------------------------------------------------------


def test_feature_sanity_of_noise_free_faces_shows_both_eyes_moving_together():
    person = _one(Spec())

    sanity = person["feature_sanity"]
    assert person["n_images"] == 450 and person["n_valid"] == 450
    # x offsets span -1..1 across the screen; y is read back at half scale.
    assert sanity["left_iris_x"]["p5"] < -0.8 and sanity["left_iris_x"]["p95"] > 0.8
    assert abs(sanity["left_iris_x"]["median"]) < 0.15
    assert sanity["left_iris_y"]["p95"] == pytest.approx(sanity["right_iris_y"]["p95"])
    assert sanity["corr_left_right_x"] == pytest.approx(1.0)
    assert sanity["corr_left_right_y"] == pytest.approx(1.0)
    assert sanity["corr_left_x_target_x"] == pytest.approx(1.0)
    assert sanity["corr_right_x_target_x"] == pytest.approx(1.0)
    assert sanity["corr_left_y_target_y"] == pytest.approx(1.0)
    assert sanity["target_x_signs_match"] is True
    assert sanity["eyeos_roll_deg"] == {"median": 0.0, "p5": 0.0, "p95": 0.0}


def test_feature_sanity_flags_an_eye_whose_x_runs_the_wrong_way():
    person = diagnostics.diagnose([_person(Spec())])["persons"]["p00"]
    assert person["feature_sanity"]["target_x_signs_match"] is True

    flipped = _person(Spec())
    flipped.data.landmarks = [
        dataclasses.replace(
            lm,
            right_iris_center=Point3D(
                2 * 0.62 - lm.right_iris_center.x, lm.right_iris_center.y
            ),  # mirrored about the right eye's centre
        )
        for lm in flipped.data.landmarks
    ]

    sanity = diagnostics.diagnose([flipped])["persons"]["p00"]["feature_sanity"]
    assert sanity["corr_right_x_target_x"] == pytest.approx(-1.0)
    assert sanity["corr_left_right_x"] == pytest.approx(-1.0)
    assert sanity["target_x_signs_match"] is False
    # A mirrored eye axis keeps the iris inside the eye: only the sign check sees it.
    assert sanity["x_medians_within_eye"] is True


def test_feature_sanity_flags_irises_paired_with_the_wrong_eyes():
    healthy = _person(Spec(noise=0.02))
    assert diagnostics.diagnose([healthy])["persons"]["p00"]["feature_sanity"][
        "x_medians_within_eye"
    ]

    swapped = _person(Spec(noise=0.02))
    swapped.data.landmarks = [
        dataclasses.replace(
            lm, left_iris_center=lm.right_iris_center, right_iris_center=lm.left_iris_center
        )
        for lm in swapped.data.landmarks
    ]

    sanity = diagnostics.diagnose([swapped])["persons"]["p00"]["feature_sanity"]
    # Both offsets still follow the target the same way, so the sign check passes ...
    assert sanity["target_x_signs_match"] is True
    assert sanity["corr_left_right_x"] > 0.9
    # ... but each iris sits several half-widths outside its eye.
    assert sanity["left_iris_x"]["median"] > 3 and sanity["right_iris_x"]["median"] < -3
    assert sanity["x_medians_within_eye"] is False

    # One eye is enough: both irises taken from the left eye.
    one = _person(Spec(noise=0.02))
    one.data.landmarks = [
        dataclasses.replace(lm, right_iris_center=lm.left_iris_center) for lm in one.data.landmarks
    ]
    sanity = diagnostics.diagnose([one])["persons"]["p00"]["feature_sanity"]
    assert abs(sanity["left_iris_x"]["median"]) < 0.5
    assert sanity["x_medians_within_eye"] is False


def test_feature_sanity_reads_the_eyeos_roll_spread():
    sanity = _one(Spec(roll_range_deg=10.0, roll_landmarks=True))["feature_sanity"]

    roll = sanity["eyeos_roll_deg"]
    assert -10.0 <= roll["p5"] < -7.0
    assert 7.0 < roll["p95"] <= 10.0


def test_feature_sanity_and_the_ceiling_read_each_eyes_y_on_its_own():
    person = _person(Spec())
    rng = np.random.default_rng(5)
    # The left eye's y keeps the target with some noise; the right eye's y ignores it.
    person.data.landmarks = [
        dataclasses.replace(
            lm,
            left_iris_center=Point3D(
                lm.left_iris_center.x, lm.left_iris_center.y + rng.normal(0.0, 0.006)
            ),
            right_iris_center=Point3D(lm.right_iris_center.x, 0.40 + rng.normal(0.0, 0.01)),
        )
        for lm in person.data.landmarks
    ]

    result = diagnostics.diagnose([person])["persons"]["p00"]

    sanity, both = result["feature_sanity"], result["in_sample_ceiling"]["both_eyes"]
    assert sanity["corr_left_y_target_y"] > 0.8
    assert abs(sanity["corr_right_y_target_y"]) < 0.2
    assert abs(sanity["corr_left_right_y"]) < 0.2
    # x is untouched and fits exactly; y does not.
    assert both["r2_x"] == pytest.approx(1.0)
    assert 0.5 < both["r2_y"] < 0.95


def test_landmark_noise_lowers_the_agreement_between_the_eyes():
    sanity = _one(Spec(noise=0.3))["feature_sanity"]

    # Signal variance 1/3 against noise variance 0.09 per eye: corr = 0.333 / 0.423.
    assert 0.65 < sanity["corr_left_right_x"] < 0.9


# ---------------------------------------------------------------------------
# b. in-sample ceiling, c. constant baselines
# ---------------------------------------------------------------------------


def test_the_in_sample_fit_is_the_shipping_calibration():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 0.3, (40, 4))
    y = rng.uniform(0, 1000, (40, 2))
    calibration = Calibration(W, H)
    for row, target in zip(x, y, strict=True):
        f = FeatureVector(*row, 0.0, 0.0, 0.0)
        calibration.add_sample((float(target[0]), float(target[1])), f)
    calibration.fit()

    weights = diagnostics.shipping_affine_fit(x, y)
    predicted = diagnostics.affine_predict(weights, x)

    expected = [calibration.predict(FeatureVector(*row, 0.0, 0.0, 0.0)) for row in x]
    assert predicted == pytest.approx(np.array(expected), abs=1e-9)


def test_noise_free_ceiling_is_about_0_px_for_both_eyes_each_eye_and_with_head_pose():
    ceiling = _one(Spec(yaw_range_deg=10.0, pitch_range_deg=8.0))["in_sample_ceiling"]

    for model in ("both_eyes", "left_eye_only", "right_eye_only", "with_head_pose"):
        # Rounding to whole pixels, as the cursor does, is all that is left.
        assert ceiling[model]["mean_px"] < 0.75, model
        assert ceiling[model]["rms_px"] < 0.75, model
        assert ceiling[model]["r2_x"] == pytest.approx(1.0), model
        assert ceiling[model]["r2_y"] == pytest.approx(1.0), model


def test_the_with_head_pose_ceiling_uses_eyeos_roll_too():
    # The iris offsets shift with the head's roll, which EyeOS reads from the eye corners.
    ceiling = _one(Spec(roll_range_deg=10.0, roll_landmarks=True, shift_per_roll_deg=0.02))[
        "in_sample_ceiling"
    ]

    assert ceiling["both_eyes"]["mean_px"] > 30
    assert ceiling["with_head_pose"]["mean_px"] < 1.0


def test_the_ceiling_of_a_noisy_person_is_proportionate_and_shows_a_dead_eye():
    spec = Spec(noise=0.05)
    person = _person(spec)
    # The right eye's iris never moves: its features carry no signal at all.
    person.data.landmarks = [
        dataclasses.replace(lm, right_iris_center=Point3D(0.62, 0.40))
        for lm in person.data.landmarks
    ]

    ceiling = diagnostics.diagnose([person])["persons"]["p00"]["in_sample_ceiling"]

    # Noise 0.05 (one eye) on a +-1 offset per half screen: about 0.05 * 640 = 32 px in x.
    assert 10 < ceiling["both_eyes"]["mean_px"] < 60
    assert ceiling["left_eye_only"]["mean_px"] == pytest.approx(
        ceiling["both_eyes"]["mean_px"], rel=0.01
    )
    assert ceiling["right_eye_only"]["r2_x"] == pytest.approx(0.0, abs=1e-9)
    assert ceiling["right_eye_only"]["mean_px"] > 200


def test_a_per_day_bias_inflates_the_all_days_ceiling_but_not_a_calibration_per_day():
    # The all-days ceiling bounds ONE calibration used across every day. A different iris
    # offset on each day defeats that, yet a calibration made within each day absorbs it.
    person = _one(Spec(noise=0.005, day_bias=(0.3, -0.3, 0.0)))

    ceiling, within = person["in_sample_ceiling"], person["within_day"]
    assert ceiling["both_eyes"]["rms_px"] > 60
    assert ceiling["per_day"]["days"] == 3
    assert ceiling["per_day"]["mean_px"] < 10
    assert ceiling["per_day"]["rms_px"] < 10
    # Gate 2 judges the calibration set's leave-one-out RMS, and it passes on every day.
    assert within["validation_error_px_mean"] < 60
    assert within["gate2_passed_days"] == within["days_used"] == 3


def test_the_per_day_ceiling_fits_each_day_within_day_uses_on_its_own():
    spec = Spec(days=(150, 30, 70), noise=0.05, day_bias=(0.1, -0.1, 0.05))
    person = _person(spec)

    ceiling = diagnostics.diagnose([person])["persons"]["p00"]["in_sample_ceiling"]["per_day"]

    # day02 has 30 images, below the within-day minimum of 60: it is left out here too.
    means, rms = [], []
    for day in ("day01", "day03"):
        rows = [i for i, d in enumerate(person.days) if d == day]
        features = [extract_features(person.data.landmarks[i]) for i in rows]
        iris = np.array([f.iris_vector() for f in features])
        targets = np.asarray(person.data.targets_px)[rows]
        weights = diagnostics.shipping_affine_fit(iris, targets)
        clamped = protocol.clamp_to_screen(diagnostics.affine_predict(weights, iris), W, H)
        errors = np.hypot(*(clamped - targets).T)
        means.append(float(np.mean(errors)))
        rms.append(float(np.sqrt(np.mean(errors**2))))
    assert ceiling["days"] == 2
    assert ceiling["mean_px"] == pytest.approx(np.mean(means), rel=1e-9)
    assert ceiling["rms_px"] == pytest.approx(np.mean(rms), rel=1e-9)
    assert _one(spec, min_day_images=100)["in_sample_ceiling"]["per_day"]["days"] == 1


def test_the_per_day_ceiling_is_null_without_a_usable_day():
    without = diagnostics.diagnose([_person(Spec(noise=0.05), with_days=False)])
    assert without["persons"]["p00"]["in_sample_ceiling"]["per_day"] is None
    short = _one(Spec(days=(40, 40), noise=0.05))
    assert short["in_sample_ceiling"]["per_day"] is None
    assert short["in_sample_ceiling"]["both_eyes"]["mean_px"] > 0


def test_constant_baselines_predict_the_centre_and_the_persons_own_mean_target():
    spec = Spec(days=(100,), no_face=frozenset({3, 4}))
    person = _one(spec)

    targets = np.array([f.target for i, f in enumerate(_frames(spec)) if i not in (3, 4)])
    centre = protocol.clamp_to_screen(np.array([[W / 2, H / 2]]), W, H)
    own = protocol.clamp_to_screen(targets.mean(axis=0, keepdims=True), W, H)
    baselines = person["constant_baselines"]
    assert person["n_valid"] == 98
    assert baselines["screen_centre_mean_px"] == pytest.approx(
        float(np.mean(np.hypot(*(targets - centre).T)))
    )
    assert baselines["own_mean_target_mean_px"] == pytest.approx(
        float(np.mean(np.hypot(*(targets - own).T)))
    )


def test_the_constant_baselines_say_they_are_in_sample_over_all_valid_frames():
    # Not to be confused with a constant baseline scored by the protocol on test frames.
    result = _diagnose(Spec(days=(60,)))

    for text in (
        result["sections"]["constant_baselines"],
        result["metrics"]["constant_baselines.screen_centre_mean_px"],
        result["metrics"]["constant_baselines.own_mean_target_mean_px"],
    ):
        assert "all valid frames" in text
    assert "test frames" in result["sections"]["constant_baselines"]


# ---------------------------------------------------------------------------
# d. the standard (cross-day) protocol against a within-day one
# ---------------------------------------------------------------------------


def test_cross_day_reproduces_evaluate_for_the_shipping_calibration():
    spec = Spec(noise=0.05, yaw_range_deg=10.0)
    person = _person(spec, with_3d=True)

    cross = diagnostics.diagnose([person])["persons"]["p00"]["cross_day"]

    evaluated = protocol.evaluate([person.data], [models.build_variants()["eyeos"]])
    expected = evaluated["variants"]["eyeos"]["persons"]["p00"]
    # The same numbers, to the 10 significant digits the diagnostics JSON keeps.
    assert cross["mean_px"] == pytest.approx(expected["pixel_error_px"]["mean"], rel=1e-9)
    assert cross["mean_deg"] == pytest.approx(expected["angular_error_deg"]["mean"], rel=1e-9)
    assert cross["validation_error_px"] == pytest.approx(expected["validation_error_px"], rel=1e-9)
    assert cross["passes_gate2"] == expected["passes_gate2"]
    assert cross["n_calibration"] == 45 and cross["n_test"] == expected["n_test"] == 405
    assert cross["calibration_mean_distance_px"] == pytest.approx(
        expected["calibration_mean_distance_px"]
    )
    # Targets are uniform over all three days, so the 45 calibration images mix them.
    assert 2 <= cross["calibration_days"] <= 3


def test_noise_free_within_day_calibration_is_about_0_px_on_every_day():
    within = _one(Spec(), min_day_images=60)["within_day"]

    assert within["days_total"] == within["days_used"] == 3
    assert within["days_skipped"] == 0 and within["skipped_days"] == []
    assert [d["day"] for d in within["per_day"]] == ["day01", "day02", "day03"]
    for day in within["per_day"]:
        assert day["n_valid"] == 150
        assert day["n_calibration"] == 45 and day["n_test"] == 105
        assert day["mean_px"] < 0.75, day
        assert day["validation_error_px"] < 0.5, day
        assert day["passes_gate2"] is True
    assert within["mean_px"] < 0.75
    assert within["gate2_passed_days"] == 3


def test_within_day_angular_error_uses_each_days_own_3d_labels():
    person = _person(Spec(days=(80, 80)), with_3d=True)

    result = diagnostics.diagnose([person])["persons"]["p00"]

    assert result["within_day"]["mean_deg"] < 0.02
    assert result["cross_day"]["mean_deg"] < 0.02
    for day in result["within_day"]["per_day"]:
        assert day["mean_deg"] < 0.02


def test_a_per_day_iris_bias_hurts_the_cross_day_protocol_but_not_the_within_day_one():
    person = _one(Spec(noise=0.005, day_bias=(0.15, -0.15, 0.0)))

    within, cross = person["within_day"], person["cross_day"]
    # 0.15 of an eye half-width is about 0.15 * 640 = 96 px in x and 60 px in y.
    assert cross["mean_px"] > 40
    assert within["mean_px"] < 10
    assert within["cross_day_mean_px"] == cross["mean_px"]
    assert within["cross_day_validation_error_px"] == cross["validation_error_px"]
    assert within["within_over_cross_mean_px"] == pytest.approx(
        within["mean_px"] / within["cross_day_mean_px_on_used_days"]
    )
    assert within["within_over_cross_mean_px"] < 0.25


def test_within_day_summaries_are_the_means_of_the_per_day_values():
    person = _person(Spec(noise=0.05, day_bias=(0.05, -0.05, 0.0)), with_3d=True)

    result = diagnostics.diagnose([person])["persons"]["p00"]

    within, per_day = result["within_day"], result["within_day"]["per_day"]
    summaries = {
        "mean_px": "mean_px",
        "mean_deg": "mean_deg",
        "validation_error_px_mean": "validation_error_px",
        "calibration_mean_distance_px_mean": "calibration_mean_distance_px",
    }
    for summary, key in summaries.items():
        assert within[summary] == pytest.approx(np.mean([d[key] for d in per_day]), rel=1e-9)
    # The four differ, so a summary read from the wrong per-day value would show.
    assert len({round(within[s], 3) for s in summaries}) == 4
    cross = result["cross_day"]
    assert within["cross_day_mean_deg"] == cross["mean_deg"] and cross["mean_deg"] > 0


def test_calibration_days_counts_only_the_days_of_the_calibration_images():
    person = _person(Spec(noise=0.02))
    targets = np.asarray(person.data.targets_px)
    dots = np.array(target_points(W, H))
    near_a_dot = np.linalg.norm(targets[:, None, :] - dots[None], axis=2).min(axis=1) < 150
    # Every image near a dot is in day01; the rest are split between day02 and day03.
    person.days = [
        "day01" if near else ("day02" if i % 2 else "day03") for i, near in enumerate(near_a_dot)
    ]

    result = diagnostics.diagnose([person])["persons"]["p00"]

    assert result["within_day"]["days_total"] == 3
    # The K = 5 images nearest each dot all lie within 150 px of it, so all in day01.
    assert result["cross_day"]["calibration_mean_distance_px"] < 100
    assert result["cross_day"]["calibration_days"] == 1


def test_within_over_cross_compares_the_cross_day_error_on_the_same_days():
    # No drift between days. day03 is short, so within_day skips it, and noisy.
    person = _person(Spec(days=(150, 150, 40), noise=0.01))
    rng = np.random.default_rng(0)
    assert person.days is not None
    person.data.landmarks = [
        (
            dataclasses.replace(
                lm,
                left_iris_center=Point3D(
                    lm.left_iris_center.x + rng.normal(0.0, 0.012), lm.left_iris_center.y
                ),
                right_iris_center=Point3D(
                    lm.right_iris_center.x + rng.normal(0.0, 0.012), lm.right_iris_center.y
                ),
            )
            if day == "day03"
            else lm
        )
        for lm, day in zip(person.data.landmarks, person.days, strict=True)
    ]

    result = diagnostics.diagnose([person])["persons"]["p00"]

    # The cross-day errors per day, worked out here from the protocol itself.
    prepared = protocol.prepare(person.data)
    run = protocol.evaluate_person(
        models.build_variants()["eyeos"], prepared, [], 5, protocol.Limits()
    )
    assert run.pixel_errors is not None
    targets = np.asarray(person.data.targets_px)[prepared.valid_rows]
    selected, _ = protocol.select_calibration_set(targets, target_points(W, H), 5)
    calibration = {i for per_dot in selected for i in per_dot}
    test = [i for i in range(len(targets)) if i not in calibration]
    days = [person.days[row] for row in prepared.valid_rows]
    by_day = {
        day: float(
            np.mean([e for i, e in zip(test, run.pixel_errors, strict=True) if days[i] == day])
        )
        for day in ("day01", "day02", "day03")
    }

    within = result["within_day"]
    assert [d["day"] for d in within["per_day"]] == ["day01", "day02"]
    for entry in within["per_day"]:
        assert entry["cross_day_mean_px"] == pytest.approx(by_day[entry["day"]], rel=1e-9)
    on_used = (by_day["day01"] + by_day["day02"]) / 2
    assert within["cross_day_mean_px_on_used_days"] == pytest.approx(on_used, rel=1e-9)
    assert within["within_over_cross_mean_px"] == pytest.approx(
        within["mean_px"] / on_used, rel=1e-9
    )
    # The all-frames figure stays, and here the skipped day's frames dominate it.
    assert within["cross_day_mean_px"] == result["cross_day"]["mean_px"]
    assert within["cross_day_mean_px"] > 2 * on_used


def test_days_with_too_few_valid_images_are_skipped_and_listed():
    spec = Spec(days=(150, 30, 70), no_face=frozenset(range(150, 155)))

    within = _one(spec)["within_day"]
    assert within["days_total"] == 3
    assert [d["day"] for d in within["per_day"]] == ["day01", "day03"]
    assert within["skipped_days"] == [{"day": "day02", "n_valid": 25}]
    assert within["days_used"] == 2 and within["days_skipped"] == 1

    assert [d["day"] for d in _one(spec, min_day_images=100)["within_day"]["per_day"]] == ["day01"]
    # Below 5 per dot x 9 dots + 1 test image, a day cannot be run whatever the minimum.
    low = _one(spec, min_day_images=10)["within_day"]
    assert [d["day"] for d in low["per_day"]] == ["day01", "day03"]
    assert [d["n_test"] for d in _one(spec, k_per_dot=2)["within_day"]["per_day"]] == [
        150 - 18,
        70 - 18,
    ]


def test_within_day_is_skipped_with_the_reason_when_no_image_has_a_day():
    without = diagnostics.diagnose([_person(Spec(), with_days=False)])["persons"]["p00"]
    assert "day" in without["within_day"]["skipped"]
    assert without["cross_day"]["calibration_days"] is None

    unfoldered = _person(Spec())
    unfoldered.days = [None] * len(unfoldered.days)
    result = diagnostics.diagnose([unfoldered])["persons"]["p00"]
    assert "day" in result["within_day"]["skipped"]


def test_a_person_too_small_for_the_protocol_keeps_the_in_sample_sections():
    person = _one(Spec(days=(30,)))

    # The protocol's own reason.
    assert "30 valid frames; 5 per dot x 9 dots" in person["cross_day"]["skipped"]
    assert person["within_day"]["days_used"] == 0
    assert person["within_day"]["mean_px"] is None
    assert person["within_day"]["cross_day_mean_px"] is None
    assert person["within_day"]["cross_day_mean_px_on_used_days"] is None
    assert person["within_day"]["within_over_cross_mean_px"] is None
    assert "skipped" not in person["in_sample_ceiling"]


def test_the_aggregate_compares_within_and_cross_day_on_the_same_persons():
    result = _diagnose(
        Spec(noise=0.02, day_bias=(0.1, -0.1, 0.0)),
        Spec("p01", seed=1, noise=0.02),
        Spec("p02", seed=2, noise=0.02, days=(40, 40)),  # cross-day only: no day has 60
    )

    aggregate = result["aggregate"]
    assert aggregate["cross_day"]["mean_px"]["persons"] == 3
    assert aggregate["within_day"]["mean_px"]["persons"] == 2
    assert aggregate["within_day"]["cross_day_mean_px"]["persons"] == 2
    persons = result["persons"]
    assert aggregate["within_day"]["cross_day_mean_px"]["mean"] == pytest.approx(
        (persons["p00"]["cross_day"]["mean_px"] + persons["p01"]["cross_day"]["mean_px"]) / 2
    )
    assert aggregate["within_day"]["days_used_total"] == 6
    assert aggregate["within_day"]["gate2_passed_days_total"] == sum(
        p["within_day"]["gate2_passed_days"] for p in persons.values()
    )
    assert aggregate["cross_day"]["passes_gate2"]["persons"] == 3


# ---------------------------------------------------------------------------
# e. error by head-pose distance from the calibration pose
# ---------------------------------------------------------------------------


def test_bins_are_closed_on_the_right_like_gate_2s_limit():
    deltas = np.array([0.0, 5.0, 5.0001, 10.0, 15.0, 15.5, 20.0, 25.0])
    errors = np.arange(8, dtype=float)

    bins = diagnostics.bin_errors(deltas, errors)

    assert [(b["lo_deg"], b["hi_deg"]) for b in bins] == [
        (0.0, 5.0),
        (5.0, 10.0),
        (10.0, 15.0),
        (15.0, 20.0),
        (20.0, None),
    ]
    assert [b["n"] for b in bins] == [2, 2, 1, 2, 1]
    assert [b["mean_px"] for b in bins] == [0.5, 2.5, 4.0, 5.5, 7.0]
    assert diagnostics.bin_errors(np.array([1.0]), np.array([3.0]))[1]["mean_px"] is None


def test_error_by_head_pose_splits_the_cross_day_test_frames_and_matches_evaluate():
    # The iris offsets shift with the head's yaw, so frames far from the calibration
    # pose are predicted worse.
    spec = Spec(noise=0.02, yaw_range_deg=30.0, pitch_range_deg=4.0, shift_per_yaw_deg=0.01)
    person = _person(spec)

    result = diagnostics.diagnose([person])["persons"]["p00"]

    by_pose, cross = result["error_by_head_pose"], result["cross_day"]
    bins = by_pose["bins"]
    assert sum(b["n"] for b in bins) == cross["n_test"]
    assert all(b["n"] > 0 for b in bins)

    def pooled(chosen: list[dict[str, Any]]) -> float:
        return sum(b["n"] * b["mean_px"] for b in chosen) / sum(b["n"] for b in chosen)

    assert pooled(bins) == pytest.approx(cross["mean_px"])
    assert by_pose["within_limit_mean_px"] == pytest.approx(pooled(bins[:3]))
    assert by_pose["beyond_limit_mean_px"] == pytest.approx(pooled(bins[3:]))
    evaluated = protocol.evaluate([person.data], [models.build_variants()["eyeos"]])
    expected = evaluated["variants"]["eyeos"]["persons"]["p00"]["head_pose_within_limit_share"]
    assert by_pose["within_limit_share"] == pytest.approx(expected)
    assert 0.3 < by_pose["within_limit_share"] < 0.9
    assert bins[-1]["mean_px"] > 2 * bins[0]["mean_px"]


def test_without_head_motion_every_frame_is_within_the_limit():
    by_pose = _one(Spec(noise=0.02))["error_by_head_pose"]

    assert by_pose["bins"][0]["n"] == 450 - 45
    assert by_pose["within_limit_share"] == 1.0
    assert by_pose["beyond_limit_mean_px"] is None


def test_error_by_head_pose_needs_the_cross_day_protocol():
    by_pose = _one(Spec(days=(30,)))["error_by_head_pose"]

    assert "cross-day" in by_pose["skipped"]


def test_the_aggregate_pools_the_head_pose_bins_over_persons():
    result = _diagnose(
        Spec(noise=0.02, yaw_range_deg=25.0, shift_per_yaw_deg=0.01),
        Spec("p01", seed=1, noise=0.02, yaw_range_deg=8.0),
    )

    p00, p01 = (result["persons"][p]["error_by_head_pose"]["bins"] for p in ("p00", "p01"))
    pooled = result["aggregate"]["error_by_head_pose"]["bins"]
    for i, bin_ in enumerate(pooled):
        parts = [b[i] for b in (p00, p01) if b[i]["n"] > 0]
        assert bin_["n"] == p00[i]["n"] + p01[i]["n"]
        assert bin_["persons"] == len(parts)
        if parts:
            assert bin_["pooled_mean_px"] == pytest.approx(
                sum(b["n"] * b["mean_px"] for b in parts) / bin_["n"]
            )
            assert bin_["mean_of_person_means_px"] == pytest.approx(
                sum(b["mean_px"] for b in parts) / len(parts)
            )
        else:
            assert bin_["pooled_mean_px"] is None
    # p01's yaw stays within +-8 deg of its calibration mean: nothing beyond 15.
    assert pooled[-1]["persons"] == 1


# ---------------------------------------------------------------------------
# f. EyeOS's head pose against the dataset's
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "angles",
    [(0, 0, 0), (20, 0, 0), (0, -15, 0), (0, 0, 10), (30, -20, 15), (-45, 25, -35), (5, 60, 0)],
)
def test_dataset_head_angles_follow_the_documented_convention(angles):
    rotations = np.array([_rotation(*angles)])

    got = diagnostics.dataset_head_angles_deg(rotations)

    assert got.shape == (1, 3)
    assert got[0] == pytest.approx(np.array(angles, dtype=float), abs=1e-9)


@pytest.mark.parametrize("axis, column", [(1, 0), (0, 1), (2, 2)])
def test_a_rodrigues_vector_about_one_camera_axis_is_that_one_angle(axis, column):
    for theta_deg in (-25.0, 10.0, 40.0):
        rvec = np.zeros(3)
        rvec[axis] = math.radians(theta_deg)

        got = diagnostics.dataset_head_angles_deg(np.array([mpii.rodrigues(rvec)]))[0]

        expected = np.zeros(3)
        expected[column] = theta_deg  # y -> yaw, x -> pitch, z -> roll
        assert got == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("name", ["none", "x180", "y180", "z180"])
def test_a_head_frame_labelled_the_other_way_round_is_recognised_and_undone(name):
    rng = random.Random(4)
    true = np.array(
        [
            _rotation(rng.uniform(-30, 30), rng.uniform(-20, 20), rng.uniform(-10, 10))
            for _ in range(50)
        ]
    )

    found, relabelled = diagnostics.relabel_head_axes(true @ _AXES[name])

    assert found == name
    assert relabelled == pytest.approx(true, abs=1e-12)


def test_head_pose_agreement_measures_the_scale_of_eyeos_degrees():
    agreement = _one(Spec(yaw_range_deg=15.0, pitch_range_deg=10.0, eyeos_scale=(1.5, 0.4)))[
        "head_pose_agreement"
    ]

    assert agreement["dataset_axes_relabelled"] == "none"
    assert agreement["yaw"]["slope"] == pytest.approx(1.5, abs=1e-9)
    assert agreement["yaw"]["intercept"] == pytest.approx(0.0, abs=1e-9)
    assert agreement["yaw"]["r"] == pytest.approx(1.0, abs=1e-9)
    assert agreement["pitch"]["slope"] == pytest.approx(0.4, abs=1e-9)
    assert agreement["pitch"]["r"] == pytest.approx(1.0, abs=1e-9)
    # Nobody rolled: no slope can be fitted.
    assert agreement["roll"] == {"slope": None, "intercept": None, "r": None}
    assert abs(agreement["dataset_pose_median_deg"]["yaw"]) < 3.0
    matrix = agreement["correlation_matrix"]
    assert matrix["eyeos_yaw"]["dataset_yaw"] == pytest.approx(1.0)
    assert abs(matrix["eyeos_yaw"]["dataset_pitch"]) < 0.2
    assert abs(matrix["eyeos_pitch"]["dataset_yaw"]) < 0.2
    assert matrix["eyeos_yaw"]["dataset_roll"] is None


def test_head_pose_agreement_survives_a_relabelled_head_frame_and_reads_roll():
    plain = _one(Spec(yaw_range_deg=15.0, pitch_range_deg=10.0, eyeos_scale=(1.5, 0.4)))
    flipped = _one(
        Spec(yaw_range_deg=15.0, pitch_range_deg=10.0, eyeos_scale=(1.5, 0.4), axes="y180")
    )

    assert flipped["head_pose_agreement"]["dataset_axes_relabelled"] == "y180"
    assert flipped["head_pose_agreement"]["yaw"] == pytest.approx(
        plain["head_pose_agreement"]["yaw"]
    )

    rolled = _one(Spec(roll_range_deg=10.0, roll_landmarks=True))["head_pose_agreement"]
    assert abs(rolled["roll"]["slope"]) == pytest.approx(1.0, abs=1e-6)
    assert abs(rolled["roll"]["r"]) == pytest.approx(1.0, abs=1e-6)


def test_the_json_states_the_head_pose_convention_and_that_it_is_unchecked():
    notes = _diagnose(Spec())["notes"]

    assert "atan2(R[0,2], R[2,2])" in notes["dataset_head_pose"]
    assert "not been checked against the real files" in notes["dataset_head_pose"]
    assert "not a result" in notes["in_sample"]


def test_the_json_says_the_all_days_ceiling_does_not_bound_a_per_sitting_calibration():
    result = _diagnose(Spec())

    for text in (result["sections"]["in_sample_ceiling"], result["notes"]["in_sample"]):
        assert "across all of a person's days" in text
        assert "per_day" in text
    assert "does not bound" in result["sections"]["in_sample_ceiling"]
    assert "validation_error_px" in result["sections"]["in_sample_ceiling"]
    # Both protocols high does not by itself say the features are the limit.
    assert "bottleneck" not in result["sections"]["within_day"]
    assert "in_sample_ceiling.per_day" in result["sections"]["within_day"]


def test_without_the_datasets_head_pose_f_and_g_say_so():
    person = diagnostics.diagnose([_person(Spec(noise=0.05), with_head=False)])["persons"]["p00"]

    assert "head pose" in person["head_pose_agreement"]["skipped"]
    split = person["variance_split"]
    assert split["r2_head_pose"] is None and split["r2_head_pose_and_day"] is None
    assert split["r2_head_pose_beyond_target"] is None
    assert split["r2_day"] is not None and split["r2_day_beyond_target"] is not None


# ---------------------------------------------------------------------------
# g. variance split: head motion, day, or landmark noise
# ---------------------------------------------------------------------------


def test_head_motion_the_iris_offsets_miss_shows_as_a_high_head_pose_share():
    person = _one(Spec(yaw_range_deg=15.0, pitch_range_deg=10.0, shift_per_yaw_deg=0.02))

    split = person["variance_split"]
    assert split["residual_rms_px"] > 20
    # The iris offsets shift up to 0.3 half-widths with yaw: about 0.92 of the leftover.
    assert split["r2_head_pose"] > 0.8
    assert split["r2_day"] < 0.05
    # The yaw is independent of the target, so all of that is beyond the target.
    assert split["r2_head_pose_beyond_target"] > 0.8
    assert split["r2_day_beyond_target"] < 0.05
    # And EyeOS's own head pose would have absorbed it.
    assert person["in_sample_ceiling"]["with_head_pose"]["mean_px"] < 0.75


def test_pure_landmark_noise_shows_as_a_low_head_pose_and_day_share():
    split = _one(Spec(days=(200, 200, 200), noise=0.05, yaw_range_deg=15.0, pitch_range_deg=10.0))[
        "variance_split"
    ]

    assert split["residual_rms_px"] > 20
    # 6 regressors (one of them constant) on 600 frames: about 0.01 from chance alone.
    assert split["r2_head_pose"] < 0.05
    assert split["r2_day"] < 0.05
    assert split["r2_head_pose_beyond_target"] < 0.05
    assert split["r2_day_beyond_target"] < 0.05


def test_a_head_that_turns_toward_the_target_is_not_read_as_head_motion():
    # Pure landmark noise: the iris offsets do not depend on the head at all. But the head
    # turns toward where the eyes look, so its yaw stands in for the target's x, and a
    # noisy fit (shrunk toward the centre) leaves a residual that grows with the target.
    split = _one(Spec(days=(200, 200, 200), noise=0.6, yaw_range_deg=3.0, yaw_per_screen_deg=30.0))[
        "variance_split"
    ]

    # The control: the target itself explains a large share of the residual ...
    assert split["r2_target"] > 0.25
    # ... so the raw head-pose share is far above chance (about 6/600) ...
    assert split["r2_head_pose"] > 0.1
    # ... but beyond the target, the head pose explains nothing.
    assert split["r2_head_pose_beyond_target"] < 0.02
    assert split["r2_day_beyond_target"] < 0.02


def test_days_that_differ_only_in_where_the_targets_were_are_not_read_as_drift():
    # Pure landmark noise, no per-day offset; but one day's targets were all on the
    # screen's left half and the other's on its right, so the day stands in for the target.
    person = _person(Spec(days=(300, 300), noise=0.6))
    person.days = ["day01" if x < W / 2 else "day02" for x, _ in person.data.targets_px]

    split = diagnostics.diagnose([person])["persons"]["p00"]["variance_split"]

    assert split["r2_day"] > 0.1
    assert split["r2_day_beyond_target"] < 0.02


def test_a_per_day_bias_shows_as_a_high_day_share():
    split = _one(Spec(noise=0.005, day_bias=(0.15, -0.15, 0.0), yaw_range_deg=10.0))[
        "variance_split"
    ]

    assert split["r2_day"] > 0.8
    assert split["r2_head_pose"] < 0.05
    assert split["r2_head_pose_and_day"] >= split["r2_day"] - 1e-12
    assert split["r2_day_beyond_target"] > 0.8
    assert split["r2_head_pose_beyond_target"] < 0.05


def test_head_pose_and_day_together_explain_more_than_either_alone():
    split = _one(
        Spec(noise=0.005, yaw_range_deg=15.0, shift_per_yaw_deg=0.01, day_bias=(0.1, -0.1, 0.0))
    )["variance_split"]

    assert split["r2_head_pose"] > 0.2 and split["r2_day"] > 0.2
    assert split["r2_head_pose_and_day"] > split["r2_head_pose"] + 0.2
    assert split["r2_head_pose_and_day"] > split["r2_day"] + 0.2
    assert split["r2_head_pose_and_day"] > 0.9


def test_the_head_pose_share_includes_the_heads_position():
    # No head rotation at all; the iris offsets shift with the head's sideways position.
    split = _one(Spec(shift_per_mm=0.005))["variance_split"]

    assert split["residual_rms_px"] > 20
    assert split["r2_head_pose"] > 0.9
    assert split["r2_head_pose_beyond_target"] > 0.9


def test_nothing_is_left_to_split_when_the_fit_is_exact():
    split = _one(Spec(yaw_range_deg=10.0))["variance_split"]

    assert split["residual_rms_px"] < 0.5
    assert split["r2_head_pose"] is None and split["r2_day"] is None
    assert split["r2_target"] is None
    assert split["r2_head_pose_beyond_target"] is None
    assert split["r2_day_beyond_target"] is None


def test_without_days_the_day_share_is_null():
    person = diagnostics.diagnose([_person(Spec(noise=0.05, yaw_range_deg=10.0), with_days=False)])
    split = person["persons"]["p00"]["variance_split"]

    assert split["r2_day"] is None and split["r2_head_pose_and_day"] is None
    assert split["r2_day_beyond_target"] is None
    assert split["r2_head_pose"] is not None
    assert split["r2_head_pose_beyond_target"] is not None
    assert split["r2_target"] is not None


# ---------------------------------------------------------------------------
# the JSON: names, descriptions, aggregates, determinism
# ---------------------------------------------------------------------------


def _metric_names(person: dict[str, Any]) -> list[str]:
    names = []
    for key, value in person.items():
        if isinstance(value, dict) and key in diagnostics.SECTIONS:
            names += [f"{key}.{m}" if m != "skipped" else "skipped" for m in value]
        else:
            names.append(key)
    return names


def test_every_metric_has_a_name_and_a_one_line_description():
    result = _diagnose(Spec(noise=0.05, yaw_range_deg=10.0), Spec("p01", seed=1))

    described = result["metrics"]
    for person in result["persons"].values():
        for name in _metric_names(person):
            assert name in described, name
    for name, text in described.items():
        assert text and "\n" not in text, name
    for section in diagnostics.SECTIONS:
        assert result["sections"][section], section
        # Aggregate-only metrics (totals, pooled values) are described on their own.
        per_person = {k for p in result["persons"].values() for k in p.get(section, {})}
        for key in result["aggregate"][section]:
            if key not in per_person:
                assert f"aggregate.{section}.{key}" in described, (section, key)


def test_aggregate_summarises_each_metric_across_persons():
    result = _diagnose(Spec(noise=0.05), Spec("p01", seed=1, noise=0.1))

    p00, p01 = (result["persons"][p]["in_sample_ceiling"]["both_eyes"] for p in ("p00", "p01"))
    across = result["aggregate"]["in_sample_ceiling"]["both_eyes"]["mean_px"]
    assert across["persons"] == 2
    assert across["mean"] == pytest.approx((p00["mean_px"] + p01["mean_px"]) / 2)
    assert across["min"] == pytest.approx(min(p00["mean_px"], p01["mean_px"]))
    assert across["max"] == pytest.approx(max(p00["mean_px"], p01["mean_px"]))
    signs = result["aggregate"]["feature_sanity"]["target_x_signs_match"]
    assert signs == {"true": 2, "persons": 2}
    assert result["aggregate"]["persons"] == 2


def test_aggregate_counts_a_text_metric_per_value():
    result = _diagnose(
        Spec(yaw_range_deg=10.0),
        Spec("p01", seed=1, yaw_range_deg=10.0, axes="y180"),
        Spec("p02", seed=2, yaw_range_deg=10.0, axes="y180"),
    )

    relabelled = result["aggregate"]["head_pose_agreement"]["dataset_axes_relabelled"]
    assert relabelled == {"counts": {"none": 1, "y180": 2}, "persons": 3}


def test_diagnose_is_deterministic_and_strict_json():
    first = _diagnose(Spec(noise=0.05, yaw_range_deg=10.0), Spec("p01", seed=1))
    second = _diagnose(Spec(noise=0.05, yaw_range_deg=10.0), Spec("p01", seed=1))

    text = json.dumps(first, allow_nan=False, sort_keys=True)
    assert text == json.dumps(second, allow_nan=False, sort_keys=True)


def _nudged(face: FaceLandmarks | None) -> FaceLandmarks | None:
    """The same face with every iris coordinate moved by one unit in the last place."""
    if face is None:
        return None
    return dataclasses.replace(
        face,
        left_iris_center=Point3D(
            math.nextafter(face.left_iris_center.x, 1.0), face.left_iris_center.y
        ),
        right_iris_center=Point3D(
            face.right_iris_center.x, math.nextafter(face.right_iris_center.y, 1.0)
        ),
    )


def test_last_bit_noise_in_the_inputs_does_not_change_the_json():
    # numpy's vectorised arctan2 and BLAS can return a result one unit in the last place
    # apart for the same input at a different memory alignment, so the same cache could
    # give a different file. The JSON is rounded so that this is very unlikely to show.
    spec = Spec(noise=0.05, yaw_range_deg=10.0)
    plain = _person(spec)
    nudged = _person(spec)
    nudged.data.landmarks = [_nudged(lm) for lm in nudged.data.landmarks]

    first = diagnostics.diagnose([plain])
    assert json.dumps(first, sort_keys=True) == json.dumps(
        diagnostics.diagnose([nudged]), sort_keys=True
    )
    assert "10 significant digits" in first["number_format"]


def test_the_json_records_the_libraries_and_does_not_promise_an_identical_file():
    result = _diagnose(Spec())

    environment = result["environment"]
    assert environment["python"] == platform.python_version()
    assert environment["numpy"] == np.__version__
    if importlib.util.find_spec("scipy") is None:
        assert environment["scipy"] is None
    else:
        import scipy

        assert environment["scipy"] == scipy.__version__
    assert "extract" in result["notes"]["environment"]
    # Rounding makes a changed digit unlikely, not impossible.
    assert "very unlikely" in result["number_format"]
    assert "cannot" not in result["number_format"]


def test_json_numbers_are_rounded_to_10_significant_digits_and_tiny_ones_to_0():
    value = 23.331758796333283
    clean = diagnostics.clean_for_json

    assert clean(value) == clean(math.nextafter(value, 100.0)) == 23.3317588
    assert clean(12345.678901234) == 12345.6789
    assert clean(np.float64(0.1234567890123)) == 0.123456789
    assert clean(1e-8) == 1e-8
    # Rounding noise around 0 (|x| below 1e-9) is written as 0, never as -0.0.
    for tiny in (-5.377642775528102e-16, 3e-16, -0.0, -9.99e-10):
        assert json.dumps(clean(tiny)) == "0.0", tiny
    assert clean(float("nan")) is None and clean(float("inf")) is None
    assert clean({"a": [np.int64(3), True, np.bool_(False), "x", None]}) == {
        "a": [3, True, False, "x", None]
    }
    assert type(clean(np.int64(3))) is int and type(clean(np.float64(2.0))) is float


def test_a_person_with_too_few_valid_frames_is_skipped_section_by_section():
    tiny = _one(Spec(days=(3,)))
    assert tiny["n_valid"] == 3
    for section in ("feature_sanity", "in_sample_ceiling", "constant_baselines"):
        assert "skipped" in tiny[section], section

    result = _diagnose(Spec(days=(3,)), Spec("p01", seed=1))
    across = result["aggregate"]["in_sample_ceiling"]["both_eyes"]["mean_px"]
    assert across["persons"] == 1


def test_a_person_without_any_face_is_reported_not_crashed():
    person = _one(Spec(days=(5,), no_face=frozenset(range(5))))

    assert person["n_images"] == 5 and person["n_valid"] == 0
    assert "skipped" in person["feature_sanity"]


def test_the_summary_has_a_line_per_person_and_a_mean_line():
    result = _diagnose(Spec(noise=0.05), Spec("p01", seed=1), Spec("p02", days=(3,)))

    lines = diagnostics.summary_table(result).splitlines()
    assert [line.split()[0] for line in lines] == ["person", "p00", "p01", "p02", "mean"]
    cross = result["aggregate"]["cross_day"]["mean_px"]["mean"]
    assert f"{cross:.1f}" in lines[-1]
    assert lines[3].split()[1:3] == ["3", "-"]  # a skipped person shows dashes


# ---------------------------------------------------------------------------
# the diagnose command, on a fake MPIIFaceGaze written to disk
# ---------------------------------------------------------------------------


@pytest.fixture
def scipy_io() -> Any:
    return pytest.importorskip("scipy.io")


def _rvec(rotation: np.ndarray) -> np.ndarray:
    """The Rodrigues vector of a rotation by less than a half turn (the inverse of
    mpiifacegaze.rodrigues), written out here so the loader is not checked against itself."""
    theta = math.acos(max(-1.0, min(1.0, (float(np.trace(rotation)) - 1.0) / 2.0)))
    if theta < 1e-12:
        return np.zeros(3)
    r = rotation
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    rvec: np.ndarray = axis / (2.0 * math.sin(theta)) * theta
    return rvec


def _annotation(frame: Frame, image: str, screen: FakePerson) -> str:
    """One pXX.txt line: 28 fields, with this frame's own head pose (fields 16-21)."""
    gaze_target = screen.gaze_target_mm(*frame.target)
    landmarks = [500.0 + 10 * i for i in range(12)]  # six (x, y) points, unused by EyeOS
    numbers = [
        *frame.target,
        *landmarks,
        *_rvec(frame.rotation),
        *frame.tvec,
        *screen.face_center,
        *gaze_target,
    ]
    return " ".join([image, *(repr(float(v)) for v in numbers), "left"])


def _write_mpii(root: Path, spec: Spec, flat: bool = False) -> list[Frame]:
    """A fake MPIIFaceGaze participant on disk: each "image" is a JSON file naming the
    face _face should draw (null = no face), in its dayNN folder (or in none if ``flat``)."""
    frames = _frames(spec)
    folder = root / spec.name
    screen = FakePerson(spec.name)  # a 1280x800 screen, like W x H
    lines = []
    for frame in frames:
        image = frame.image.split("/")[-1] if flat else frame.image
        (folder / image).parent.mkdir(parents=True, exist_ok=True)
        (folder / image).write_text(json.dumps({"face": frame.face}))
        lines.append(_annotation(frame, image, screen))
    (folder / f"{spec.name}.txt").write_text("\n".join(lines) + "\n")
    write_calibration(folder / "Calibration", screen)
    return frames


class FaceDetector:
    """Stands in for MediaPipe: draws the face a fake image names."""

    def __call__(self, image_path: Path) -> FaceLandmarks | None:
        face = json.loads(Path(image_path).read_text())["face"]
        return None if face is None else _face(**face)


def write_and_extract(root: Path, cache: Path, *specs: Spec, flat: bool = False) -> None:
    """Write the fake participants under ``root`` and extract them into ``cache``."""
    for spec in specs:
        _write_mpii(root, spec, flat=flat)
    mpii.extract(root, cache, detector_factory=FaceDetector)


def _on_disk(tmp_path: Path, *specs: Spec, flat: bool = False) -> tuple[Path, Path]:
    root, cache = tmp_path / "MPIIFaceGaze", tmp_path / "cache"
    write_and_extract(root, cache, *specs, flat=flat)
    return root, cache


def test_the_fake_annotation_holds_each_frames_own_head_pose():
    spec = Spec(days=(3,), yaw_range_deg=20.0, pitch_range_deg=10.0, roll_range_deg=5.0)
    frame = _frames(spec)[2]

    line = mpii.parse_annotation_line(_annotation(frame, frame.image, FakePerson("p00")), "here")

    assert mpii.rodrigues(np.array(line.head_rvec)) == pytest.approx(frame.rotation, abs=1e-12)
    assert line.head_tvec == pytest.approx(frame.tvec)
    assert line.target_px == frame.target


def test_load_diagnostic_person_reads_days_from_the_cache_and_head_pose_from_pxx_txt(
    tmp_path, scipy_io
):
    spec = Spec(days=(20, 30), yaw_range_deg=20.0, pitch_range_deg=10.0, no_face=frozenset({4}))
    root, cache = _on_disk(tmp_path, spec)
    frames = _frames(spec)

    person = mpii.load_diagnostic_person(root, cache, "p00")

    data, _ = mpii.load_person(root, cache, "p00")
    assert person.name == "p00"
    assert np.array_equal(person.data.targets_px, data.targets_px)
    assert person.data.landmarks == data.landmarks
    assert person.data.landmarks[4] is None
    assert person.data.to_camera_mm is not None
    assert person.days == ["day01"] * 20 + ["day02"] * 30
    assert person.head_rotation is not None and person.head_translation_mm is not None
    assert person.head_rotation == pytest.approx(np.array([f.rotation for f in frames]), abs=1e-12)
    assert person.head_translation_mm == pytest.approx(np.array([f.tvec for f in frames]))
    # The cache itself has no head-pose columns: they come from pXX.txt.
    assert not any("rvec" in c or "tvec" in c or c.startswith("head") for c in mpii.CACHE_COLUMNS)


def test_load_diagnostic_person_without_day_folders_has_no_days(tmp_path, scipy_io):
    root, cache = _on_disk(tmp_path, Spec(days=(80,), yaw_range_deg=10.0), flat=True)

    person = mpii.load_diagnostic_person(root, cache, "p00")

    assert person.days == [None] * 80
    result = diagnostics.diagnose([person])["persons"]["p00"]
    assert "day" in result["within_day"]["skipped"]
    assert "skipped" not in result["head_pose_agreement"]


def test_load_diagnostic_person_fails_loudly_when_pxx_txt_no_longer_matches_the_cache(
    tmp_path, scipy_io
):
    root, cache = _on_disk(tmp_path, Spec(days=(20,)))
    annotations = root / "p00" / "p00.txt"
    lines = annotations.read_text().splitlines()

    annotations.write_text("\n".join(lines[:5] + lines[6:]) + "\n")  # one image dropped
    with pytest.raises(mpii.DatasetFormatError, match=r"day01/0005\.jpg"):
        mpii.load_diagnostic_person(root, cache, "p00")

    moved = lines[5].split()
    moved[1] = repr(float(moved[1]) + 3.0)  # the same image, another target
    annotations.write_text("\n".join(lines[:5] + [" ".join(moved)] + lines[6:]) + "\n")
    with pytest.raises(mpii.DatasetFormatError, match=r"day01/0005\.jpg the target"):
        mpii.load_diagnostic_person(root, cache, "p00")


def test_load_diagnostic_person_matches_pxx_txt_by_image_when_the_order_differs(tmp_path, scipy_io):
    spec = Spec(days=(10, 10), yaw_range_deg=20.0)
    root, cache = _on_disk(tmp_path, spec)
    annotations = root / "p00" / "p00.txt"
    annotations.write_text("\n".join(reversed(annotations.read_text().splitlines())) + "\n")

    person = mpii.load_diagnostic_person(root, cache, "p00")

    expected = np.array([f.rotation for f in _frames(spec)])
    assert person.head_rotation == pytest.approx(expected, abs=1e-12)


def test_diagnose_cli_writes_a_deterministic_json_and_a_summary(tmp_path, scipy_io, capsys):
    root, cache = _on_disk(
        tmp_path,
        Spec(noise=0.02, yaw_range_deg=15.0, pitch_range_deg=8.0, eyeos_scale=(1.5, 0.4)),
        Spec("p01", seed=1, noise=0.02, day_bias=(0.1, -0.1, 0.0), yaw_range_deg=5.0),
    )
    first, second = tmp_path / "out" / "diag.json", tmp_path / "out" / "again.json"
    args = ["diagnose", "--root", str(root), "--cache", str(cache)]

    assert mpii.main([*args, "--out", str(first)]) == 0
    out = capsys.readouterr().out
    assert mpii.main([*args, "--out", str(second)]) == 0

    assert first.read_bytes() == second.read_bytes()
    result = json.loads(first.read_text())
    assert result["dataset"] == "MPIIFaceGaze" and result["license"] == "CC BY-NC-SA 4.0"
    assert "generated" not in result  # no timestamp: the same cache gives the same file
    assert "git_commit" in result
    config = result["config"]
    assert config["persons"] == ["p00", "p01"]
    assert config["root"] == str(root) and config["cache"] == str(cache)
    assert config["k_per_dot"] == 5 and config["min_day_images"] == 60
    assert "pXX.txt" in result["notes"]["loading"]
    assert list(result["persons"]) == ["p00", "p01"]
    p00, p01 = result["persons"]["p00"], result["persons"]["p01"]
    assert p00["head_pose_agreement"]["yaw"]["slope"] == pytest.approx(1.5, rel=1e-6)
    assert p00["head_pose_agreement"]["pitch"]["slope"] == pytest.approx(0.4, rel=1e-6)
    assert p00["within_day"]["days_used"] == 3
    assert p01["within_day"]["mean_px"] < p01["cross_day"]["mean_px"]
    assert p01["variance_split"]["r2_day"] > 0.5
    # cross_day is evaluate's eyeos result on the same cache.
    evaluated = mpii.evaluate_dataset(root, cache, variants=["eyeos"])["variants"]["eyeos"]
    for name in ("p00", "p01"):
        expected = evaluated["persons"][name]
        cross = result["persons"][name]["cross_day"]
        assert cross["mean_px"] == pytest.approx(expected["pixel_error_px"]["mean"], rel=1e-9)
        assert cross["mean_deg"] == pytest.approx(expected["angular_error_deg"]["mean"], rel=1e-9)
    lines = out.splitlines()
    assert [line.split()[0] for line in lines[:4]] == ["person", "p00", "p01", "mean"]
    assert f"wrote {first}" in out


def test_diagnose_cli_passes_persons_k_and_the_day_minimum_through(tmp_path, scipy_io):
    root, cache = _on_disk(tmp_path, Spec(days=(100, 50)), Spec("p01", seed=1, days=(100, 50, 100)))
    out = tmp_path / "diag.json"

    code = mpii.main(
        ["diagnose", "--root", str(root), "--cache", str(cache), "--out", str(out)]
        + ["--persons", "p01", "--k-per-dot", "3", "--min-day-images", "80"]
    )

    assert code == 0
    result = json.loads(out.read_text())
    assert result["config"]["persons"] == ["p01"]
    assert result["config"]["k_per_dot"] == 3 and result["config"]["min_day_images"] == 80
    within = result["persons"]["p01"]["within_day"]
    assert [d["day"] for d in within["per_day"]] == ["day01", "day03"]
    assert within["skipped_days"] == [{"day": "day02", "n_valid": 50}]
    assert result["persons"]["p01"]["cross_day"]["n_calibration"] == 27


def test_diagnose_cli_rejects_an_unknown_person(tmp_path, scipy_io, capsys):
    root, cache = _on_disk(tmp_path, Spec(days=(10,)))

    with pytest.raises(SystemExit) as excinfo:
        mpii.main(
            ["diagnose", "--root", str(root), "--cache", str(cache)]
            + ["--out", str(tmp_path / "d.json"), "--persons", "p00,p07"]
        )

    assert excinfo.value.code == 2
    assert "p07" in capsys.readouterr().err


def test_diagnose_cli_reports_an_empty_cache(tmp_path, capsys):
    out = tmp_path / "d.json"

    code = mpii.main(
        ["diagnose", "--root", str(tmp_path), "--cache", str(tmp_path / "empty")]
        + ["--out", str(out)]
    )

    assert code == 1
    assert "extract" in capsys.readouterr().err
    assert not out.exists()


def test_diagnose_cli_reports_a_missing_annotation_file(tmp_path, scipy_io, capsys):
    root, cache = _on_disk(tmp_path, Spec(days=(10,)))
    (root / "p00" / "p00.txt").unlink()
    out = tmp_path / "d.json"

    code = mpii.main(["diagnose", "--root", str(root), "--cache", str(cache), "--out", str(out)])

    assert code == 1
    assert "p00.txt" in capsys.readouterr().err
    assert not out.exists()


# ---------------------------------------------------------------------------
# docs/PUBLIC_DATASETS.md
# ---------------------------------------------------------------------------

_DOC = Path(__file__).resolve().parents[1] / "docs" / "PUBLIC_DATASETS.md"


def _diagnostics_subsection() -> str:
    text = _DOC.read_text()
    heading = "### Diagnostics: where the error comes from"
    start = text.index(heading) + len(heading)
    after = re.search(r"^##{1,2} ", text[start:], flags=re.MULTILINE)  # the next heading
    return text[start : start + after.start()] if after else text[start:]


def test_the_doc_explains_every_diagnostic_section_and_says_the_real_run_is_pending():
    section = _diagnostics_subsection()

    assert "not been run on the real data" in section
    for name in diagnostics.SECTIONS:
        assert f"`{name}`" in section, name


def test_the_docs_diagnose_command_parses_with_the_real_command_line():
    text = _diagnostics_subsection().replace("\\\n", " ")
    commands = [
        line.split()
        for line in text.splitlines()
        if line.strip().startswith("python -m eval.public.mpiifacegaze diagnose ")
    ]

    assert len(commands) == 1
    args = mpii.build_parser().parse_args(commands[0][3:])
    assert args.root == Path("datasets/MPIIFaceGaze")
    assert args.cache == Path("datasets/cache/mpiifacegaze")
    assert args.out == Path("eval/results/mpiifacegaze-diagnostics.json")
