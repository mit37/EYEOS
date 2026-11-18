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
import json
import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from eval.generate_synthetic import make_landmarks
from eval.public import diagnostics, models, protocol
from eyeos.calibration import Calibration
from eyeos.features import FeatureVector, extract_features
from eyeos.landmarks import FaceLandmarks, Point3D
from tests.fake_mpiifacegaze import FakePerson, offsets_for_target

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
    yaw, pitch and roll are uniform in +-their range, independent of the target."""

    name: str = "p00"
    days: tuple[int, ...] = (150, 150, 150)  # frames per dayNN folder
    seed: int = 0
    noise: float = 0.0  # Gaussian sigma, independently per eye and axis: landmark noise
    yaw_range_deg: float = 0.0
    pitch_range_deg: float = 0.0
    roll_range_deg: float = 0.0
    shift_per_yaw_deg: float = 0.0  # iris x offset added per degree of the dataset's yaw
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
            pitch = rng.uniform(-spec.pitch_range_deg, spec.pitch_range_deg)
            roll = rng.uniform(-spec.roll_range_deg, spec.roll_range_deg)
            noise = [rng.gauss(0.0, spec.noise) for _ in range(4)]
            tvec = (rng.uniform(-30, 30), rng.uniform(-30, 30), rng.uniform(450, 650))
            ox, oy = offsets_for_target(x, y, W, H)
            ox += spec.shift_per_yaw_deg * yaw + bias
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


def test_feature_sanity_reads_the_eyeos_roll_spread():
    sanity = _one(Spec(roll_range_deg=10.0, roll_landmarks=True))["feature_sanity"]

    roll = sanity["eyeos_roll_deg"]
    assert -10.0 <= roll["p5"] < -7.0
    assert 7.0 < roll["p95"] <= 10.0


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


# ---------------------------------------------------------------------------
# d. the standard (cross-day) protocol against a within-day one
# ---------------------------------------------------------------------------


def test_cross_day_reproduces_evaluate_for_the_shipping_calibration():
    spec = Spec(noise=0.05, yaw_range_deg=10.0)
    person = _person(spec, with_3d=True)

    cross = diagnostics.diagnose([person])["persons"]["p00"]["cross_day"]

    evaluated = protocol.evaluate([person.data], [models.build_variants()["eyeos"]])
    expected = evaluated["variants"]["eyeos"]["persons"]["p00"]
    assert cross["mean_px"] == pytest.approx(expected["pixel_error_px"]["mean"], abs=1e-12)
    assert cross["mean_deg"] == pytest.approx(expected["angular_error_deg"]["mean"], abs=1e-12)
    assert cross["validation_error_px"] == pytest.approx(expected["validation_error_px"])
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
        within["mean_px"] / cross["mean_px"]
    )
    assert within["within_over_cross_mean_px"] < 0.25


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


def test_diagnose_is_deterministic_and_strict_json():
    first = _diagnose(Spec(noise=0.05, yaw_range_deg=10.0), Spec("p01", seed=1))
    second = _diagnose(Spec(noise=0.05, yaw_range_deg=10.0), Spec("p01", seed=1))

    text = json.dumps(first, allow_nan=False, sort_keys=True)
    assert text == json.dumps(second, allow_nan=False, sort_keys=True)


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


def test_the_summary_names_every_person():
    result = _diagnose(Spec(noise=0.05), Spec("p01", seed=1))

    text = diagnostics.summary_table(result)
    assert "p00" in text and "p01" in text and "all" in text
