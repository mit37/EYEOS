import pytest

from eyeos.config import load_config


def test_defaults_load_with_no_file():
    cfg = load_config(None)
    assert cfg.screen.width_px == 1920
    assert cfg.gates.dwell_time_s == 0.8
    assert cfg.gates.no_click_zones == ()


def test_example_config_loads():
    cfg = load_config("config.example.yaml")
    assert cfg.input.source == "synthetic"
    assert cfg.output.backend == "dry_run"
    assert len(cfg.gates.no_click_zones) == 1
    zone = cfg.gates.no_click_zones[0]
    assert zone.contains(1890, 20)
    assert not zone.contains(0, 0)


def test_partial_override_keeps_other_defaults(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("screen:\n  width_px: 800\n")
    cfg = load_config(p)
    assert cfg.screen.width_px == 800
    assert cfg.screen.height_px == 1080


def test_confirmation_mode_and_confirm_key_validation(tmp_path):
    import pytest

    p = tmp_path / "c.yaml"
    p.write_text("gates:\n  confirmation_mode: keypress\noutput:\n  confirm_key: space\n")
    cfg = load_config(p)
    assert cfg.gates.confirmation_mode == "keypress"
    assert cfg.output.confirm_key == "space"

    p.write_text("gates:\n  confirmation_mode: wink\n")
    with pytest.raises(ValueError, match="confirmation_mode"):
        load_config(p)

    p.write_text("output:\n  kill_hotkey: esc\n  confirm_key: Escape\n")
    with pytest.raises(ValueError, match="must differ"):
        load_config(p)


def test_calibration_points_must_be_nine(tmp_path):
    import pytest

    p = tmp_path / "c.yaml"
    p.write_text("calibration:\n  points: 8\n")
    with pytest.raises(ValueError, match="calibration.points"):
        load_config(p)


def test_presence_confidence_threshold_must_be_a_probability(tmp_path):
    import pytest

    p = tmp_path / "c.yaml"
    p.write_text("gates:\n  presence_confidence_threshold: 1.5\n")
    with pytest.raises(ValueError, match="presence_confidence_threshold"):
        load_config(p)


@pytest.mark.parametrize(
    "yaml_text, key",
    [
        ("gates:\n  dwell_time: 0.5\n", "gates.dwell_time"),
        ("screen:\n  width: 800\n", "screen.width"),
        ("ouptut:\n  backend: dry_run\n", "config.ouptut"),
    ],
)
def test_unknown_config_keys_get_a_readable_error(tmp_path, yaml_text, key):
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    with pytest.raises(ValueError, match=key.replace(".", r"\.")):
        load_config(p)


def test_a_misspelled_no_click_zone_key_is_named(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("gates:\n  no_click_zones:\n    - {x_min: 0, y_min: 0, x_max: 10, ymax: 10}\n")
    with pytest.raises(ValueError, match=r"gates\.no_click_zones\[0\]\.ymax"):
        load_config(cfg)


@pytest.mark.parametrize(
    "text, message",
    [
        ("output: esc\n", "output must be a mapping"),
        ("gates: 5\n", "gates must be a mapping"),
        ("- a\n- b\n", "config must be a mapping"),
        ("gates:\n  no_click_zones: {x_min: 0}\n", "no_click_zones must be a list"),
        ("gates:\n  no_click_zones:\n    - 7\n", r"no_click_zones\[0\] must be a mapping"),
    ],
)
def test_a_section_that_is_not_a_mapping_is_reported(tmp_path, text, message):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(text)
    with pytest.raises(ValueError, match=message):
        load_config(cfg)
