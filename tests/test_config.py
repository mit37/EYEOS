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
