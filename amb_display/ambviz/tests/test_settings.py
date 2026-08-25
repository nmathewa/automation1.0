import pytest

from ambviz.settings import Settings


def test_defaults():
    s = Settings.load()
    assert s.output.pixels == 60
    assert s.audio.source == "mic"
    assert s.effect.name == "spectrum"
    assert s.warnings == []


def test_file_then_env_then_override(tmp_path, monkeypatch):
    cfg = tmp_path / "rig.toml"
    cfg.write_text('[output]\npixels = 30\nhost = "10.0.0.1"\n')
    monkeypatch.setenv("AMBVIZ_OUTPUT_PIXELS", "40")

    assert Settings.load(cfg, env=False).output.pixels == 30      # file
    assert Settings.load(cfg).output.pixels == 40                  # env beats file
    s = Settings.load(cfg, overrides={"output": {"pixels": 50}})   # override beats env
    assert s.output.pixels == 50
    assert s.output.host == "10.0.0.1"                             # untouched keys survive


def test_json_config(tmp_path):
    cfg = tmp_path / "rig.json"
    cfg.write_text('{"output": {"pixels": 24}}')
    assert Settings.load(cfg).output.pixels == 24


def test_unknown_key_is_named():
    with pytest.raises(KeyError, match="output.pixles"):
        Settings.load(overrides={"output": {"pixles": 60}})


def test_unknown_section():
    with pytest.raises(KeyError, match=r"\[nope\]"):
        Settings.load(overrides={"nope": {"a": 1}})


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        Settings.load("does-not-exist.toml")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"dsp": {"max_frequency": 30000}}, "Nyquist"),
        ({"dsp": {"min_frequency": 9000, "max_frequency": 200}}, "must be below"),
        ({"audio": {"fps": 200}, "output": {"pixels": 600}}, "exceeds what"),
        ({"effect": {"name": "strobe"}}, "unknown effect"),
        ({"effect": {"brightness": 4}}, "brightness"),
        ({"output": {"device": "serial"}}, "output.device"),
        ({"audio": {"source": "wav"}}, "wav_path is empty"),
        ({"smoothing": {"red": (0.0, 0.5)}}, "strictly between"),
        ({"audio": {"input_device": 1.5}}, "device index, a name"),
    ],
)
def test_validation_rejects(overrides, message):
    with pytest.raises(ValueError, match=message):
        Settings.load(overrides=overrides)


def test_warnings_do_not_raise():
    s = Settings.load(overrides={"output": {"pixels": 63}})
    assert any("odd" in w for w in s.warnings)

    s = Settings.load(overrides={"output": {"pixels": 200}, "audio": {"fps": 30}})
    assert any("signed char" in w for w in s.warnings)

    s = Settings.load(overrides={"output": {"pixels": 16}, "dsp": {"fft_bins": 24}})
    assert any("cannot be resolved" in w for w in s.warnings)


def test_empty_string_means_unset():
    assert Settings.load(overrides={"audio": {"input_device": ""}}).audio.input_device is None


def test_input_device_accepts_an_index_or_a_name():
    """Indices get reordered between boots; names do not."""
    assert Settings.load(overrides={"audio": {"input_device": 13}}).audio.input_device == 13
    assert Settings.load(overrides={"audio": {"input_device": "pulse"}}).audio.input_device == "pulse"


def test_toml_roundtrip(tmp_path):
    original = Settings.load(overrides={"output": {"pixels": 42}, "effect": {"name": "scroll"}})
    path = tmp_path / "dump.toml"
    path.write_text(original.to_toml())
    assert Settings.load(path, env=False).to_dict() == original.to_dict()


def test_gamma_table_ships_with_the_package():
    assert Settings.load().output.gamma_table_path().exists()
