# --- Audio file existence tests (append to tests/test_alerts.py) ---

import base64
import types
import app_v2 as m


class DummyContainer:
    """Fake Streamlit container used for audio placeholder."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def empty(self):
        pass


def test_load_audio_b64_returns_none_when_missing(tmp_path, monkeypatch):
    # point ALARM_PATH to a non-existent file
    missing = tmp_path / "nope.mp3"
    monkeypatch.setattr(m, "ALARM_PATH", str(missing))

    assert m._load_audio_b64(str(missing)) is None


def test_triggerAudio_warns_when_file_missing(monkeypatch, tmp_path):
    # point ALARM_PATH to a non-existent file
    missing = tmp_path / "nope.mp3"
    monkeypatch.setattr(m, "ALARM_PATH", str(missing))

    # stub streamlit warning
    warnings = {"n": 0, "msg": ""}

    class FakeSt:
        def warning(self, msg):
            warnings["n"] += 1
            warnings["msg"] = msg

    monkeypatch.setattr(m, "st", FakeSt())

    # ensure components.html is NOT called
    class FakeComponents:
        def html(self, *args, **kwargs):
            raise AssertionError("components.html should not be called when file is missing")

    monkeypatch.setattr(m, "components", FakeComponents())

    m.triggerAudio(DummyContainer(), loop=False)

    assert warnings["n"] == 1
    assert "Alarm audio file not found" in warnings["msg"]


def test_load_audio_b64_reads_file(tmp_path):
    # create a tiny fake mp3 file (any bytes are fine for base64 test)
    p = tmp_path / "beep.mp3"
    data = b"\x00\x01fake-mp3-bytes\x02\x03"
    p.write_bytes(data)

    b64 = m._load_audio_b64(str(p))
    assert isinstance(b64, str)
    assert base64.b64decode(b64.encode("utf-8")) == data


def test_triggerAudio_calls_components_html_when_file_exists(monkeypatch, tmp_path):
    # create fake mp3
    p = tmp_path / "beep.mp3"
    p.write_bytes(b"fake-mp3-bytes")
    monkeypatch.setattr(m, "ALARM_PATH", str(p))

    # stub st.warning (should NOT be called)
    class FakeSt:
        def warning(self, msg):
            raise AssertionError("st.warning should not be called when file exists")

    monkeypatch.setattr(m, "st", FakeSt())

    # capture components.html call
    calls = {"n": 0, "html": ""}

    class FakeComponents:
        def html(self, html, **kwargs):
            calls["n"] += 1
            calls["html"] = html

    monkeypatch.setattr(m, "components", FakeComponents())

    m.triggerAudio(DummyContainer(), loop=False)

    assert calls["n"] == 1
    assert "<audio" in calls["html"]  # basic sanity check
