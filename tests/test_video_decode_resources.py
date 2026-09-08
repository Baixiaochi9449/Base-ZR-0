from fractions import Fraction
from types import SimpleNamespace

import av
import numpy as np
import pytest
import torch

from lerobot.common.datasets import video_utils as video


class FakeDecoder:
    def __init__(self, events, codec, failure):
        self.events, self.codec, self.failure = events, codec, failure
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.codec.is_open = True
        self.events.append(("decode", self.codec.thread_count))
        if self.failure == "decode":
            raise MemoryError("injected codec allocation failure")
        if self.index == 3:
            raise StopIteration
        self.index += 1
        frame = SimpleNamespace(pts=self.index - 1, time_base=Fraction(1, 10))
        def convert():
            if self.failure == "convert":
                raise RuntimeError("conversion failed")
            return SimpleNamespace(to_ndarray=lambda: np.full((2, 3, 3), frame.pts, dtype=np.uint8))
        frame.to_rgb = convert
        return frame

    def close(self):
        self.events.append("generator.close")
        if self.failure == "generator_close":
            raise RuntimeError("generator close failed")


def install_decoder(monkeypatch, failure=None, public_codec_close=True):
    events = []
    codec = SimpleNamespace(is_open=False, thread_count=0, name="libdav1d", thread_type="SLICE")
    def close_codec():
        events.append("codec.close")
        if failure == "codec_close":
            raise RuntimeError("codec close failed")
        codec.is_open = False
    if public_codec_close:
        codec.close = close_codec
    stream = SimpleNamespace(time_base=Fraction(1, 10), codec_context=codec)
    decoder = FakeDecoder(events, codec, failure)
    expected_threads = [1]
    def seek(offset, **kwargs):
        assert codec.thread_count == expected_threads[0]
        assert not codec.is_open
        events.append(("seek", offset, kwargs["backward"], kwargs["any_frame"]))
        if failure == "seek":
            raise RuntimeError("seek failed")
    def close_container():
        events.append("container.close")
        if failure == "container_close":
            raise RuntimeError("container close failed")
    container = SimpleNamespace(streams=SimpleNamespace(video=[stream]), seek=seek,
                                decode=lambda **kwargs: decoder, close=close_container)
    def open_container(path, **kwargs):
        assert kwargs["options"]["threads"] == str(expected_threads[0])
        events.append("open")
        if failure == "open":
            raise RuntimeError("open failed")
        return container
    monkeypatch.setattr(av, "open", open_container)
    monkeypatch.delenv("LEROBOT_PYAV_THREADS", raising=False)
    return events, expected_threads


@pytest.mark.parametrize("timestamps", [[0.0], [0.2], [0.1, 0.0, 0.1]])
def test_normal_early_and_exhausted_release_and_selection(monkeypatch, timestamps):
    events, _ = install_decoder(monkeypatch)
    result = video.decode_video_frames("fake.mp4", timestamps, .051, "pyav")
    assert events[-3:] == ["generator.close", "codec.close", "container.close"]
    torch.testing.assert_close(result[:, 0, 0, 0], torch.tensor(timestamps) * 10 / 255, rtol=0, atol=0)
    install_decoder(monkeypatch)
    frames, times = video.decode_video_frames_pyav("fake.mp4", [1.0])
    assert len(frames) == 3 and times == [0., .1, .2]


@pytest.mark.parametrize("failure", ["open", "seek", "decode", "convert", "generator_close", "codec_close", "container_close"])
def test_all_failure_paths_propagate_and_close_owned_resources(monkeypatch, failure):
    events, _ = install_decoder(monkeypatch, failure)
    with pytest.raises((RuntimeError, MemoryError)):
        video.decode_video_frames("fake.mp4", [0.], .05, "pyav")
    if failure != "open":
        assert events[-1] == "container.close"
    if failure not in {"open", "seek"}:
        assert "generator.close" in events and "codec.close" in events


def test_missing_public_codec_close_uses_only_container_api(monkeypatch):
    events, _ = install_decoder(monkeypatch, public_codec_close=False)
    video.decode_video_frames("fake.mp4", [0.], .05, "pyav")
    assert events[-2:] == ["generator.close", "container.close"]


@pytest.mark.parametrize("explicit,environment,expected", [(None, None, 1), (None, "2", 2), (3, "2", 3), (0, "2", 0)])
def test_thread_config_reaches_open_and_codec_before_decode(monkeypatch, explicit, environment, expected):
    events, wanted = install_decoder(monkeypatch)
    wanted[0] = expected
    if environment is not None:
        monkeypatch.setenv("LEROBOT_PYAV_THREADS", environment)
    video.decode_video_frames("fake.mp4", [0.], .05, "pyav", num_threads=explicit)
    assert ("decode", expected) in events


@pytest.mark.parametrize("value", [-1, True, 1.5, "bad"])
def test_invalid_threads_rejected_before_open(monkeypatch, value):
    events, _ = install_decoder(monkeypatch)
    with pytest.raises(ValueError):
        video.decode_video_frames("fake.mp4", [0.], .05, "pyav", num_threads=value)
    assert not events


def test_tolerance_failure_happens_after_cleanup(monkeypatch):
    events, _ = install_decoder(monkeypatch)
    with pytest.raises(AssertionError, match="tolerance"):
        video.decode_video_frames("fake.mp4", [.05], .001, "pyav")
    assert events[-3:] == ["generator.close", "codec.close", "container.close"]
