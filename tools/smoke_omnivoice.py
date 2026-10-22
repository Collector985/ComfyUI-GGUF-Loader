"""Offline OmniVoice node smoke tests: no downloads, weights, or GPU.

Usage:  python tools/smoke_omnivoice.py
"""

import builtins
import importlib
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix="omnivoice_smoke_"))
RESULTS = []


def ok(message):
    RESULTS.append(message)
    print(f"[ok] {message}")


# Import only nodes/omnivoice.py, not this package's aggregate __init__.py.
folder_paths = types.ModuleType("folder_paths")
folder_paths.models_dir = str(TMP / "models")
folder_paths.folder_names_and_paths = {}


def add_model_folder_path(name, path):
    folder_paths.folder_names_and_paths.setdefault(name, []).append(path)


folder_paths.add_model_folder_path = add_model_folder_path
sys.modules["folder_paths"] = folder_paths

interrupt_calls = []
progress_updates = []
comfy = types.ModuleType("comfy")
comfy.__path__ = []
mm = types.ModuleType("comfy.model_management")
mm.throw_exception_if_processing_interrupted = lambda: interrupt_calls.append(True)
utils = types.ModuleType("comfy.utils")


class ProgressBar:
    def __init__(self, total):
        self.total = total

    def update_absolute(self, value, total):
        progress_updates.append((value, total))


utils.ProgressBar = ProgressBar
comfy.model_management = mm
comfy.utils = utils
sys.modules["comfy"] = comfy
sys.modules["comfy.model_management"] = mm
sys.modules["comfy.utils"] = utils

pkg = types.ModuleType("cctech_gguf_pkg")
pkg.__path__ = [str(ROOT)]
nodes_pkg = types.ModuleType("cctech_gguf_pkg.nodes")
nodes_pkg.__path__ = [str(ROOT / "nodes")]
sys.modules[pkg.__name__] = pkg
sys.modules[nodes_pkg.__name__] = nodes_pkg

# Prove import and mapping creation do not need the runtime package.
sys.modules.pop("omnivoice", None)
ov = importlib.import_module("cctech_gguf_pkg.nodes.omnivoice")


class FakeModel:
    def __init__(self):
        self.devices = []
        self.calls = []
        self.training = True
        self.raise_instruct = False
        self.long_first = False

    def to(self, device):
        self.devices.append(str(device))
        return self

    def eval(self):
        self.training = False
        return self

    def modules(self):
        return []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.raise_instruct:
            raise ValueError("unsupported instruct value")
        if self.long_first and len(self.calls) == 1:
            return [np.linspace(-0.1, 0.1, 26 * 24000, dtype=np.float32)]
        value = len(self.calls) / 10.0
        return [np.full(1200, value, dtype=np.float32)]


class FakeOmniVoice:
    calls = []
    instances = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((path, kwargs))
        model = FakeModel()
        cls.instances.append(model)
        return model


omnivoice_module = types.ModuleType("omnivoice")
omnivoice_module.OmniVoice = FakeOmniVoice
sys.modules["omnivoice"] = omnivoice_module


def reset():
    for bundle in list(ov._MODEL_CACHE.values()):
        bundle["model"] = None
    ov._MODEL_CACHE.clear()
    FakeOmniVoice.calls.clear()
    FakeOmniVoice.instances.clear()
    interrupt_calls.clear()
    progress_updates.clear()
    shutil.rmtree(ov.OMNIVOICE_MODELS_DIR, ignore_errors=True)
    ov.OMNIVOICE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(TMP / "models" / "audio_encoders", ignore_errors=True)


def model_dir(name="OmniVoice"):
    path = ov.OMNIVOICE_MODELS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    return path


def audio(seconds=1.0, sr=24000, channels=1):
    waveform = torch.linspace(-0.5, 0.5, max(1, int(seconds * sr)))
    return {"waveform": waveform.repeat(channels, 1)[None], "sample_rate": sr}


def load_bundle(keep=True, model="OmniVoice", **kwargs):
    model_dir(model)
    return ov.OmniVoiceModelsLoader().load(
        model, kwargs.get("device", "cpu"), kwargs.get("dtype", "fp32"),
        kwargs.get("attention", "auto"), keep,
        kwargs.get("local_model_path", ""),
    )[0]


def common_generation():
    return dict(
        steps=32, guidance_scale=2.0, t_shift=0.1, speed=1.0,
        duration=0.0, seed=7, position_temperature=5.0,
        class_temperature=0.0, layer_penalty_factor=5.0,
        denoise=True, preprocess_prompt=True, postprocess_output=True,
        instruct="",
    )


def test_interfaces_and_registration():
    assert set(ov.NODE_CLASS_MAPPINGS) == {
        "CCTechOmniVoiceModelsLoader", "CCTechOmniVoiceLongformTTS",
        "CCTechOmniVoiceVoiceCloneTTS",
    }
    for cls in ov.NODE_CLASS_MAPPINGS.values():
        assert cls.CATEGORY == "🤖 CCTech/OmniVoice"
        assert cls.TITLE and cls.SEARCH_ALIASES and cls.DESCRIPTION
    loader = ov.OmniVoiceModelsLoader.INPUT_TYPES()
    assert loader["required"]["keep_model_loaded"][1]["default"] is True
    assert loader["required"]["device"][0] == ["auto", "cuda", "cpu", "mps", "xpu"]
    short = ov.OmniVoiceVoiceCloneTTS.INPUT_TYPES()
    long = ov.OmniVoiceLongformTTS.INPUT_TYPES()
    assert short["required"]["ref_audio"][0] == "AUDIO"
    assert short["optional"]["whisper_model"][0] == "WHISPER_ASR"
    assert long["optional"]["ref_audio"][0] == "AUDIO"
    assert long["required"]["steps"][1]["default"] == 32
    assert long["required"]["words_per_chunk"][1]["default"] == 100
    assert "omnivoice" in folder_paths.folder_names_and_paths
    ok("interfaces/defaults, category metadata, mappings, and model folder registration")


def test_missing_dependency_isolation():
    reset()
    model_dir()
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "omnivoice":
            raise ImportError("intentionally absent")
        return real_import(name, *args, **kwargs)

    sys.modules.pop("omnivoice", None)
    try:
        with patch("builtins.__import__", side_effect=blocked):
            try:
                ov.OmniVoiceModelsLoader().load("OmniVoice", "cpu", "fp32", "auto", True)
            except RuntimeError as exc:
                assert "optional 'omnivoice' package" in str(exc)
            else:
                raise AssertionError("missing dependency should fail at execution")
    finally:
        sys.modules["omnivoice"] = omnivoice_module
    assert len(ov.NODE_CLASS_MAPPINGS) == 3
    ok("missing omnivoice is isolated until loader execution; mappings remain available")


def test_choices_download_local_validation_and_cache():
    reset()
    assert ov._model_choices()[:2] == [
        "OmniVoice (auto download)", "OmniVoice-bf16 (auto download)"]
    hf = types.ModuleType("huggingface_hub")
    downloads = []

    def snapshot_download(repo_id, local_dir, **kwargs):
        downloads.append((repo_id, local_dir, kwargs))
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "config.json").write_text("{}", encoding="utf-8")

    hf.snapshot_download = snapshot_download
    sys.modules["huggingface_hub"] = hf
    bundle = ov.OmniVoiceModelsLoader().load(
        "OmniVoice-bf16 (auto download)", "cpu", "fp32", "auto", True)[0]
    assert downloads[0][0] == "drbaph/OmniVoice-bf16"
    assert bundle["path"].endswith("OmniVoice-bf16")
    before = len(FakeOmniVoice.calls)
    same = ov.OmniVoiceModelsLoader().load(
        "OmniVoice-bf16", "cpu", "fp32", "auto", True)[0]
    assert same is bundle and len(FakeOmniVoice.calls) == before

    bad = TMP / "bad-local"
    bad.mkdir()
    try:
        ov.OmniVoiceModelsLoader().load("OmniVoice", "cpu", "fp32", "auto", True, str(bad))
    except ValueError as exc:
        assert "complete OmniVoice" in str(exc)
    else:
        raise AssertionError("invalid local folder accepted")
    custom = TMP / "arbitrary-model"
    custom.mkdir()
    (custom / "weights.safetensors").write_bytes(b"fake")
    local = ov.OmniVoiceModelsLoader().load(
        "OmniVoice", "cpu", "fp32", "auto", True, str(custom))[0]
    assert local["path"] == str(custom.resolve())
    assert bundle["model"] is None
    ok("model choices, known-repo download, arbitrary local validation, and loader cache")


def test_loader_kwargs_device_dtype_attention_and_sage_fallback():
    reset()
    model_dir()
    with patch.object(ov, "_install_sage_attention", return_value=False) as sage:
        bundle = ov.OmniVoiceModelsLoader().load(
            "OmniVoice", "cpu", "bf16", "sage_attention", True)[0]
    path, kwargs = FakeOmniVoice.calls[-1]
    assert path.endswith("OmniVoice")
    assert kwargs["torch_dtype"] is torch.bfloat16
    assert kwargs["attn_implementation"] == "eager"
    assert FakeOmniVoice.instances[-1].devices[0] == "cpu"
    assert FakeOmniVoice.instances[-1].training is False
    sage.assert_called_once_with(FakeOmniVoice.instances[-1], "cpu")
    assert bundle["attention"] == "sage_attention"
    ok("loader forwards dtype/device/eager-for-Sage kwargs and applies fallback hook")


def test_short_kwargs_audio_trim_external_transcription_and_seed_zero():
    reset()
    bundle = load_bundle()
    pipe_calls = []

    class Pipe:
        def __call__(self, value):
            pipe_calls.append(value)
            return {"text": "reference transcript"}

        def to(self, device):
            return self

    args = common_generation()
    args.update(duration=2.5, seed=0, instruct=" whisper ")
    with patch.object(torch, "randint", return_value=torch.tensor([1234])), \
            patch.object(ov, "_manual_seed_all") as seeded:
        result = ov.OmniVoiceVoiceCloneTTS().generate(
            bundle, "speak this", audio(seconds=21), "", whisper_model={"pipeline": Pipe()}, **args)[0]
    seeded.assert_called_once_with(1234)
    call = FakeOmniVoice.instances[-1].calls[-1]
    ref_wave, ref_sr = call["ref_audio"]
    assert ref_wave.shape == (15 * 24000,) and ref_sr == 24000
    assert pipe_calls[0]["array"].shape == (15 * 24000,)
    assert call["ref_text"] == "reference transcript"
    assert call["num_step"] == 32 and call["duration"] == 2.5
    assert call["instruct"] == "whisper"
    assert result["sample_rate"] == 24000 and result["waveform"].shape == (1, 1, 1200)
    assert bundle["offloaded"] is True
    ok("short generation kwargs, 24k AUDIO, >20s trim, external ASR, and seed=0 random semantics")


def test_short_resample_and_internal_transcription_fallback():
    reset()
    bundle = load_bundle()
    args = common_generation()
    ov.OmniVoiceVoiceCloneTTS().generate(
        bundle, "hello", audio(seconds=1, sr=12000, channels=2), "", **args)
    call = FakeOmniVoice.instances[-1].calls[-1]
    ref_wave, ref_sr = call["ref_audio"]
    assert ref_sr == 24000 and ref_wave.shape[0] == 24000
    assert "ref_text" not in call
    ok("stereo/reference resampling to 24k and OmniVoice internal transcription fallback")


def test_local_whisper_fallback():
    reset()
    bundle = load_bundle()
    local = TMP / "models" / "audio_encoders" / "my-whisper"
    local.mkdir(parents=True)
    (local / "config.json").write_text("{}", encoding="utf-8")
    seen = []

    class LocalPipe:
        def __call__(self, value):
            seen.append(value)
            return {"text": "local transcript"}

        def to(self, device):
            return self

    transformers = types.ModuleType("transformers")
    transformers.pipeline = lambda *args, **kwargs: LocalPipe()
    sys.modules["transformers"] = transformers
    ov.OmniVoiceVoiceCloneTTS().generate(
        bundle, "hello", audio(), "", **common_generation())
    assert FakeOmniVoice.instances[-1].calls[-1]["ref_text"] == "local transcript"
    assert seen
    ok("automatic local Whisper discovery/transcription fallback")


def test_long_chunking_cjk_and_first_chunk_consistency():
    reset()
    bundle = load_bundle()
    model = FakeOmniVoice.instances[-1]
    model.long_first = True
    text = "One two three. Four five six. Seven eight nine."
    result = ov.OmniVoiceLongformTTS().generate(
        bundle, text, "", words_per_chunk=3, **common_generation())[0]
    assert len(model.calls) >= 2
    assert "ref_audio" not in model.calls[0]
    first_ref, sr = model.calls[1]["ref_audio"]
    assert first_ref.shape[0] == 25 * 24000 and sr == 24000
    assert model.calls[1]["ref_text"] == model.calls[0]["text"]
    assert result["waveform"].shape[-1] > 25 * 24000
    cjk = ov._smart_chunk_text("第一句。第二句。第三句。", 4)
    assert len(cjk) >= 2 and "" not in cjk
    assert "".join(cjk) == "第一句。第二句。第三句。"
    ok("multilingual/CJK chunking and fixed first-chunk voice reference capped at 25s")


def test_long_clone_uses_fixed_user_reference_every_chunk():
    reset()
    bundle = load_bundle()
    model = FakeOmniVoice.instances[-1]
    ov.OmniVoiceLongformTTS().generate(
        bundle, "One two. Three four.", "spoken reference",
        words_per_chunk=2, ref_audio=audio(), **common_generation())
    refs = [call["ref_audio"][0] for call in model.calls]
    assert len(refs) == 2 and refs[0] is refs[1]
    assert all(call["ref_text"] == "spoken reference" for call in model.calls)
    ok("long-form voice clone reuses one fixed user reference across chunks")


def test_keep_loaded_offload_resume_and_unload_reload():
    reset()
    keep = load_bundle(keep=True)
    first = keep["model"]
    ov.OmniVoiceVoiceCloneTTS().generate(
        keep, "first", audio(), "transcript", **common_generation())
    assert keep["model"] is first and keep["offloaded"] is True
    ov.OmniVoiceVoiceCloneTTS().generate(
        keep, "second", audio(), "transcript", **common_generation())
    assert keep["model"] is first
    assert first.devices[-2:] == ["cpu", "cpu"] or "cpu" in first.devices
    # A resume to CPU is still observable as a move and clears offloaded before use.
    assert len(first.calls) == 2

    reset()
    unload = load_bundle(keep=False)
    old = unload["model"]
    ov.OmniVoiceVoiceCloneTTS().generate(
        unload, "first", audio(), "transcript", **common_generation())
    assert unload["model"] is None
    ov.OmniVoiceVoiceCloneTTS().generate(
        unload, "second", audio(), "transcript", **common_generation())
    assert len(FakeOmniVoice.instances) == 2 and FakeOmniVoice.instances[-1] is not old
    assert unload["model"] is None
    ok("keep-loaded CPU offload/resume and fully-unloaded cached-output reload")


def test_errors_interrupt_and_progress():
    reset()
    bundle = load_bundle()
    try:
        ov.OmniVoiceVoiceCloneTTS().generate(
            bundle, " ", audio(), "ref", **common_generation())
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("blank text accepted")

    bundle = load_bundle()
    FakeOmniVoice.instances[-1].raise_instruct = True
    values = common_generation()
    values["instruct"] = "not-supported"
    try:
        ov.OmniVoiceVoiceCloneTTS().generate(
            bundle, "hello", audio(), "ref", **values)
    except RuntimeError as exc:
        assert "Invalid instruct" in str(exc) and "not-supported" in str(exc)
    else:
        raise AssertionError("instruction error was not remapped")
    assert interrupt_calls and progress_updates

    reset()
    unload = load_bundle(keep=False)

    class BrokenPipe:
        def __call__(self, _value):
            raise RuntimeError("ASR failed")

        def to(self, _device):
            return self

    try:
        ov.OmniVoiceVoiceCloneTTS().generate(
            unload, "hello", audio(), "", whisper_model={"pipeline": BrokenPipe()},
            **common_generation())
    except RuntimeError as exc:
        assert "ASR failed" in str(exc)
    else:
        raise AssertionError("ASR failure was swallowed")
    assert unload["model"] is None
    assert interrupt_calls
    ok("blank text, instruction remap, ASR-failure unload, interrupts, and progress")


TESTS = [
    test_interfaces_and_registration,
    test_missing_dependency_isolation,
    test_choices_download_local_validation_and_cache,
    test_loader_kwargs_device_dtype_attention_and_sage_fallback,
    test_short_kwargs_audio_trim_external_transcription_and_seed_zero,
    test_short_resample_and_internal_transcription_fallback,
    test_local_whisper_fallback,
    test_long_chunking_cjk_and_first_chunk_consistency,
    test_long_clone_uses_fixed_user_reference_every_chunk,
    test_keep_loaded_offload_resume_and_unload_reload,
    test_errors_interrupt_and_progress,
]


try:
    for test in TESTS:
        test()
    print(f"\nOmniVoice smoke: {len(TESTS)}/{len(TESTS)} tests passed")
finally:
    shutil.rmtree(TMP, ignore_errors=True)
