# (c) CCTech || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""OmniVoice TTS nodes, ported from ComfyUI-omnivoice-tts.

Source: Saganaki22/ComfyUI-OmniVoice-TTS (Apache-2.0), adapted from the local
checkout at ``D:/Projects/ComfyUI/ComfyUI-omnivoice-tts``.  This port keeps
the source Longform and Voice Clone generation surfaces while moving model,
device, dtype, attention, and residency controls to an explicit house-style
loader.  The source repository's Apache-2.0 attribution is retained here.

The ``omnivoice`` package and all optional runtime integrations are imported
lazily.  A missing or incompatible optional dependency therefore cannot stop
this module from registering, or prevent the rest of Comfy-GGUF from loading.
"""

from __future__ import annotations

import gc
import importlib.util
import logging
import os
import re
import threading
from pathlib import Path

import numpy as np
import torch

try:
    import folder_paths
except ImportError:  # lets the module remain inspectable outside ComfyUI
    folder_paths = None


logger = logging.getLogger(__name__)
CATEGORY = "\U0001F916 CCTech/OmniVoice"
OMNIVOICE_SAMPLE_RATE = 24000
MODELS_FOLDER_NAME = "omnivoice"
AUTO_DOWNLOAD_SUFFIX = " (auto download)"

HF_MODELS = {
    "OmniVoice": "k2-fsa/OmniVoice",
    "OmniVoice-bf16": "drbaph/OmniVoice-bf16",
}
DEVICE_CHOICES = ["auto", "cuda", "cpu", "mps", "xpu"]
DTYPE_CHOICES = ["auto", "bf16", "fp16", "fp32"]
ATTENTION_CHOICES = ["auto", "eager", "sage_attention"]


def _models_dir() -> Path:
    if folder_paths is not None:
        base = Path(folder_paths.models_dir) / MODELS_FOLDER_NAME
    else:
        base = Path(__file__).resolve().parents[1] / "models" / MODELS_FOLDER_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


OMNIVOICE_MODELS_DIR = _models_dir()
if folder_paths is not None:
    try:
        folder_paths.add_model_folder_path(MODELS_FOLDER_NAME, str(OMNIVOICE_MODELS_DIR))
    except Exception:
        logger.debug("Could not register models/omnivoice", exc_info=True)


def _valid_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "config.json").is_file():
        return True
    suffixes = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf"}
    try:
        return any(item.is_file() and item.suffix.lower() in suffixes for item in path.iterdir())
    except OSError:
        return False


def _known_model_path(model_name: str) -> Path:
    return OMNIVOICE_MODELS_DIR / model_name


def _model_choices() -> list[str]:
    choices = []
    known_names = set(HF_MODELS)
    for name in HF_MODELS:
        choices.append(f"{name}{AUTO_DOWNLOAD_SUFFIX}")
        if _valid_model_dir(_known_model_path(name)):
            choices.append(name)
    try:
        for path in sorted(OMNIVOICE_MODELS_DIR.iterdir(), key=lambda item: item.name.lower()):
            if path.name not in known_names and _valid_model_dir(path):
                choices.append(path.name)
    except OSError:
        pass
    return choices


def _download_model(model_name: str) -> str:
    repo_id = HF_MODELS[model_name]
    target = _known_model_path(model_name)
    target.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    logger.info("OmniVoice: downloading %s to %s", repo_id, target)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "*.h5"],
    )
    if not _valid_model_dir(target):
        raise RuntimeError(f"Downloaded OmniVoice folder is incomplete: {target}")
    return str(target)


def _resolve_model_path(model_name: str, local_model_path: str = "") -> str:
    override = (local_model_path or "").strip()
    if override:
        path = Path(override).expanduser()
        if not _valid_model_dir(path):
            raise ValueError(
                f"local_model_path {override!r} is not a complete OmniVoice model folder "
                "(expected config.json or model weights)."
            )
        return str(path.resolve())

    name = model_name
    if name.endswith(AUTO_DOWNLOAD_SUFFIX):
        name = name[: -len(AUTO_DOWNLOAD_SUFFIX)]
    if name in HF_MODELS:
        path = _known_model_path(name)
        return str(path) if _valid_model_dir(path) else _download_model(name)
    path = OMNIVOICE_MODELS_DIR / name
    if not _valid_model_dir(path):
        raise FileNotFoundError(
            f"OmniVoice model {name!r} was not found under {OMNIVOICE_MODELS_DIR}."
        )
    return str(path)


def _xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if _xpu_available():
        return "xpu"
    return "cpu"


def _supports_bf16() -> bool:
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_capability()[0] >= 8
        except Exception:
            return False
    return _xpu_available()


def _resolve_dtype(choice: str, device: str) -> torch.dtype:
    if choice == "bf16":
        if device == "cuda" and not _supports_bf16():
            logger.warning(
                "OmniVoice: bf16 was requested on a pre-Ampere CUDA device; "
                "fp16 is usually the compatible choice"
            )
        return torch.bfloat16
    if choice == "fp16":
        return torch.float16
    if choice == "fp32":
        return torch.float32
    if device == "cuda":
        return torch.bfloat16 if _supports_bf16() else torch.float16
    if device == "xpu":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def _target_device(device: str) -> str:
    return "cuda:0" if device == "cuda" else device


def _empty_accelerator_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if _xpu_available():
        torch.xpu.empty_cache()


def _move_pipeline(pipe, device: str) -> None:
    for attr in ("model", "_model"):
        inner = getattr(pipe, attr, None)
        if inner is not None and hasattr(inner, "to"):
            try:
                inner.to(device)
                return
            except Exception:
                pass
    try:
        pipe.to(device)
    except Exception:
        pass


def _sage_forward_factory(sage_fn, v2: bool, target_dtype: torch.dtype | None = None,
                          pv_accum_dtype: str = "fp32"):
    def _forward(self, hidden_states, position_embeddings, attention_mask=None,
                 past_key_values=None, cache_position=None, **kwargs):
        if attention_mask is not None or past_key_values is not None:
            return self._omnivoice_original_forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        original_dtype = q.dtype
        dtype = target_dtype or getattr(self.q_proj.weight, "dtype", torch.float16)
        if not v2 and self.num_key_value_groups != 1:
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)
        if v2:
            output = sage_fn(
                q.to(dtype), k.to(dtype), v.to(dtype), tensor_layout="HND",
                is_causal=True, qk_quant_gran="per_warp",
                pv_accum_dtype=pv_accum_dtype,
            )
        else:
            output = sage_fn(q.to(dtype), k.to(dtype), v.to(dtype),
                             is_causal=True, smooth_k=True)
        if isinstance(output, tuple):
            output = output[0]
        output = output.to(original_dtype).transpose(1, 2).contiguous()
        return self.o_proj(output.reshape(*input_shape, -1)), None
    return _forward


def _install_sage_attention(model, device: str) -> bool:
    """Use source-compatible SageAttention V2, then V1, else eager."""
    sage_fn = None
    v2 = False
    target_dtype = None
    pv_accum_dtype = "fp32"
    if device == "cuda":
        try:
            from sageattention.core import (
                sageattn_qk_int8_pv_fp16_cuda,
                sageattn_qk_int8_pv_fp8_cuda,
                sageattn_qk_int8_pv_fp8_cuda_sm90,
            )
            arch = torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1]
            if arch >= 120:
                sage_fn = sageattn_qk_int8_pv_fp8_cuda
                pv_accum_dtype = "fp32+fp32"
            elif arch >= 90:
                sage_fn = sageattn_qk_int8_pv_fp8_cuda_sm90
                pv_accum_dtype = "fp32+fp32"
            elif arch == 89:
                sage_fn = sageattn_qk_int8_pv_fp8_cuda
                pv_accum_dtype = "fp32+fp32"
            elif arch >= 80:
                sage_fn = sageattn_qk_int8_pv_fp16_cuda
            if sage_fn is not None:
                v2 = True
        except Exception:
            pass
    if sage_fn is None:
        try:
            from sageattention import sageattn
            sage_fn = sageattn
            target_dtype = (torch.bfloat16 if device in ("cuda", "xpu") and _supports_bf16()
                            else torch.float16 if device == "cuda" else torch.float32)
        except Exception:
            logger.warning("OmniVoice: SageAttention unavailable; falling back to eager attention")
            return False

    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
        replacement = _sage_forward_factory(
            sage_fn, v2, target_dtype, pv_accum_dtype)
        count = 0
        for module in model.modules():
            if isinstance(module, Qwen3Attention):
                if not hasattr(module, "_omnivoice_original_forward"):
                    module._omnivoice_original_forward = module.forward
                module.forward = replacement.__get__(module, Qwen3Attention)
                count += 1
        logger.info("OmniVoice: SageAttention patched %d Qwen3 attention layers", count)
        return count > 0
    except Exception as exc:
        logger.warning("OmniVoice: SageAttention patch failed (%s); using eager", exc)
        return False


def _load_raw_model(path: str, device: str, dtype: torch.dtype, attention: str):
    try:
        from omnivoice import OmniVoice
    except Exception as exc:
        detail = str(exc)
        if "HiggsAudioV2Tokenizer" in detail:
            detail += " Install or upgrade to transformers>=5.3.0."
        raise RuntimeError(
            "OmniVoice nodes require the optional 'omnivoice' package and its runtime "
            f"dependencies. Import failed: {detail}"
        ) from exc

    kwargs = {"torch_dtype": dtype}
    if importlib.util.find_spec("accelerate") is not None:
        kwargs["low_cpu_mem_usage"] = True
    if attention in ("eager", "sage_attention"):
        kwargs["attn_implementation"] = "eager"
    model = OmniVoice.from_pretrained(path, **kwargs)
    model = model.to(_target_device(device))
    model.eval()
    if attention == "sage_attention":
        _install_sage_attention(model, device)
    return model


_MODEL_CACHE: dict[tuple, dict] = {}
_CACHE_LOCK = threading.RLock()
_INFERENCE_LOCK = threading.RLock()


def _bundle_key(path: str, device: str, dtype_name: str, attention: str,
                keep_model_loaded: bool) -> tuple:
    return (os.path.normcase(os.path.abspath(path)), device, dtype_name,
            attention, keep_model_loaded)


def _new_bundle(path: str, device: str, dtype_name: str, attention: str,
                keep_model_loaded: bool) -> dict:
    resolved_device = _resolve_device(device)
    return {
        "model": None,
        "path": path,
        "device": resolved_device,
        "target_device": _target_device(resolved_device),
        "dtype_name": dtype_name,
        "dtype": _resolve_dtype(dtype_name, resolved_device),
        "attention": attention,
        "keep_model_loaded": bool(keep_model_loaded),
        "offloaded": False,
        "lock": threading.RLock(),
    }


def _ensure_model(bundle: dict):
    if not isinstance(bundle, dict) or "path" not in bundle:
        raise ValueError("omnivoice_model is not a valid OMNIVOICE_MODEL loader output")
    with bundle["lock"]:
        model = bundle.get("model")
        if model is None:
            model = _load_raw_model(bundle["path"], bundle["device"],
                                    bundle["dtype"], bundle["attention"])
            bundle["model"] = model
            bundle["offloaded"] = False
        elif bundle.get("offloaded"):
            model.to(bundle["target_device"])
            internal_asr = getattr(model, "_asr_pipe", None)
            if internal_asr is not None:
                _move_pipeline(internal_asr, bundle["target_device"])
            bundle["offloaded"] = False
            logger.info("OmniVoice: resumed cached model on %s", bundle["target_device"])
        return model


def _release_bundle(bundle: dict, force_unload: bool = False) -> None:
    with bundle["lock"]:
        model = bundle.get("model")
        if model is None:
            return
        internal_asr = getattr(model, "_asr_pipe", None)
        if bundle["keep_model_loaded"] and not force_unload:
            try:
                model.to("cpu")
                if internal_asr is not None:
                    _move_pipeline(internal_asr, "cpu")
                bundle["offloaded"] = True
                logger.info("OmniVoice: model offloaded to CPU; cached bundle can resume it")
            except Exception as exc:
                bundle["offloaded"] = False
                logger.warning("OmniVoice: CPU offload failed: %s", exc)
        else:
            if internal_asr is not None:
                _move_pipeline(internal_asr, "cpu")
            try:
                model.to("cpu")
            except Exception:
                pass
            bundle["model"] = None
            bundle["offloaded"] = False
            with _CACHE_LOCK:
                for key, value in list(_MODEL_CACHE.items()):
                    if value is bundle:
                        _MODEL_CACHE.pop(key, None)
            logger.info("OmniVoice: model unloaded; cached loader output will reload on reuse")
        gc.collect()
        _empty_accelerator_cache()


def _manual_seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if _xpu_available():
        torch.xpu.manual_seed_all(seed)
    np.random.seed(seed % (2**32))


def _actual_seed(seed: int) -> int:
    return seed if seed != 0 else int(torch.randint(0, 2**31, (1,)).item())


def _to_numpy_audio(audio) -> np.ndarray:
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim >= 2 and audio.shape[0] == 1:
        audio = audio.squeeze(0)
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    return np.ascontiguousarray(audio.reshape(-1), dtype=np.float32)


def _numpy_to_audio(audio: np.ndarray) -> dict:
    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32).reshape(1, 1, -1)).contiguous()
    return {"waveform": waveform, "sample_rate": OMNIVOICE_SAMPLE_RATE}


def _resample(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return np.ascontiguousarray(audio, dtype=np.float32)
    try:
        import soxr
        return np.asarray(soxr.resample(audio, source_sr, target_sr), dtype=np.float32)
    except ImportError:
        tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32))[None, None]
        size = max(1, round(tensor.shape[-1] * target_sr / source_sr))
        return torch.nn.functional.interpolate(
            tensor, size=size, mode="linear", align_corners=False
        )[0, 0].numpy()


def _comfy_audio_to_numpy(audio: dict, target_sr: int = OMNIVOICE_SAMPLE_RATE) -> np.ndarray:
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise ValueError("ref_audio must be a ComfyUI AUDIO value")
    source_sr = int(audio.get("sample_rate") or 0)
    if source_sr <= 0:
        raise ValueError("ref_audio carries no valid sample_rate")
    waveform = audio["waveform"]
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().float().cpu().numpy()
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=0)
    waveform = waveform.reshape(-1)
    if waveform.size == 0:
        raise ValueError("ref_audio waveform is empty")
    return _resample(waveform, source_sr, target_sr)


def _prepare_auto_reference(audio: np.ndarray, preprocess_prompt: bool) -> np.ndarray:
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if not preprocess_prompt or len(audio) <= 20 * OMNIVOICE_SAMPLE_RATE:
        return audio
    max_samples = 15 * OMNIVOICE_SAMPLE_RATE
    try:
        from omnivoice.utils.audio import trim_long_audio
        trimmed = trim_long_audio(
            audio[None], OMNIVOICE_SAMPLE_RATE, max_duration=15.0,
            min_duration=3.0, trim_threshold=20.0,
        )
        trimmed = _to_numpy_audio(trimmed)
    except Exception as exc:
        logger.warning("OmniVoice: silence-aware reference trim failed (%s); using first 15s", exc)
        trimmed = audio[:max_samples]
    if trimmed.size == 0 or trimmed.size > max_samples:
        trimmed = audio[:max_samples]
    logger.info("OmniVoice: auto-reference trimmed to %.1fs", len(trimmed) / OMNIVOICE_SAMPLE_RATE)
    return np.ascontiguousarray(trimmed, dtype=np.float32)


def _log_reference_duration(audio: np.ndarray) -> None:
    duration = len(audio) / OMNIVOICE_SAMPLE_RATE
    if duration < 1.0:
        logger.warning(
            "OmniVoice: reference audio is only %.1fs; 3-15s works best", duration)
    elif duration > 30.0:
        logger.warning(
            "OmniVoice: reference audio is %.1fs; longer than 15s may cause issues",
            duration,
        )


def _whisper_models_dir() -> Path:
    if folder_paths is not None:
        return Path(folder_paths.models_dir) / "audio_encoders"
    return OMNIVOICE_MODELS_DIR.parent / "audio_encoders"


def _find_local_whisper() -> str | None:
    base = _whisper_models_dir()
    preferred = ["openai_whisper-large-v3-turbo", "openai_whisper-large-v3",
                 "openai_whisper-medium", "openai_whisper-small", "openai_whisper-tiny"]
    paths = [base / name for name in preferred]
    try:
        paths.extend(path for path in sorted(base.iterdir()) if path not in paths)
    except OSError:
        pass
    return str(next((path for path in paths if _valid_model_dir(path)), "")) or None


def _load_whisper(path: str, bundle: dict):
    from transformers import pipeline
    return pipeline(
        "automatic-speech-recognition", model=path,
        torch_dtype=bundle["dtype"], device=bundle["target_device"],
    )


def _external_whisper_pipeline(whisper_model):
    if whisper_model is None:
        return None
    if isinstance(whisper_model, dict):
        return whisper_model.get("pipeline") or whisper_model.get("pipe")
    return whisper_model


def _transcribe_reference(audio: np.ndarray, whisper_model, bundle: dict) -> str:
    pipe = _external_whisper_pipeline(whisper_model)
    borrowed = pipe is not None
    if pipe is None:
        local = _find_local_whisper()
        if local:
            try:
                pipe = _load_whisper(local, bundle)
                logger.info("OmniVoice: using local Whisper model %s", local)
            except Exception as exc:
                logger.warning("OmniVoice: local Whisper load failed (%s); using internal fallback", exc)
    if pipe is None:
        logger.info("OmniVoice: no external/local Whisper available; model will auto-transcribe")
        return ""
    try:
        _move_pipeline(pipe, bundle["target_device"])
        result = pipe({"array": audio.astype(np.float32, copy=False),
                       "sampling_rate": OMNIVOICE_SAMPLE_RATE})
        return str(result.get("text", "") if isinstance(result, dict) else result).strip()
    finally:
        _move_pipeline(pipe, "cpu")
        if not borrowed:
            del pipe
            gc.collect()
            _empty_accelerator_cache()


def _progress(total: int):
    try:
        from comfy.utils import ProgressBar
        return ProgressBar(total)
    except Exception:
        return None


def _check_interrupt() -> None:
    try:
        import comfy.model_management as mm
        mm.throw_exception_if_processing_interrupted()
    except ImportError:
        return


def _update_progress(pbar, value: int, total: int) -> None:
    if pbar is not None:
        pbar.update_absolute(value, total)


def _is_cjk(char: str) -> bool:
    cp = ord(char)
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
            or 0x20000 <= cp <= 0x2A6DF or 0xF900 <= cp <= 0xFAFF
            or 0x3040 <= cp <= 0x30FF or 0xAC00 <= cp <= 0xD7AF
            or 0x0E00 <= cp <= 0x0EFF or 0x1000 <= cp <= 0x109F
            or 0x1780 <= cp <= 0x17FF)


def _chunk_by_characters(text: str, chars_per_chunk: int, sentence_end) -> list[str]:
    if len(text) <= chars_per_chunk:
        return [text]
    chunks, pos = [], 0
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        target = min(pos + chars_per_chunk, len(text))
        if target == len(text):
            chunk = text[pos:].strip()
            if chunk:
                chunks.append(chunk)
            break
        matches = list(sentence_end.finditer(text[pos:target]))
        end = pos + matches[-1].end() if matches else target
        chunk = text[pos:end].strip()
        if chunk:
            chunks.append(chunk)
        pos = end
    return chunks or [text]


def _smart_chunk_text(text: str, words_per_chunk: int = 100) -> list[str]:
    if words_per_chunk <= 0:
        return [text]
    sentence_end = re.compile(r"(?:[.!?]+(?:\s|$)|[。？！\u0964\u0965\u061F\u104B\u0F0D]+)")
    cjk_count = sum(_is_cjk(ch) for ch in text)
    alpha_count = sum(ch.isalpha() or _is_cjk(ch) for ch in text)
    if alpha_count and cjk_count / alpha_count > 0.3:
        return _chunk_by_characters(text, words_per_chunk, sentence_end)
    words = text.split()
    if len(words) <= words_per_chunk:
        return [text]
    chunks, current = [], []
    for word in words:
        current.append(word)
        if len(current) >= words_per_chunk:
            joined = " ".join(current)
            matches = list(sentence_end.finditer(joined))
            if matches:
                split = matches[-1].end()
                final, remaining = joined[:split].strip(), joined[split:].strip()
                if final:
                    chunks.append(final)
                current = remaining.split() if remaining else []
            else:
                chunks.append(joined)
                current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


def _generation_widgets(include_chunking: bool) -> dict:
    required = {
        "text": ("STRING", {"multiline": True, "default":
            "Hello! This is a test of OmniVoice text to speech synthesis."}),
        "ref_text": ("STRING", {"multiline": True, "default": "",
            "tooltip": "Reference transcript. Empty uses connected/local/internal Whisper."}),
        "steps": ("INT", {"default": 32, "min": 4, "max": 64, "step": 1}),
        "guidance_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
        "t_shift": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
        "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
        "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.5}),
        "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1,
                          "tooltip": "Random seed. 0 = random."}),
        "position_temperature": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.5}),
        "class_temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1}),
        "layer_penalty_factor": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.5}),
        "denoise": ("BOOLEAN", {"default": True}),
        "preprocess_prompt": ("BOOLEAN", {"default": True}),
        "postprocess_output": ("BOOLEAN", {"default": True}),
        "instruct": ("STRING", {"multiline": False, "default": "",
            "tooltip": "A model-supported comma-separated dialect/style instruction."}),
    }
    if include_chunking:
        required["words_per_chunk"] = ("INT", {"default": 100, "min": 0,
                                                  "max": 500, "step": 10})
    return required


def _generate_kwargs(text, steps, guidance_scale, t_shift, speed,
                     position_temperature, class_temperature,
                     layer_penalty_factor, denoise, preprocess_prompt,
                     postprocess_output, duration, instruct) -> dict:
    kwargs = {
        "text": text, "num_step": steps, "guidance_scale": guidance_scale,
        "t_shift": t_shift, "speed": speed,
        "position_temperature": position_temperature,
        "class_temperature": class_temperature,
        "layer_penalty_factor": layer_penalty_factor, "denoise": denoise,
        "preprocess_prompt": preprocess_prompt,
        "postprocess_output": postprocess_output,
    }
    if duration > 0:
        kwargs["duration"] = duration
    if instruct and instruct.strip():
        kwargs["instruct"] = instruct.strip()
    return kwargs


def _call_generate(model, kwargs):
    try:
        with torch.inference_mode():
            return model.generate(**kwargs)
    except ValueError as exc:
        message = str(exc).lower()
        if "instruct" in message or "unsupported" in message:
            raise RuntimeError(
                f"Invalid instruct value {kwargs.get('instruct')!r}. The model only accepts "
                f"specific values. Original error:\n{exc}"
            ) from exc
        raise


class OmniVoiceModelsLoader:
    CATEGORY = CATEGORY
    TITLE = "OmniVoice Models Loader ⚡"
    SEARCH_ALIASES = ["omnivoice loader", "load tts model", "speech model", "voice model"]
    DESCRIPTION = ("Load OmniVoice from models/omnivoice, a known Hugging Face model, "
                   "or an arbitrary local model folder.")
    RETURN_TYPES = ("OMNIVOICE_MODEL",)
    RETURN_NAMES = ("omnivoice_model",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        choices = _model_choices()
        return {"required": {
            "model": (choices, {"default": choices[0]}),
            "device": (DEVICE_CHOICES, {"default": "auto"}),
            "dtype": (DTYPE_CHOICES, {"default": "auto"}),
            "attention": (ATTENTION_CHOICES, {"default": "auto"}),
            "keep_model_loaded": ("BOOLEAN", {"default": True,
                "tooltip": "Keep weights cached on CPU between runs; otherwise fully unload and reload safely."}),
        }, "optional": {
            "local_model_path": ("STRING", {"default": "",
                "tooltip": "Absolute path to any complete local OmniVoice model folder."}),
        }}

    def load(self, model, device, dtype, attention, keep_model_loaded,
             local_model_path=""):
        path = _resolve_model_path(model, local_model_path)
        key = _bundle_key(path, device, dtype, attention, keep_model_loaded)
        with _INFERENCE_LOCK:
            with _CACHE_LOCK:
                bundle = _MODEL_CACHE.get(key)
                old_bundles = [] if bundle is not None else list(_MODEL_CACHE.values())
                if bundle is None:
                    _MODEL_CACHE.clear()
                    bundle = _new_bundle(path, device, dtype, attention, keep_model_loaded)
                    _MODEL_CACHE[key] = bundle
            for old in old_bundles:
                _release_bundle(old, force_unload=True)
            _ensure_model(bundle)
        return (bundle,)


class OmniVoiceVoiceCloneTTS:
    CATEGORY = CATEGORY
    TITLE = "OmniVoice Voice Clone TTS ⚡"
    SEARCH_ALIASES = ["omnivoice", "voice clone", "short form tts", "reference audio", "speech"]
    DESCRIPTION = ("Short-form zero-shot voice cloning for 600+ languages, with optional "
                   "connected Whisper ASR or automatic local/internal transcription.")
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        required = {"omnivoice_model": ("OMNIVOICE_MODEL",)}
        required.update(_generation_widgets(False))
        required["text"] = ("STRING", {"multiline": True, "default":
            "Hello! This is a test of voice cloning with OmniVoice."})
        # Preserve source order: ref_audio is a required short-form input.
        ordered = {"omnivoice_model": required.pop("omnivoice_model"),
                   "text": required.pop("text"),
                   "ref_audio": ("AUDIO", {"tooltip": "3-15 seconds of clear speech works best."}),
                   "ref_text": required.pop("ref_text")}
        ordered.update(required)
        return {"required": ordered,
                "optional": {"whisper_model": ("WHISPER_ASR",)}}

    def generate(self, omnivoice_model, text, ref_audio, ref_text, steps,
                 guidance_scale, t_shift, speed, duration, seed,
                 position_temperature, class_temperature, layer_penalty_factor,
                 denoise, preprocess_prompt, postprocess_output, instruct,
                 whisper_model=None):
        with _INFERENCE_LOCK:
            _check_interrupt()
            if not text.strip():
                raise ValueError("Text cannot be empty.")
            model = _ensure_model(omnivoice_model)
            result = None
            try:
                pbar = _progress(4)
                reference = _comfy_audio_to_numpy(ref_audio)
                effective_text = ref_text.strip()
                if not effective_text:
                    reference = _prepare_auto_reference(reference, preprocess_prompt)
                    effective_text = _transcribe_reference(
                        reference, whisper_model, omnivoice_model)
                _log_reference_duration(reference)
                ref_tensor = torch.from_numpy(reference).float()
                _update_progress(pbar, 1, 4)
                _manual_seed_all(_actual_seed(seed))
                _update_progress(pbar, 2, 4)
                _check_interrupt()
                kwargs = _generate_kwargs(
                    text, steps, guidance_scale, t_shift, speed,
                    position_temperature, class_temperature, layer_penalty_factor,
                    denoise, preprocess_prompt, postprocess_output, duration, instruct,
                )
                kwargs["ref_audio"] = (ref_tensor, OMNIVOICE_SAMPLE_RATE)
                if effective_text:
                    kwargs["ref_text"] = effective_text
                result = _numpy_to_audio(_to_numpy_audio(_call_generate(model, kwargs)[0]))
                _update_progress(pbar, 4, 4)
            finally:
                _release_bundle(omnivoice_model)
        if result is None:
            raise RuntimeError("Generation failed; see the ComfyUI log.")
        return (result,)


class OmniVoiceLongformTTS:
    CATEGORY = CATEGORY
    TITLE = "OmniVoice Longform TTS ⚡"
    SEARCH_ALIASES = ["omnivoice", "long form tts", "text to speech", "voice clone", "multilingual speech"]
    DESCRIPTION = ("Long-form multilingual TTS with sentence-aware chunking, optional voice "
                   "cloning, and first-chunk voice consistency for automatic voices.")
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        required = {"omnivoice_model": ("OMNIVOICE_MODEL",)}
        required.update(_generation_widgets(True))
        return {"required": required, "optional": {
            "ref_audio": ("AUDIO",),
            "whisper_model": ("WHISPER_ASR",),
        }}

    def generate(self, omnivoice_model, text, ref_text, steps,
                 guidance_scale, t_shift, speed, duration, seed,
                 position_temperature, class_temperature, layer_penalty_factor,
                 denoise, preprocess_prompt, postprocess_output, instruct,
                 words_per_chunk, ref_audio=None, whisper_model=None):
        with _INFERENCE_LOCK:
            _check_interrupt()
            if not text.strip():
                raise ValueError("Text cannot be empty.")
            model = _ensure_model(omnivoice_model)
            result = None
            try:
                _manual_seed_all(_actual_seed(seed))
                use_clone = ref_audio is not None
                ref_tensor = None
                effective_text = ref_text.strip()
                if use_clone:
                    reference = _comfy_audio_to_numpy(ref_audio)
                    if not effective_text:
                        reference = _prepare_auto_reference(reference, preprocess_prompt)
                        effective_text = _transcribe_reference(
                            reference, whisper_model, omnivoice_model)
                    _log_reference_duration(reference)
                    ref_tensor = torch.from_numpy(reference).float()

                chunks = _smart_chunk_text(text, words_per_chunk)
                pbar = _progress(len(chunks) + 1)
                _update_progress(pbar, 1, len(chunks) + 1)
                audio_chunks = []
                auto_reference = None
                first_chunk_text = ""
                for index, chunk in enumerate(chunks):
                    _check_interrupt()
                    kwargs = _generate_kwargs(
                        chunk, steps, guidance_scale, t_shift, speed,
                        position_temperature, class_temperature, layer_penalty_factor,
                        denoise, preprocess_prompt, postprocess_output, duration, instruct,
                    )
                    if use_clone:
                        kwargs["ref_audio"] = (ref_tensor, OMNIVOICE_SAMPLE_RATE)
                        if effective_text:
                            kwargs["ref_text"] = effective_text
                    elif index > 0 and auto_reference is not None:
                        kwargs["ref_audio"] = (auto_reference, OMNIVOICE_SAMPLE_RATE)
                        kwargs["ref_text"] = first_chunk_text
                    audio = _to_numpy_audio(_call_generate(model, kwargs)[0])
                    audio_chunks.append(audio)
                    if not use_clone and index == 0 and len(chunks) > 1:
                        auto_reference = torch.from_numpy(
                            audio[:25 * OMNIVOICE_SAMPLE_RATE]
                        ).float()
                        first_chunk_text = chunk
                    _update_progress(pbar, index + 2, len(chunks) + 1)
                result = _numpy_to_audio(audio_chunks[0] if len(audio_chunks) == 1
                                         else np.concatenate(audio_chunks))
            finally:
                _release_bundle(omnivoice_model)
        if result is None:
            raise RuntimeError("Generation failed; see the ComfyUI log.")
        return (result,)


NODE_CLASS_MAPPINGS = {
    "CCTechOmniVoiceModelsLoader": OmniVoiceModelsLoader,
    "CCTechOmniVoiceLongformTTS": OmniVoiceLongformTTS,
    "CCTechOmniVoiceVoiceCloneTTS": OmniVoiceVoiceCloneTTS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CCTechOmniVoiceModelsLoader": "OmniVoice Models Loader ⚡",
    "CCTechOmniVoiceLongformTTS": "OmniVoice Longform TTS ⚡",
    "CCTechOmniVoiceVoiceCloneTTS": "OmniVoice Voice Clone TTS ⚡",
}
