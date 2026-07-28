"""Vision-Language Model clients for VLFM.

Aligned with the upstream `rai-opensource/vlfm <https://github.com/rai-opensource/vlfm>`_
VLM interface pattern.  The upstream uses BLIP2 ITM (Image-Text Matching)
which returns a single cosine-similarity score per image.  We provide
pluggable backends that follow the same contract — ``score_image(image, text)
→ float`` — so the value map receives one scalar per observation.

Backends
--------
- :class:`BLIP2ITMVLMClient` — local BLIP-2 ITM via LAVIS (the original upstream backend)
- :class:`CLIPVLMClient` — local CLIP ViT-B/32 via HuggingFace transformers
- :class:`SigLIPVLMClient` — local SigLIP via HuggingFace transformers
- :class:`OpenAIVLMClient` — cloud GPT-4o (or compatible) API
"""

from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------


@dataclass
class VLMDirectionResult:
    """Result of scoring N directional images against a text instruction."""

    chosen_index: int
    scores: list[float] = field(default_factory=list)
    confidence: float = 1.0
    reasoning: str = ""
    raw_response: str = ""


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class VLMClient(ABC):
    """Abstract interface for a vision-language scorer.

    Subclass this and implement :meth:`score_image` (and optionally
    :meth:`score_directions`) to add a new backend.
    """

    supports_text_only: bool = False

    def validate_environment(self) -> None:
        """Raise when required runtime dependencies are unavailable."""

    def prepare(self) -> None:
        """Load local model resources before camera/render resources are opened."""

    @abstractmethod
    def score_image(self, image: bytes, text: str) -> float:
        """Score a single image against a text prompt.

        This is the core ITM (Image-Text Matching) primitive.  The return
        value is a scalar cosine similarity (or analogous score) where
        **higher is better**.  The VLFM value map uses this per-frame.

        Parameters
        ----------
        image:
            PNG/JPEG image bytes.
        text:
            Natural-language instruction (e.g. "Seems like there is a chair ahead.").

        Returns
        -------
        float
            Calibrated or normalised relevance score in ``[0, 1]``.
        """
        ...

    def score_directions(
        self,
        images: list[bytes],
        query_text: str,
    ) -> VLMDirectionResult:
        """Score each of N images against *query_text*.

        The default implementation scores each image independently via
        :meth:`score_image`.  Backends that support batched inference
        should override this for efficiency.

        Parameters
        ----------
        images:
            List of PNG/JPEG bytes, one per candidate direction.
        query_text:
            Natural-language instruction describing what to look for.

        Returns
        -------
        VLMDirectionResult
            Chosen index, per-direction scores, and optional metadata.
        """
        if not images:
            raise ValueError("At least one image is required")
        scores = [self.score_image(img, query_text) for img in images]
        chosen = int(np.argmax(scores))
        return VLMDirectionResult(
            chosen_index=chosen,
            scores=scores,
            confidence=float(np.max(scores)),
            raw_response=f"scores: {scores}",
        )

    @staticmethod
    def encode_image(image_bytes: bytes, fmt: str = "png") -> str:
        """Encode raw image bytes as a base64 data-URI string."""
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/{fmt};base64,{b64}"


# ---------------------------------------------------------------------------
# CLIP — local model, no API calls
# ---------------------------------------------------------------------------


def _load_model_with_fallback(model_cls, model_name: str, device: str):
    """Load a HF model with fallback for offline environments.

    1. Normal ``from_pretrained`` (needs network to verify revision).
    2. ``local_files_only=True`` (uses cached files without network).
    3. Via hf-mirror.com.
    """
    import os as _os

    # Attempt 1: normal
    try:
        return model_cls.from_pretrained(model_name).to(device)
    except Exception:
        pass

    # Attempt 2: local files only
    try:
        return model_cls.from_pretrained(model_name, local_files_only=True).to(device)
    except Exception:
        pass

    # Attempt 3: via mirror
    _prev = _os.environ.get("HF_ENDPOINT", "")
    _os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    try:
        return model_cls.from_pretrained(model_name).to(device)
    finally:
        if _prev:
            _os.environ["HF_ENDPOINT"] = _prev
        else:
            _os.environ.pop("HF_ENDPOINT", None)


def _load_processor_via_mirror(processor_cls, model_name: str):
    """Load a processor via the HF mirror."""
    import os as _os

    _prev = _os.environ.get("HF_ENDPOINT", "")
    _os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    try:
        return processor_cls.from_pretrained(model_name)
    finally:
        if _prev:
            _os.environ["HF_ENDPOINT"] = _prev
        else:
            _os.environ.pop("HF_ENDPOINT", None)


def _load_processor_with_fallback(processor_cls, model_name: str):
    """Load a Hugging Face processor from the network, cache, or mirror."""
    errors = []
    try:
        return processor_cls.from_pretrained(model_name)
    except Exception as exc:
        errors.append(exc)

    try:
        return processor_cls.from_pretrained(model_name, local_files_only=True)
    except Exception as exc:
        errors.append(exc)

    try:
        return _load_processor_via_mirror(processor_cls, model_name)
    except Exception as exc:
        errors.append(exc)

    raise OSError(
        f"Cannot load processor for '{model_name}'. The cached processor/tokenizer "
        "files are incomplete or unavailable. Run once without HF_HUB_OFFLINE=1 "
        "to download the full processor snapshot."
    ) from errors[-1]


def _load_tokenizer_with_fallback(tokenizer_cls, model_name: str):
    """Load a tokenizer with fallback for offline environments."""
    import os as _os

    # Attempt 1: local files only
    try:
        return tokenizer_cls.from_pretrained(model_name, local_files_only=True)
    except Exception:
        pass

    # Attempt 2: via mirror
    _prev = _os.environ.get("HF_ENDPOINT", "")
    _os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    try:
        return tokenizer_cls.from_pretrained(model_name)
    except Exception:
        pass
    finally:
        if _prev:
            _os.environ["HF_ENDPOINT"] = _prev
        else:
            _os.environ.pop("HF_ENDPOINT", None)

    raise OSError(
        f"Cannot load tokenizer for '{model_name}' — network unavailable "
        f"and files are not cached locally.\n"
        f"Download with:\n"
        f"  export HF_ENDPOINT=https://hf-mirror.com\n"
        f"  python -c \"from transformers import AutoProcessor; "
        f"AutoProcessor.from_pretrained('{model_name}')\""
    )


# ---------------------------------------------------------------------------
# BLIP-2 ITM — the original upstream VLFM backend
# ---------------------------------------------------------------------------


class BLIP2ITMVLMClient(VLMClient):
    """HTTP client for an isolated BLIP-2 ITM service.

    LAVIS pins an old Transformers release that conflicts with SigLIP, so the
    model runs in a separate environment. The service maps raw ITC cosine
    similarity from ``[-1, 1]`` into the planner's ``[0, 1]`` score contract.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:12182",
        timeout: float = 30.0,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and > 0")
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        # BLIP2 normally runs on localhost or a trusted LAN host. Bypass
        # process-wide HTTP proxies so loopback requests cannot be intercepted.
        self._opener = urllib_request.build_opener(urllib_request.ProxyHandler({}))

    def validate_environment(self) -> None:
        """The main environment needs no BLIP2/LAVIS dependencies."""

    def prepare(self) -> None:
        health = self._request_json("GET", "/health", timeout=min(self.timeout, 5.0))
        if health.get("status") != "ready":
            raise RuntimeError(f"BLIP2 ITM service is not ready: {health}")

    def score_image(self, image: bytes, text: str) -> float:
        if not isinstance(image, (bytes, bytearray)) or not image:
            raise ValueError("image must contain encoded image bytes")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        result = self._request_json(
            "POST",
            "/score",
            payload={
                "image_b64": base64.b64encode(image).decode("ascii"),
                "text": text.strip(),
            },
        )
        score = float(result["score"])
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise RuntimeError(f"BLIP2 service returned invalid score: {score}")
        return score

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib_request.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with self._opener.open(request, timeout=timeout or self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"BLIP2 service HTTP {exc.code} at {path}: {detail}"
            ) from exc
        except (urllib_error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                f"Cannot reach BLIP2 ITM service at {self.base_url}: {exc}"
            ) from exc
        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"BLIP2 service returned invalid JSON: {body[:200]}") from exc
        if not isinstance(result, dict):
            raise RuntimeError("BLIP2 service response must be a JSON object")
        return result


# ---------------------------------------------------------------------------
# CLIP — local model, no API calls
# ---------------------------------------------------------------------------


class CLIPVLMClient(VLMClient):
    """Score images with a local CLIP model via HuggingFace transformers.

    Uses ``openai/clip-vit-base-patch32`` by default.

    Requires: ``pip install transformers torch pillow``
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        *,
        device: str = "cpu",
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string")
        self.model_name = model_name
        self.device = device
        self._model: Optional[object] = None
        self._processor: Optional[object] = None

    def validate_environment(self) -> None:
        try:
            import torch
            import transformers  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError as exc:
            raise ImportError("CLIP requires transformers, torch, and pillow") from exc
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CLIP requested {self.device}, but CUDA is unavailable")

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import CLIPModel
        except ImportError as exc:
            raise ImportError(
                "CLIP requires 'transformers'.  Install with: "
                "pip install transformers torch pillow"
            ) from exc

        self._model = _load_model_with_fallback(CLIPModel, self.model_name, self.device)
        self._processor = self._load_clip_processor()
        self._model.eval()

    def _load_clip_processor(self):
        """Load CLIPProcessor with fallback for offline environments."""
        from transformers import CLIPProcessor, CLIPImageProcessor, CLIPTokenizer

        # Attempt 1: normal loading
        try:
            return CLIPProcessor.from_pretrained(self.model_name)
        except Exception:
            pass

        # Attempt 2: via hf-mirror.com
        try:
            return _load_processor_via_mirror(CLIPProcessor, self.model_name)
        except Exception:
            pass

        # Attempt 3: construct image processor from defaults
        image_processor = CLIPImageProcessor(
            do_resize=True,
            size={"shortest_edge": 224},
            do_center_crop=True,
            crop_size={"height": 224, "width": 224},
            do_rescale=True,
            rescale_factor=1.0 / 255.0,
            do_normalize=True,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
        )

        tokenizer = _load_tokenizer_with_fallback(CLIPTokenizer, self.model_name)

        return CLIPProcessor(image_processor=image_processor, tokenizer=tokenizer)

    def prepare(self) -> None:
        self._lazy_load()

    # ------------------------------------------------------------------
    def score_image(self, image: bytes, text: str) -> float:
        """CLIP cosine similarity mapped from ``[-1, 1]`` to ``[0, 1]``."""
        self._lazy_load()
        import torch
        from PIL import Image
        from io import BytesIO

        pil = Image.open(BytesIO(image)).convert("RGB")
        inputs = self._processor(
            text=[text],
            images=[pil],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            image_features = torch.nn.functional.normalize(outputs.image_embeds, dim=-1)
            text_features = torch.nn.functional.normalize(outputs.text_embeds, dim=-1)
            cosine = torch.sum(image_features[0] * text_features[0])
            score = float(((cosine.clamp(-1.0, 1.0) + 1.0) / 2.0).cpu())
        return score

    def score_directions(
        self,
        images: list[bytes],
        query_text: str,
    ) -> VLMDirectionResult:
        """Batched CLIP scoring for efficiency."""
        self._lazy_load()
        import torch
        from PIL import Image
        from io import BytesIO

        if not images:
            raise ValueError("CLIP requires at least one image")

        pil_images = [Image.open(BytesIO(img)).convert("RGB") for img in images]
        inputs = self._processor(
            text=[query_text],
            images=pil_images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            image_features = torch.nn.functional.normalize(outputs.image_embeds, dim=-1)
            text_features = torch.nn.functional.normalize(outputs.text_embeds, dim=-1)
            cosine = text_features[0] @ image_features.T
            scores = ((cosine.clamp(-1.0, 1.0) + 1.0) / 2.0).cpu().numpy()

        chosen = int(np.argmax(scores))
        return VLMDirectionResult(
            chosen_index=chosen,
            scores=scores.tolist(),
            confidence=float(np.max(scores)),
            reasoning=f"[CLIP] direction {chosen} best matches '{query_text[:60]}...'",
            raw_response=f"scores: {scores.tolist()}",
        )


# ---------------------------------------------------------------------------
# SigLIP — stronger local model
# ---------------------------------------------------------------------------


class SigLIPVLMClient(VLMClient):
    """Score images with a local SigLIP model via HuggingFace transformers.

    Uses ``google/siglip-base-patch16-224`` by default.

    Requires: ``pip install transformers torch pillow``
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        *,
        device: str = "cpu",
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string")
        self.model_name = model_name
        self.device = device
        self._model: Optional[object] = None
        self._processor: Optional[object] = None

    def validate_environment(self) -> None:
        try:
            import torch
            import transformers  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError as exc:
            raise ImportError("SigLIP requires transformers, torch, and pillow") from exc
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"SigLIP requested {self.device}, but CUDA is unavailable")

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "SigLIP requires 'transformers'. Install with: "
                "pip install transformers torch pillow"
            ) from exc

        self._model = _load_model_with_fallback(
            AutoModel,
            self.model_name,
            self.device,
        )
        self._processor = _load_processor_with_fallback(
            AutoProcessor,
            self.model_name,
        )
        self._model.eval()

    def prepare(self) -> None:
        self._lazy_load()

    # ------------------------------------------------------------------
    def score_image(self, image: bytes, text: str) -> float:
        """SigLIP image-text probability in ``[0, 1]``."""
        self._lazy_load()
        import torch
        from PIL import Image
        from io import BytesIO

        pil = Image.open(BytesIO(image)).convert("RGB")
        inputs = self._processor(
            text=[text],
            images=[pil],
            return_tensors="pt",
            padding="max_length",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits_per_text  # (1, 1)
            score = float(torch.sigmoid(logits[0, 0]).cpu())
        return score

    def score_directions(
        self,
        images: list[bytes],
        query_text: str,
    ) -> VLMDirectionResult:
        """Batched SigLIP scoring for efficiency."""
        self._lazy_load()
        import torch
        from PIL import Image
        from io import BytesIO

        if not images:
            raise ValueError("SigLIP requires at least one image")

        pil_images = [Image.open(BytesIO(img)).convert("RGB") for img in images]
        inputs = self._processor(
            text=[query_text],
            images=pil_images,
            return_tensors="pt",
            padding="max_length",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits_per_text  # (1, N)
            scores = torch.sigmoid(logits[0]).cpu().numpy()

        chosen = int(np.argmax(scores))
        return VLMDirectionResult(
            chosen_index=chosen,
            scores=scores.tolist(),
            confidence=float(np.max(scores)),
            reasoning=f"[SigLIP] direction {chosen} best matches '{query_text[:60]}...'",
            raw_response=f"scores: {scores.tolist()}",
        )


# ---------------------------------------------------------------------------
# OpenAI GPT-4o — cloud API
# ---------------------------------------------------------------------------


class OpenAIVLMClient(VLMClient):
    """Score images with OpenAI GPT-4o (or compatible) API.

    Supports both :meth:`score_image` (with image and text) and
    :meth:`score_directions` (with multiple images).  Text-only
    direction scoring is available when no images are provided.

    Requires: ``pip install openai``
    """

    supports_text_only = True

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str = "",
        base_url: Optional[str] = None,
        max_tokens: int = 300,
        temperature: float = 0.3,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not np.isfinite(temperature) or not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature

    def validate_environment(self) -> None:
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise ImportError("OpenAI backend requires the openai package") from exc
        if not self.api_key:
            raise ValueError("OpenAI backend requires a non-empty API key")

    def score_image(self, image: bytes, text: str) -> float:
        """Ask GPT-4o to rate image-text relevance on a 0–100 scale."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI client requires 'openai'.  Install with: pip install openai"
            ) from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        data_uri = self.encode_image(image)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a robot navigation relevance scorer. "
                    "Given an image and a text instruction, rate how "
                    "relevant the image is to the instruction on a 0-100 "
                    "integer scale. Reply with ONLY the number."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Instruction: {text}\nRelevance score (0-100):"},
                    {"type": "image_url", "image_url": {"url": data_uri, "detail": "low"}},
                ],
            },
        ]

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=10,
            temperature=0.0,
        )
        raw = response.choices[0].message.content or "0"

        import re

        nums = re.findall(r"\d+", raw)
        score = float(nums[0]) if nums else 0.0
        return score / 100.0  # normalise to [0, 1]

    def score_directions(
        self,
        images: list[bytes],
        query_text: str,
    ) -> VLMDirectionResult:
        """Multi-image direction scoring with GPT-4o."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI client requires 'openai'.  Install with: pip install openai"
            ) from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        content: list[dict] = [{"type": "text", "text": query_text}]
        for img_bytes in images:
            data_uri = self.encode_image(img_bytes)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri, "detail": "low"},
                }
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a robot navigation assistant. Given an instruction "
                    "and images from different directions, pick the direction "
                    "most likely to lead toward the goal. "
                    "Always respond with a direction number first."
                ),
            },
            {"role": "user", "content": content},
        ]

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        raw = response.choices[0].message.content or ""

        import re

        candidate_count = max(len(images), 1)
        range_match = re.search(r"0-(\d+)", query_text)
        if range_match:
            candidate_count = max(candidate_count, int(range_match.group(1)) + 1)

        chosen = 0
        for match in re.finditer(r"\b(\d+)\b", raw):
            idx = int(match.group(1))
            if 0 <= idx < candidate_count:
                chosen = idx
                break

        return VLMDirectionResult(
            chosen_index=chosen,
            reasoning=raw,
            raw_response=raw,
        )
