import os
import json
import time
import logging
import base64
import tempfile
import imageio
import io
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
import numpy as np
# from utils.other_utils import extract_json

# Configure logging
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """
    Abstract base class for LLM providers.

    Concrete subclasses must implement `make_request(messages, temperature)` which
    returns the raw string response from the model.
    """

    def __init__(self, model: str, api_key_env: str = None, rate_limit_seconds: float = 0.0,
                 max_retries: int = 3):
        self.model = model
        self.rate_limit_seconds = rate_limit_seconds
        self.max_retries = max_retries
        self.api_key_env = api_key_env
        self.last_usage: Dict[str, Any] = {}

    @abstractmethod
    def make_request(self, messages: List[Dict[str, Any]], temperature: float = 0.0,
                     json_mode: bool = False) -> str:
        """
        Make a request to the provider using a list of messages (role/content).
        Returns the provider's text output (string).
        """
        raise NotImplementedError

    def make_text_request(self, text: str, temperature: float = 0.0) -> str:
        """
        Convenience wrapper for single-text input.
        """
        messages = [{"role": "user", "content": text}]
        return self.make_request(messages, temperature=temperature)

    @abstractmethod
    def format_multimodal_message(self, text: str, media: str, mime: str, media_type: str, is_url: bool = False) -> List[Dict[str, Any]]:
        """
        Provider-specific formatting hook.
        - text: user text prompt
        - media: either base64 data string (no data: prefix) OR an http(s)/data URL
        - mime: "image/png" or "video/mp4" (may be None if media is a URL and unknown)
        - media_type: "image" or "video"
        - is_url: True if `media` is a URL (http/https/data:), False if `media` is base64 data
        Returns a list of message dict(s) ready to pass to make_request(...)
        """
        raise NotImplementedError

    def make_text_and_image_request(self, text: str, image: Any, temperature: float = 0.0) -> str:
        """
        Accepts `image` as either:
          - numpy.ndarray -> encoded to PNG base64 (same as before)
          - str URL starting with http://, https://, or data: -> forwarded as URL
        Delegates to provider hook with is_url flag.
        """
        # If image is a URL (http/https/data:), forward directly
        if isinstance(image, str) and (image.startswith("http://") or image.startswith("https://") or image.startswith("data:")):
            media = image
            is_url = True
            mime = None
        else:
            # lazy import Pillow
            try:
                from PIL import Image
            except ImportError:
                raise RuntimeError("Pillow is required. Install with `pip install pillow`")

            if not isinstance(image, np.ndarray):
                raise TypeError("image must be a numpy.ndarray (H x W x C) or an http(s)/data URL string")

            if image.ndim == 2:
                mode = "L"
            elif image.ndim == 3:
                c = image.shape[2]
                if c not in (1, 3, 4):
                    raise ValueError("image must have 1, 3, or 4 channels")
            else:
                raise ValueError("image must be 2D (H,W) or 3D (H,W,C)")

            pil = Image.fromarray(image.astype("uint8"))
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            media = base64.b64encode(png_bytes).decode("ascii")
            is_url = False
            mime = "image/png"

        messages = self.format_multimodal_message(text=text, media=media, mime=mime, media_type="image", is_url=is_url)
        return self.make_request(messages, temperature=temperature)

    def make_multimodal_request(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Any,
        temperature: float = 0.0,
    ) -> str:
        """Send a system-role prompt plus a user-role prompt with an attached image.

        Unlike ``make_text_and_image_request`` (which stuffs everything into a
        single user message), this method creates proper role separation so the
        model treats the system prompt with higher priority.
        """
        if isinstance(image, str) and (
            image.startswith("http://")
            or image.startswith("https://")
            or image.startswith("data:")
        ):
            media = image
            is_url = True
            mime = None
        else:
            try:
                from PIL import Image as PILImage
            except ImportError:
                raise RuntimeError("Pillow is required. Install with `pip install pillow`")
            if not isinstance(image, np.ndarray):
                raise TypeError(
                    "image must be a numpy.ndarray (H x W x C) or an http(s)/data URL string"
                )
            pil = PILImage.fromarray(image.astype("uint8"))
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            media = base64.b64encode(buf.getvalue()).decode("ascii")
            is_url = False
            mime = "image/png"

        user_messages = self.format_multimodal_message(
            text=user_prompt, media=media, mime=mime,
            media_type="image", is_url=is_url,
        )
        messages = [{"role": "system", "content": system_prompt}] + user_messages
        return self.make_request(messages, temperature=temperature)

    def make_text_and_video_request(self, text: str, video: Any, temperature: float = 0.0) -> str:
        """
        Accepts `video` as either:
          - numpy.ndarray of frames (T, H, W, C) -> encoded to mp4 base64
          - str URL starting with http://, https://, or data: -> forwarded as URL
        Delegates to provider hook with is_url flag.
        """
        if isinstance(video, str) and (video.startswith("http://") or video.startswith("https://") or video.startswith("data:")):
            media = video
            is_url = True
            mime = None
        else:
            # require numpy frames for non-URL path
            if not isinstance(video, np.ndarray):
                raise TypeError("video must be a numpy.ndarray of shape (T, H, W, C) or an http(s)/data URL string")

            if video.ndim != 4:
                raise ValueError("video must have shape (T, H, W, C)")

            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp_path = tmp.name
            tmp.close()
            try:
                with imageio.get_writer(tmp_path, format="ffmpeg", mode="I", fps=24) as writer:
                    for frame in video:
                        writer.append_data(frame.astype("uint8"))

                with open(tmp_path, "rb") as f:
                    mp4_bytes = f.read()
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

            media = base64.b64encode(mp4_bytes).decode("ascii")
            is_url = False
            mime = "video/mp4"

        messages = self.format_multimodal_message(text=text, media=media, mime=mime, media_type="video", is_url=is_url)
        return self.make_request(messages, temperature=temperature)


class OpenAIProvider(BaseLLM):
    """
    Provider using the 'openai' Python client and chat.completions.create(...) interface.
    Expects environment variable (default: OPENAI_API_KEY) to be set unless api_key_env is provided.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key_env: Optional[str] = "OPENAI_API_KEY",
                 rate_limit_seconds: float = 0.0, max_retries: int = 3, **openai_client_kwargs):
        super().__init__(model=model, api_key_env=api_key_env,
                         rate_limit_seconds=rate_limit_seconds, max_retries=max_retries)
        try:
            from openai import OpenAI  # lazy import
        except Exception as e:
            raise ImportError(
                "OpenAI client import failed. Install `openai` and ensure it's in your environment."
            ) from e

        api_key = os.environ.get(api_key_env) if api_key_env else None
        if not api_key:
            logger.warning(f"OPENAI API key env '{api_key_env}' not found. Client may still work if you pass key differently.")
        # instantiate client; allow additional kwargs if user wants to specify
        self._client = OpenAI(api_key=api_key, **openai_client_kwargs)

    def format_multimodal_message(self, text: str, media: str, mime: str, media_type: str, is_url: bool = False) -> List[Dict[str, Any]]:
        """
        Build OpenAI-style message content parts.
        If is_url is True and media is an http(s)/data URL, use that URL directly.
        Otherwise construct data:{mime};base64,{media}.
        """
        if is_url:
            data_uri_or_url = media
        else:
            data_uri_or_url = f"data:{mime};base64,{media}"

        if media_type == "image":
            content = [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": data_uri_or_url, "detail": "high"}}
            ]
        elif media_type == "video":
            content = [
                {"type": "text", "text": text},
                {"type": "video_url", "video_url": {"url": data_uri_or_url}}
            ]
        else:
            content = [{"type": "text", "text": text}]
        return [{"role": "user", "content": content}]

    def make_request(self, messages: List[Dict[str, Any]], temperature: float = 0.0,
                     json_mode: bool = False) -> str:
        """
        Simplified: always normalize SDK response to dict first, then extract content.
        Returns the assistant's content string when available, or the JSON-serialized dict otherwise.
        """
        attempt = 0
        last_exc = None
        while attempt < self.max_retries:
            attempt += 1
            if self.rate_limit_seconds:
                time.sleep(self.rate_limit_seconds)
            try:
                kwargs = dict(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                )
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = self._client.chat.completions.create(**kwargs)
                data = _normalize_sdk_response(resp)

                usage = data.get("usage") or {}
                self.last_usage = {
                    "model": data.get("model", self.model),
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                }

                # canonical OpenAI chat response path
                choices = data.get("choices") or []
                if choices:
                    first = choices[0]
                    # some SDKs return a 'message' dict under the choice
                    message = first.get("message") or {}
                    content = message.get("content")
                    if content is not None:
                        return content

                    # older/alternate responses might put text under 'text'
                    txt = first.get("text")
                    if txt is not None:
                        return txt

                # fallback: top-level text
                if "text" in data:
                    return data["text"]

                # final fallback: return JSON string of the normalized dict
                return json.dumps(data)
            except Exception as e:
                # quick heuristic: if it's a BadRequestError from openai give up immediately
                status_code = getattr(getattr(e, "response", None), "status_code", None)
                if status_code and 400 <= status_code < 500 and status_code != 429:
                    logger.error("Client error (won't retry): %s", e)
                    raise

                last_exc = e

                logger.warning("OpenAIProvider attempt %d failed: %s", attempt, e, exc_info=True)
                time.sleep(0.5 * attempt)
                continue
        raise RuntimeError(f"OpenAIProvider failed after {self.max_retries} attempts") from last_exc




class LLMFactory:
    """
    Factory for creating provider objects by name.
    Supported names: "openai".
    You can also pass a class object that inherits from BaseLLM as `provider_class`.
    """

    @staticmethod
    def create(provider_name: str = "openai", provider_class: Optional[type] = None,
               model: str = None, api_key_env: Optional[str] = None,
               rate_limit_seconds: float = 0.0, max_retries: int = 3,
               **provider_kwargs) -> BaseLLM:
        name = provider_name.lower() if provider_name else None
        model = model or "gpt-4o-mini"

        if provider_class:
            if not issubclass(provider_class, BaseLLM):
                raise TypeError("provider_class must be a subclass of BaseLLM")
            return provider_class(model=model, api_key_env=api_key_env,
                                  rate_limit_seconds=rate_limit_seconds, max_retries=max_retries, **provider_kwargs)

        if name == "openai":
            return OpenAIProvider(model=model, api_key_env=api_key_env or "OPENAI_API_KEY",
                                  rate_limit_seconds=rate_limit_seconds, max_retries=max_retries, **provider_kwargs)
        else:
            raise ValueError(f"Unknown provider_name '{provider_name}'. Supported: 'openai', or pass provider_class.")




def _normalize_sdk_response(resp: Any) -> Dict[str, Any]:
    """
    Turn an SDK response (OpenAIObject, genai object, dict, etc.) into a plain dict.
    Prefer `to_dict()` if available; otherwise use json round-trip fallback that uses __dict__.
    """
    if hasattr(resp, "to_dict") and callable(getattr(resp, "to_dict")):
        try:
            return resp.to_dict()
        except Exception as exc:
            logger.warning("to_dict() failed on %s: %s", type(resp).__name__, exc)

    if isinstance(resp, dict):
        return resp

    try:
        return dict(resp)
    except Exception as exc:
        logger.warning("dict() cast failed on %s: %s", type(resp).__name__, exc)

    try:
        return json.loads(json.dumps(resp, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception as exc:
        logger.error(
            "All normalization methods failed for %s: %s. "
            "Returning raw string wrapper.",
            type(resp).__name__, exc,
        )
        return {"raw_response": str(resp)}


if __name__ == "__main__":
    openai_llm = LLMFactory.create("openai", model="gpt-4o-mini")

    print("\n=== TEXT PROMPT TEST ===")
    try:
        text_response_oai = openai_llm.make_text_request("Say hello from OpenAI.")
        print("OpenAI response:", text_response_oai)
    except Exception as e:
        print("Text test failed:", e)

    print("\n=== IMAGE PROMPT TEST ===")
    try:
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:, :, 0] = 255

        text = "What color is this image?"
        img_response_oai = openai_llm.make_text_and_image_request(text, img)
        print("OpenAI image response:", img_response_oai)
    except Exception as e:
        print("Image test failed:", e)
