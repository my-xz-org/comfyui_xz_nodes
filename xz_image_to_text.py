import base64
import io
import json
import urllib.error
import urllib.request

from PIL import Image
import torch

MAX_SEED = 0xFFFFFFFFFFFFFFFF


class XZImageToText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": MAX_SEED,
                        "control_after_generate": True,
                    },
                ),
                "base_url": ("STRING", {"default": "https://api.openai.com/v1"}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model_id": ("STRING", {"default": "gpt-4o-mini"}),
                "temperature": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "max_tokens": (
                    "INT",
                    {"default": 10240, "min": 1, "max": 262144, "step": 1},
                ),
                "presence_penalty": (
                    "FLOAT",
                    {"default": 1.5, "min": -2.0, "max": 2.0, "step": 0.1},
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "default": "You are a helpful image captioning assistant.",
                        "multiline": True,
                    },
                ),
                "user_prompt": (
                    "STRING",
                    {"default": "Describe the image.", "multiline": True},
                ),
            },
            "optional": {
                "top_k": (
                    "INT",
                    {"default": 20, "min": 0, "max": 4096, "step": 1},
                ),
                "min_p": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "run"
    CATEGORY = "xzinfra/api"

    def run(
        self,
        image,
        seed,
        base_url,
        api_key,
        model_id,
        temperature,
        top_p,
        max_tokens,
        presence_penalty,
        system_prompt,
        user_prompt,
        top_k=None,
        min_p=None,
        repetition_penalty=None,
    ):
        if not api_key:
            raise ValueError("api_key is required.")
        if not model_id:
            raise ValueError("model_id is required.")
        if not user_prompt:
            raise ValueError("user_prompt is required.")

        endpoint = base_url.rstrip("/") + "/chat/completions"
        captions = []
        seed_value = int(seed)

        for img in image:
            data_url = self._image_to_data_url(img)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            )
            payload = {
                "model": model_id,
                "seed": seed_value,
                "temperature": float(temperature),
                "top_p": float(top_p),
                "max_tokens": int(max_tokens),
                "presence_penalty": float(presence_penalty),
                "messages": messages,
            }
            if top_k is not None:
                payload["top_k"] = int(top_k)
            if min_p is not None:
                payload["min_p"] = float(min_p)
            if repetition_penalty is not None:
                payload["repetition_penalty"] = float(repetition_penalty)
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            body = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                endpoint,
                data=body,
                headers=headers,
                method="POST",
            )

            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    response_body = response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="ignore")
                raise RuntimeError(
                    f"OpenAI request failed: HTTP {exc.code} {exc.reason} {error_body}"
                ) from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"OpenAI request failed: {exc}") from exc

            data = json.loads(response_body)
            captions.append(self._extract_content(data))

        if len(captions) == 1:
            return (captions[0],)
        return (captions,)

    @staticmethod
    def _image_to_data_url(img_tensor):
        img = img_tensor.detach().cpu().clamp(0, 1)
        img = (img * 255).to(torch.uint8).numpy()
        pil = Image.fromarray(img)
        buffer = io.BytesIO()
        pil.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def _extract_content(data):
        choices = data.get("choices")
        if not choices:
            raise RuntimeError(f"OpenAI response missing choices: {data}")
        message = choices[0].get("message", {})
        content = message.get("content")
        if content is None:
            raise RuntimeError(f"OpenAI response missing message content: {data}")
        return content
