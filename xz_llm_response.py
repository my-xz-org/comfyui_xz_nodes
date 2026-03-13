import json
import urllib.error
import urllib.request

MAX_SEED = 0xFFFFFFFFFFFFFFFF


class XZLlmResponse:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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
                        "default": "You are a helpful assistant.",
                        "multiline": True,
                    },
                ),
                "user_message": (
                    "STRING",
                    {"default": "", "multiline": True},
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
    RETURN_NAMES = ("response",)
    FUNCTION = "run"
    CATEGORY = "xzinfra/api"

    def run(
        self,
        seed,
        base_url,
        api_key,
        model_id,
        temperature,
        top_p,
        max_tokens,
        presence_penalty,
        system_prompt,
        user_message,
        top_k=None,
        min_p=None,
        repetition_penalty=None,
    ):
        if not api_key:
            raise ValueError("api_key is required.")
        if not model_id:
            raise ValueError("model_id is required.")
        if not user_message:
            raise ValueError("user_message is required.")

        endpoint = base_url.rstrip("/") + "/chat/completions"
        seed_value = int(seed)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

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
                f"API request failed: HTTP {exc.code} {exc.reason} {error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"API request failed: {exc}") from exc

        data = json.loads(response_body)
        return (self._extract_content(data),)

    @staticmethod
    def _extract_content(data):
        choices = data.get("choices")
        if not choices:
            raise RuntimeError(f"API response missing choices: {data}")
        message = choices[0].get("message", {})
        content = message.get("content")
        if content is None:
            raise RuntimeError(f"API response missing message content: {data}")
        return content
