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
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "run"
    CATEGORY = "xzinfra/api"

    def run(self, seed, base_url, api_key, model_id, system_prompt, user_message):
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
            "messages": messages,
        }
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
