from .nodes import XZImageToText, XZLlmResponse

NODE_CLASS_MAPPINGS = {
    "XZImageToText": XZImageToText,
    "XZLlmResponse": XZLlmResponse,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XZImageToText": "XZ Image To Text",
    "XZLlmResponse": "XZ LLM Response",
}
