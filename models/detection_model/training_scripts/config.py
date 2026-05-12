# Hugging Face token for authentication
# Get your token from: https://huggingface.co/settings/tokens
HF_TOKEN = "your_hugging_face_token"

# Model path (Hugging Face model ID or local path).
# Unsloth: either an unsloth/...-bnb-4bit id or official meta-llama/... (gated; needs HF_TOKEN).
MODEL_PATH = "unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit"

# Use Unsloth FastLanguageModel + get_peft_model (import unsloth before transformers; pip install unsloth unsloth_zoo).
USE_UNSLOTH = True
# None = auto dtype (Unsloth picks a stable dtype for your GPU)
UNSLOTH_DTYPE = None

