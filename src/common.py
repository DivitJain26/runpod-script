import os

import requests
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
VLLM_API_KEY = os.getenv("VLLM_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN", "")

if not RUNPOD_API_KEY:
    raise RuntimeError("RUNPOD_API_KEY is not set")
if not VLLM_API_KEY:
    raise RuntimeError("VLLM_API_KEY is not set")

API = "https://api.runpod.io"
SESSION = requests.Session()
SESSION.headers["Authorization"] = f"Bearer {RUNPOD_API_KEY}"

POD_NAME = "metroleads-inference"
VLLM_PORT = 8000


def pod_base_url(pod_id):
    """Build the public proxy URL for a pod id."""
    return f"https://{pod_id}-{VLLM_PORT}.proxy.runpod.net"