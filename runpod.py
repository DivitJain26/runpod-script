import os
import secrets
import time

import requests
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN", "")

if not RUNPOD_API_KEY:
    raise RuntimeError("RUNPOD_API_KEY is not set")

API = "https://api.runpod.io"
SESSION = requests.Session()
SESSION.headers["Authorization"] = f"Bearer {RUNPOD_API_KEY}"

# Fresh per run. Anyone with the proxy URL but not this key gets a 401.
VLLM_API_KEY = secrets.token_urlsafe(32)

# Order of attempts: every cloud for the first GPU, then every cloud for the
# next GPU, and so on. GPU wins over cloud.
GPU_PREFERENCE = [
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA A100 80GB PCIe",
    # "NVIDIA A40",
]
CLOUD_PREFERENCE = ["COMMUNITY", "SECURE"]
USABLE = {"LOW", "MEDIUM", "HIGH"}  # anything but NONE

VLLM_IMAGE = "vllm/vllm-openai:latest"
VLLM_ARGS = (
    "--model openai/gpt-oss-120b "
    "--max-model-len 30000 "
    "--gpu-memory-utilization 0.95 "
    "--kv-offloading-size 50 "
    "--kv-offloading-backend native "
)
POD_ENV = {
    "HF_TOKEN": HF_TOKEN,
    "VLLM_API_KEY": VLLM_API_KEY,
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
}


def candidates(region):
    response = SESSION.get(
        f"{API}/v2/catalog/datacenters",
        params={"regions": region, "include": "GPU_AVAILABILITY"},
        timeout=30,
    )
    response.raise_for_status()

    rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    found = []
    for datacenter in response.json()["dataCenters"]:
        for gpu in datacenter.get("gpuAvailability", []):
            if gpu["id"] in GPU_PREFERENCE and gpu["availability"] in USABLE:
                for cloud in CLOUD_PREFERENCE:
                    if cloud == "COMMUNITY" and any(
                        c[0] == gpu["id"] and c[2] == "COMMUNITY" for c in found
                    ):
                        continue  # already have one Community attempt for this GPU
                    found.append((gpu["id"], datacenter["id"], cloud, gpu["availability"]))

    found.sort(
        key=lambda c: (
            GPU_PREFERENCE.index(c[0]),
            CLOUD_PREFERENCE.index(c[2]),
            rank[c[3]],
        )
    )
    return [(gpu_id, dc_id, cloud) for gpu_id, dc_id, cloud, _ in found]


def create(gpu_id, datacenter_id, cloud):
    body = {
        "name": "metroleads-inference",
        "image": VLLM_IMAGE,
        "cloud": cloud,
        "gpu": {"id": gpu_id, "count": 1},
        "disk": 120,
        "ports": ["8000/http"],
        "env": POD_ENV,
        "args": VLLM_ARGS,
    }
    if cloud == "SECURE":
        body["dataCenterIds"] = [datacenter_id]
    return SESSION.post(f"{API}/v2/pods", json=body, timeout=60)


def create_with_backoff(gpu_id, datacenter_id, cloud, attempts=5):
    for attempt in range(attempts):
        response = create(gpu_id, datacenter_id, cloud)

        if response.status_code == 429:
            time.sleep(int(response.headers.get("Retry-After", 5)))
            continue

        if response.status_code >= 500:
            time.sleep(2 ** attempt)
            continue

        return response

    raise RuntimeError(
        f"{attempts} transient failures for {gpu_id}/{cloud}; upstream unhealthy"
    )


def deploy(region=None, rounds=5):
    for round_no in range(1, rounds + 1):
        print(f"\n=== Placement attempt {round_no}/{rounds} ===")

        candidate_list = candidates(region)
        print(f"Found {len(candidate_list)} candidates")

        for gpu_id, datacenter_id, cloud in candidate_list:
            print(f"Trying {gpu_id} / {datacenter_id} / {cloud}...")

            response = create_with_backoff(gpu_id, datacenter_id, cloud)

            print(f"  -> HTTP {response.status_code}: {response.text[:300]}")

            if response.status_code == 201:
                return response.json()

            if response.status_code in (400, 403):
                # 400: no capacity here right now. 403: no access to this pool.
                continue

            if response.status_code == 402:
                raise SystemExit(f"Cannot deploy: {response.text}")

            if response.status_code == 422:
                raise SystemExit(f"Bad request: {response.text}")

            response.raise_for_status()

        print("Nothing placed. Refreshing catalog...")
        time.sleep(5)

    raise SystemExit(f"Could not place a Pod after {rounds} attempts.")


if __name__ == "__main__":
    pod = deploy()
    print(f"\n{pod['id']} placed in {pod['dataCenterId']} on {pod['gpu']['id']} ({pod['cloud']})")
    print(f"base url : https://{pod['id']}-8000.proxy.runpod.net")
    print(f"api key  : {VLLM_API_KEY}")