import os
import time

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


# Order of attempts: every cloud for the first GPU, then every cloud for the
# next GPU, and so on. GPU wins over cloud.
GPU_PREFERENCE = [
    # "NVIDIA A100-SXM4-80GB",
    # "NVIDIA A100 80GB PCIe",
    # "NVIDIA A40",
    "NVIDIA RTX 2000 Ada Generation"
]
CLOUD_PREFERENCE = ["COMMUNITY", "SECURE"]
USABLE = {"LOW", "MEDIUM", "HIGH"}  # anything but NONE

VLLM_IMAGE = "vllm/vllm-openai:latest"
VLLM_ARGS = (
    "Qwen/Qwen3-4B "
    "--max-model-len 4000 "
    # "--model openai/gpt-oss-120b "
    # "--max-model-len 30000 "
    "--gpu-memory-utilization 0.95 "
    # "--kv-offloading-size 50 "
    # "--kv-offloading-backend native "
)
POD_ENV = {
    "HF_TOKEN": HF_TOKEN,
    "VLLM_API_KEY": VLLM_API_KEY,
    "HF_XET_HIGH_PERFORMANCE": "1",
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
        "disk": 100,
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
    last_detail = None

    for round_no in range(1, rounds + 1):
        candidate_list = candidates(region)
        if not candidate_list:
            print(f"[{round_no}/{rounds}] catalog reports no usable GPUs")
        
        for gpu_id, datacenter_id, cloud in candidate_list:
            where = datacenter_id if cloud == "SECURE" else "any"
            response = create_with_backoff(gpu_id, datacenter_id, cloud)

            if response.status_code == 201:
                return response.json()

            last_detail = response.json().get("detail", response.text[:120])

            if response.status_code in (400, 403):
                print(f"[{round_no}/{rounds}] {gpu_id} / {cloud} / {where}: no capacity")
                continue
            if response.status_code == 402:
                raise SystemExit(f"insufficient balance: {last_detail}")
            if response.status_code == 422:
                raise SystemExit(f"bad request: {last_detail}")
            response.raise_for_status()

        if round_no < rounds:
            time.sleep(5)

    raise SystemExit(f"no pod placed after {rounds} rounds. last error: {last_detail}")


def wait_running(pod_id, timeout_s=900):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        pod = SESSION.get(f"{API}/v2/pods/{pod_id}", timeout=30).json()
        if pod["status"] == "RUNNING":
            return pod
        if pod["status"] in ("EXITED", "ERROR", "TERMINATED"):
            raise SystemExit(f"pod died: {pod['status']}")
        time.sleep(10)
    raise SystemExit("pod never reached RUNNING")


def wait_model_loaded(pod_id, timeout_s=1200):
    url = f"https://{pod_id}-8000.proxy.runpod.net/v1/models"
    headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                return url.rsplit("/v1", 1)[0]
            if r.status_code == 401:
                raise SystemExit("vLLM rejected the api key")
        except requests.RequestException:
            pass
        time.sleep(15)
    raise SystemExit("model never loaded")


if __name__ == "__main__":
    pod = deploy()
    print(f"pod {pod['id']} placed, waiting...")
    wait_running(pod["id"])
    base_url = wait_model_loaded(pod["id"])
    print(f"\nready")
    print(f"base url : {base_url}")
    print(f"api key  : {VLLM_API_KEY}")