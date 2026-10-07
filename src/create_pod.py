import time

from common import API, HF_TOKEN, POD_NAME, SESSION, VLLM_API_KEY, VLLM_PORT, pod_base_url
from get_pod import get_by_id, is_model_loaded
from terminate_pod import terminate


# Order of attempts: every cloud for the first GPU, then every cloud for the
# next GPU, and so on. GPU wins over cloud.
GPU_PREFERENCE = [
    # "NVIDIA A100-SXM4-80GB",
    # "NVIDIA A100 80GB PCIe",
    # "NVIDIA A40",
    "NVIDIA RTX A5000"
    # "NVIDIA RTX 2000 Ada Generation"
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

ALLOCATE_DEADLINE_S = 2 * 60 * 60   # keep hunting for a GPU this long
ALLOCATE_RETRY_S = 30               # pause between catalog re-reads
MAX_POD_ATTEMPTS = 3                # fresh pods to try if one boots but never loads
RUNNING_TIMEOUT_S = 900
MODEL_TIMEOUT_S = 1200


class PodFailed(Exception):
    """A placed pod did not become usable; caller should terminate and retry."""


def candidates(region=None):
    """Return (gpu_id, datacenter_id, cloud) triples the catalog says are deployable, best first."""
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
                        continue  # one Community attempt per GPU is enough
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
    """Issue one POST /v2/pods for a single (gpu, dc, cloud) candidate."""
    body = {
        "name": POD_NAME,
        "image": VLLM_IMAGE,
        "cloud": cloud,
        "gpu": {"id": gpu_id, "count": 1},
        "disk": 50,
        "ports": [f"{VLLM_PORT}/http"],
        "env": POD_ENV,
        "args": VLLM_ARGS,
    }
    if cloud == "SECURE":
        body["dataCenterIds"] = [datacenter_id]
    return SESSION.post(f"{API}/v2/pods", json=body, timeout=60)


def create_with_backoff(gpu_id, datacenter_id, cloud, attempts=5):
    """Retry create() in place on 429 / 5xx, which are not the candidate's fault."""
    for attempt in range(attempts):
        response = create(gpu_id, datacenter_id, cloud)
        if response.status_code == 429:
            time.sleep(int(response.headers.get("Retry-After", 5)))
            continue
        if response.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        return response
    raise RuntimeError(f"{attempts} transient failures for {gpu_id}/{cloud}; upstream unhealthy")


def allocate_pod(region=None):
    """Walk candidates until one places, re-reading the catalog until ALLOCATE_DEADLINE_S expires."""
    deadline = time.time() + ALLOCATE_DEADLINE_S
    last_detail = None
    round_no = 0

    while time.time() < deadline:
        round_no += 1
        for gpu_id, datacenter_id, cloud in candidates(region):
            response = create_with_backoff(gpu_id, datacenter_id, cloud)

            if response.status_code == 201:
                return response.json()

            last_detail = response.json().get("detail", response.text[:120])

            if response.status_code in (400, 403):
                continue  # no capacity here right now, or no access to this pool
            if response.status_code == 402:
                raise SystemExit(f"insufficient balance: {last_detail}")
            if response.status_code == 422:
                raise SystemExit(f"bad request: {last_detail}")
            response.raise_for_status()

        print(f"[allocate] round {round_no}: nothing placed, retrying in {ALLOCATE_RETRY_S}s")
        time.sleep(ALLOCATE_RETRY_S)

    raise SystemExit(f"no pod placed within {ALLOCATE_DEADLINE_S}s. last error: {last_detail}")


def wait_running(pod_id, timeout_s=RUNNING_TIMEOUT_S):
    """Block until Runpod reports RUNNING; raise PodFailed if it dies or times out."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        pod = get_by_id(pod_id)
        if pod is None:
            raise PodFailed(f"pod {pod_id} disappeared")
        status = pod.get("status")
        if status == "RUNNING":
            return pod
        if status in ("EXITED", "ERROR", "TERMINATED"):
            raise PodFailed(f"pod {pod_id} died with status {status}")
        time.sleep(10)
    raise PodFailed(f"pod {pod_id} never reached RUNNING in {timeout_s}s")


def wait_model_loaded(pod_id, timeout_s=MODEL_TIMEOUT_S):
    """Block until vLLM answers /v1/models; raise PodFailed on timeout or bad key."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if is_model_loaded(pod_id):
                return pod_base_url(pod_id)
        except RuntimeError as exc:
            raise PodFailed(str(exc)) from exc
        time.sleep(15)
    raise PodFailed(f"model not loaded on {pod_id} in {timeout_s}s")


def provision():
    """Allocate → wait running → wait model loaded; on any failure terminate and start over."""
    for attempt in range(1, MAX_POD_ATTEMPTS + 1):
        pod = allocate_pod()
        pod_id = pod["id"]
        print(f"[provision] attempt {attempt}: {pod_id} on {pod['gpu']['id']} ({pod['cloud']})")

        try:
            wait_running(pod_id)
            base_url = wait_model_loaded(pod_id)
            return pod, base_url
        except PodFailed as exc:
            print(f"[provision] {exc}; terminating and retrying")
            terminate(pod_id)

    raise SystemExit(f"gave up after {MAX_POD_ATTEMPTS} pods failed to become ready")


if __name__ == "__main__":
    pod, base_url = provision()
    print(f"\nready: {pod['id']}  {base_url}")
