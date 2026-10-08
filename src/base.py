import os
import time

import requests

API = "https://api.runpod.io"
SESSION = requests.Session()
SESSION.headers["Authorization"] = f"Bearer {os.environ['RUNPOD_API_KEY']}"

# Most preferred GPU first. The loop stops at the first one that places.
GPU_PREFERENCE = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 5090",
    "NVIDIA H100 PCIe",
]
USABLE = {"LOW", "MEDIUM", "HIGH"}  # anything but NONE


def candidates(region):
    """(gpu_id, datacenter_id) pairs that the catalog reports as deployable,
    ordered by GPU_PREFERENCE then by descending stock."""
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
                found.append((gpu["id"], datacenter["id"], gpu["availability"]))

    found.sort(key=lambda c: (GPU_PREFERENCE.index(c[0]), rank[c[2]]))
    return [(gpu_id, dc_id) for gpu_id, dc_id, _ in found]


def create(gpu_id, datacenter_id):
    return SESSION.post(
        f"{API}/v2/pods",
        json={
            "name": "inference-worker",
            "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
            "gpu": {"id": gpu_id, "count": 1},
            "dataCenterIds": [datacenter_id],
            "disk": 50,
        },
        timeout=60,
    )


def create_with_backoff(gpu_id, datacenter_id, attempts=5):
    """Rate limits and 5xx are not the candidate's fault, so they are
    retried in place rather than moving to the next GPU."""
    for attempt in range(attempts):
        response = create(gpu_id, datacenter_id)
        if response.status_code == 429:
            # Retry-After is integer seconds on this API.
            time.sleep(int(response.headers.get("Retry-After", 5)))
            continue
        if response.status_code >= 500:
            time.sleep(2**attempt)
            continue
        return response
    raise RuntimeError(f"{attempts} transient failures for {gpu_id}; upstream unhealthy")


def deploy(region="EUROPE"):
    last_detail = None

    for gpu_id, datacenter_id in candidates(region):
        response = create_with_backoff(gpu_id, datacenter_id)

        if response.status_code == 201:
            return response.json()

        problem = response.json()
        last_detail = problem.get("detail")

        if response.status_code == 422:
            # Contract violation — identical on every candidate.
            raise SystemExit(f"Bad request: {problem.get('errors', last_detail)}")
        if response.status_code == 402:
            raise SystemExit(f"Cannot deploy: {last_detail}")
        if response.status_code in (400, 403):
            # 403: no access to this pool. 400: rule violation, or this
            # GPU/data center could not be placed. Either way, move on.
            continue

        response.raise_for_status()

    # Every candidate was refused. A rule violation fails the same way on
    # all of them, so the last detail is the useful signal here.
    raise SystemExit(f"No candidate placed in {region}. Last error: {last_detail}")


if __name__ == "__main__":
    pod = deploy()
    print(f"{pod['id']} placed in {pod['dataCenterId']} on {pod['gpu']['id']}")