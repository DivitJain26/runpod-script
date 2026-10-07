import requests
from common import API, POD_NAME, SESSION, VLLM_API_KEY, pod_base_url


def get_by_id(pod_id):
    """Return one pod's record, or None if Runpod has no pod with that id."""
    r = SESSION.get(f"{API}/v2/pods/{pod_id}", timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def get_all(name=None, include_terminated=False):
    """Return every pod on the account, newest first; filter by name and drop terminated unless asked."""
    pods = []
    cursor = None
    while True:
        params = {"limit": 100}
        if cursor:
            params["cursor"] = cursor
        r = SESSION.get(f"{API}/v2/pods", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        for p in data["pods"]:
            if name and p["name"] != name:
                continue
            if not include_terminated and p["status"] == "TERMINATED":
                continue
            pods.append(p)
        if not data["pagination"]["hasNextPage"]:
            return pods
        cursor = data["pagination"]["nextCursor"]


def get_active(name=POD_NAME):
    """Return the one RUNNING pod with this name, or None."""
    running = [p for p in get_all(name=name) if p["status"] == "RUNNING"]
    if len(running) > 1:
        print(f"[get_active] warning: {len(running)} running pods named {name}, using newest")
    return running[0] if running else None

def is_model_loaded(pod_id, timeout=10):
    """Return True if vLLM on this pod answers /v1/models with the right key; False otherwise."""
    try:
        r = requests.get(
            f"{pod_base_url(pod_id)}/v1/models",
            headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
            timeout=timeout,
        )
    except requests.RequestException:
        return False  # pod not reachable yet, or proxy not routing
    if r.status_code == 401:
        raise RuntimeError("vLLM rejected the api key")
    return r.status_code == 200


def loaded_models(pod_id, timeout=10):
    """Return the model names vLLM is serving, or [] if it is not up."""
    if not is_model_loaded(pod_id, timeout):
        return []
    r = requests.get(
        f"{pod_base_url(pod_id)}/v1/models",
        headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
        timeout=timeout,
    )
    return [m["id"] for m in r.json().get("data", [])]

if __name__ == "__main__":
    for p in get_all():
        print(f"{p['id']}  {p['name']:<24} {p['status']:<13} {p['gpu']['id']}")
    pod = get_active()
    print(f"\nactive: {pod_base_url(pod['id']) if pod else 'none'}")
