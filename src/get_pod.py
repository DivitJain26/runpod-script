from common import API, POD_NAME, SESSION, pod_base_url


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


if __name__ == "__main__":
    for p in get_all():
        print(f"{p['id']}  {p['name']:<24} {p['status']:<13} {p['gpu']['id']}")
    pod = get_active()
    print(f"\nactive: {pod_base_url(pod['id']) if pod else 'none'}")
