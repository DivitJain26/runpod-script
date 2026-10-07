import sys
import time

from common import API, POD_NAME, SESSION
from get_pod import get_all, get_by_id


def terminate(pod_id):
    """Permanently delete one pod. Returns True if it is gone (or already was)."""
    r = SESSION.post(f"{API}/v2/pods/{pod_id}/action", json={"action": "terminate"}, timeout=30)
    if r.status_code in (204, 404):
        return True
    print(f"[terminate] {pod_id} failed: HTTP {r.status_code} {r.text[:120]}")
    return False


def terminate_all(name=POD_NAME):
    """Terminate every non-terminated pod with this name. Returns the ids it killed."""
    return [p["id"] for p in get_all(name=name) if terminate(p["id"])]


def wait_terminated(pod_id, timeout_s=120):
    """Block until Runpod reports TERMINATED or the pod is gone; False on timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        pod = get_by_id(pod_id)
        if pod is None or pod["status"] == "TERMINATED":
            return True
        time.sleep(5)
    return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("terminated" if terminate(sys.argv[1]) else "failed")
    else:
        killed = terminate_all()
        print(f"terminated {len(killed)}: {killed}" if killed else "nothing to terminate")
