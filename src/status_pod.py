import sys

from common import POD_NAME, pod_base_url
from get_pod import get_active, get_by_id, loaded_models


def status(pod_id=None, name=POD_NAME):
    """Check the pod first; only if it is RUNNING ask vLLM which models are loaded.

    Returns {"pod": record|None, "healthy": bool, "models": [..], "url": str|None}.
    "healthy" means the pod is RUNNING and at least one model is served.
    """
    pod = get_by_id(pod_id) if pod_id else get_active(name)
    if pod is None:
        return {"pod": None, "healthy": False, "models": [], "url": None}

    if pod["status"] != "RUNNING":
        return {"pod": pod, "healthy": False, "models": [], "url": None}

    models = loaded_models(pod["id"])
    return {
        "pod": pod,
        "healthy": bool(models),
        "models": models,
        "url": pod_base_url(pod["id"]) if models else None,
    }


if __name__ == "__main__":
    s = status(sys.argv[1] if len(sys.argv) > 1 else None)
    pod = s["pod"]

    if pod is None:
        print("pod: none")
    else:
        print(f"pod:    {pod['id']}  {pod['status']}  {pod['gpu']['id']}")
        print(f"model:  {', '.join(s['models']) or 'not loaded'}")
        print(f"url:    {s['url'] or '-'}")

    print(f"health: {'ok' if s['healthy'] else 'not ready'}")
    sys.exit(0 if s["healthy"] else 1)
