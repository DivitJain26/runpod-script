import base64
import logging
import os
import time
import ujson as json


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

RUNPOD_API = 'https://api.runpod.io'
PODS_URL = '{}/v2/pods'.format(RUNPOD_API)

HEADERS = {"Authorization": "Bearer {}".format(RUNPOD_API_KEY)}
VLLM_HEADERS = {"Authorization": "Bearer {}".format(VLLM_API_KEY)}

POD_NAME = "metroleads-inference"
VLLM_PORT = 8000


def pod_base_url(pod_id):
    """Build the public proxy URL for a pod id."""
    return f"https://{pod_id}-{VLLM_PORT}.proxy.runpod.net"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


### get 
def get_by_id(pod_id):
    """
    Fetch a RunPod pod by its ID.

    Sends a GET request to the RunPod API and returns the pod details as a dictionary.
    Returns None if the pod does not exist or if the API request fails.
    """
    
    url = '{}/{}'.format(PODS_URL, pod_id)
    
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        
        if r.status_code != 200:
            logger.error('get_by_id failed pod_id=%s http_status=%s response=%s', pod_id, r.status_code, r.text)
            return None
        
        pod = json.loads(r.text)
        logger.info('RunPod pod details pod_id=%s status=%s gpu=%s cloud=%s', pod['id'], pod['status'], pod.get('gpu', {}).get('id'), pod['cloud'])
        return pod
    
    except requests.RequestException as ex:
        logger.exception('RunPod API request failed for pod_id=%s error=%s', pod_id, ex)
        return None
 
 
def get_active():
    """
    Fetch all RUNNING pods on RunPod.

    Returns a list of RUNNING pods, or None if the API request fails.
    """
    try:
        r = requests.get(PODS_URL, headers=HEADERS, timeout=30)

        if r.status_code != 200:
            logger.error('get_active failed http_status=%s response=%s', r.status_code, r.text)
            return None

        data = json.loads(r.text)
        active = [pod for pod in data.get('pods', []) if pod.get('status') == 'RUNNING']

        for pod in active:
            logger.info('RunPod pod details pod_id=%s status=%s gpu=%s cloud=%s', pod['id'], pod['status'], pod.get('gpu', {}).get('id'), pod['cloud'])

        logger.info('RunPod active pods count=%s', len(active))
        return active

    except requests.RequestException as ex:
        logger.exception('RunPod API request failed for active pods error=%s', ex)
        return None
    
    
## healthcheck
def health_check(pod_id):
    """
    Health check a RunPod pod.

    Returns 200 if the model is loaded and serving, otherwise an error status.
    """
    pod = get_by_id(pod_id)
    if not pod:
        logger.warning("Health check failed: pod not found pod_id=%s", pod_id)
        return 404

    if pod.get("status") != "RUNNING":
        logger.warning("Health check skipped pod_id=%s status=%s", pod_id, pod.get("status"))
        return 409

    url = '{}/v1/models'.format(pod_base_url(pod_id))

    try:
        r = requests.get(url, headers=VLLM_HEADERS, timeout=10)
        
        if r.status_code != 200:
            logger.error('Health check failed pod_id=%s http_status=%s', pod_id, r.status_code)
            return r.status_code

        data = json.loads(r.text)
        models = [model.get('id') for model in data.get('data', []) if model.get('id')]
        
        if not models:
            logger.warning('vLLM reachable but no models loaded pod_id=%s', pod_id)
            return 503

        logger.info('Health check ok pod_id=%s models=%s', pod_id, models)
        return 200

    except requests.RequestException as ex:
        logger.warning('Health check unreachable pod_id=%s error=%s', pod_id, ex)
        return 503
    
## terminate
def terminate_by_id(pod_id):
    """
    Permanently delete a RunPod pod.

    Returns 204 if the pod is terminated, otherwise the error status from the API.
    """
    
    url = '{}/{}'.format(PODS_URL, pod_id)

    try:
        r = requests.delete(url, headers=HEADERS, timeout=30)

        if r.status_code not in (200, 204):
            logger.error('terminate_by_id failed pod_id=%s http_status=%s response=%s', pod_id, r.status_code, r.text)
            return r.status_code

        logger.info('RunPod pod terminated pod_id=%s http_status=%s', pod_id, r.status_code)
        return r.status_code

    except requests.RequestException as ex:
        logger.exception('RunPod API request failed for pod_id=%s error=%s', pod_id, ex)
        return 503
    

## Create

GPU_PREFERENCE = (
    'NVIDIA A100-SXM4-80GB',
    'NVIDIA A100 80GB PCIe',
    'NVIDIA RTX PRO 6000 Blackwell Server Edition'
)
CLOUD_PREFERENCE = ('COMMUNITY', 'SECURE')
USABLE_AVAILABILITY = ('LOW', 'MEDIUM', 'HIGH')
AVAILABILITY_RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

def candidates(region=None):
    """
    Read the RunPod catalog and return deployable (gpu_id, datacenter_id, cloud) triples, best first.

    Returns [] if the catalog call fails or nothing matches GPU_PREFERENCE / USABLE_AVAILABILITY.
    """

    url = '{}/v2/catalog/datacenters'.format(RUNPOD_API)
    params = {'regions': region, 'include': 'GPU_AVAILABILITY'}

    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    except requests.RequestException as ex:
        logger.exception('Catalog request failed error=%s', ex)
        return []

    if r.status_code != 200:
        logger.error('Catalog failed http_status=%s response=%s', r.status_code, r.text)
        return []

    found, seen = [], set()

    datacenters = json.loads(r.text).get('dataCenters', [])
    for dc in datacenters:

        available_gpus = dc.get('gpuAvailability', [])
        for gpu in available_gpus:

            if gpu['id'] not in GPU_PREFERENCE or gpu['availability'] not in USABLE_AVAILABILITY:
                continue

            for cloud in CLOUD_PREFERENCE:
                key = (gpu['id'], dc['id'] if cloud == 'SECURE' else None, cloud)  # community is not dc-scoped
                if key not in seen:
                    seen.add(key)
                    found.append((gpu['id'], dc['id'], cloud, gpu['availability']))

    found.sort(key=lambda c: (GPU_PREFERENCE.index(c[0]), CLOUD_PREFERENCE.index(c[2]), AVAILABILITY_RANK[c[3]]))
    
    logger.info('Catalog: %s candidates', len(found))
    for gpu_id, dc_id, cloud, availability in found:
        logger.info('Candidate gpu=%s datacenter=%s cloud=%s availability=%s', gpu_id, dc_id, cloud, availability)
        
    return [(gpu_id, dc_id, cloud) for gpu_id, dc_id, cloud, _ in found]