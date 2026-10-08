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

API = "https://api.runpod.io"
HEADERS = {"Authorization": "Bearer {}".format(RUNPOD_API_KEY)}

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
    try:
        r = requests.get("{}/v2/pods/{}".format(API, pod_id), headers=HEADERS, timeout=30)
        if r.status_code != 200:
            logger.error('get_by_id failed pod_id=%s http_status=%s response=%s', pod_id, r.status_code, r.text)
            return None
        pod = json.loads(r.text)
        logger.info('RunPod pod details pod_id=%s status=%s gpu=%s cloud=%s', pod['id'], pod['status'], pod.get('gpu', {}).get('id'), pod['cloud'])
        return pod
    except requests.RequestException as ex:
        logger.exception('RunPod API request failed error=%s', ex)
        return None
 
 
# def get_active():
#     """
#     GET /v2/pods, following the cursor. Returns every RUNNING pod on the account, newest first.
#     """
#     active = []
#     cursor = None
#     page = 0
#     while True:
#         page += 1
#         r = SESSION.get('{}/v2/pods'.format(API), params={'limit': 100, 'cursor': cursor}, timeout=30)
#         r.raise_for_status()
#         data = r.json()
#         logger.debug('Pods page %d: %d pods', page, len(data['pods']))
#         active += [p for p in data['pods'] if p['status'] == 'RUNNING']
#         if not data['pagination']['hasNextPage']:
#             break
#         cursor = data['pagination']['nextCursor']
 
#     if active:
#         logger.info('%d RUNNING pods, $%.2f/h total', len(active), sum(p['cost'] for p in active))
#     else:
#         logger.warning('No RUNNING pods')
#     return active