import base64
import logging
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
HEADERS = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}

POD_NAME = "metroleads-inference"
VLLM_PORT = 8000


def pod_base_url(pod_id):
    """Build the public proxy URL for a pod id."""
    return f"https://{pod_id}-{VLLM_PORT}.proxy.runpod.net"


logger = logging.getLogger(__name__)


### get 
def get_by_id(pod_id):
    """
    GET /v2/pods/{id}. Returns the pod dict, or None if missing or the call failed.
    """
    try:
        r = SESSION.get('{}/v2/pods/{}'.format(API, pod_id), timeout=30)
        if r.status_code == 404:
            logger.warning('Pod %s not found', pod_id)
            return None
        r.raise_for_status()
        pod = r.json()
        logger.info('Pod %s: %s on %s (%s)', pod_id, pod['status'], pod.get('gpu', {}).get('id'), pod['cloud'])
        return pod
    except requests.RequestException as e:
        logger.error('get_by_id(%s) failed: %s', pod_id, e)
        return None
 
 
def get_active():
    """
    GET /v2/pods, following the cursor. Returns every RUNNING pod on the account, newest first.
    """
    active = []
    cursor = None
    page = 0
    while True:
        page += 1
        r = SESSION.get('{}/v2/pods'.format(API), params={'limit': 100, 'cursor': cursor}, timeout=30)
        r.raise_for_status()
        data = r.json()
        logger.debug('Pods page %d: %d pods', page, len(data['pods']))
        active += [p for p in data['pods'] if p['status'] == 'RUNNING']
        if not data['pagination']['hasNextPage']:
            break
        cursor = data['pagination']['nextCursor']
 
    if active:
        logger.info('%d RUNNING pods, $%.2f/h total', len(active), sum(p['cost'] for p in active))
    else:
        logger.warning('No RUNNING pods')
    return active