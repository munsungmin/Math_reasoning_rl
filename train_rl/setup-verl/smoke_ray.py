import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json

import ray
ray.init(num_cpus=2, include_dashboard=False, object_store_memory=100 * 1024**2,
         _node_ip_address='127.0.0.1')
try:
    @ray.remote
    def probe():
        import verl, torch
        return {'verl': verl.__version__, 'torch': torch.__version__}
    print('RAY PASS', json.dumps(ray.get(probe.remote(), timeout=180)), flush=True)
finally:
    ray.shutdown()
print('RAY SMOKE TEST PASSED', flush=True)
