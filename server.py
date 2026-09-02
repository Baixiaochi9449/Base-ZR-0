from policies.reasoning_vla_policy import ZR0Policy
from utils.cli_options import parse_server_options
from utils.websocket_server_policy import WebsocketPolicyServer
import random
import numpy as np
import torch
import json

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_option(args=None):
    return parse_server_options(args)

def deploy():
    opt = parse_option()
    set_all_seeds(42)

    assert opt.inference_mode in ["direct_action", "subtask_then_action"]
    # settings
    kwargs = {
        "dataset_entry": opt.dataset_entry,
        "ckpt_dir": opt.ckpt_dir,
        "inference_mode": opt.inference_mode,
        "window_size": opt.window_size,
        "num_denoised_steps": opt.num_denoised_steps,
        "max_pad_state_and_action_length": opt.max_pad_state_and_action_length,
        "device": "cuda:0",
        "use_difference_query": opt.use_difference_query,
        "num_difference_queries": opt.num_difference_queries,
        "vlm_attention_backend": opt.vlm_attention_backend,
    }
    print(json.dumps(kwargs, indent=2, ensure_ascii=False))
    policy = ZR0Policy(**kwargs)
    
    print("Start serving.")
    host = "0.0.0.0"
    port = opt.port
    server = WebsocketPolicyServer(policy, host, port)
    server.serve_forever()

if __name__ == "__main__":
    deploy()
