import os
import argparse
import yaml
from types import SimpleNamespace

def dict_to_namespace(d):
    x = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(x, k, dict_to_namespace(v))
        else:
            setattr(x, k, v)
    return x

def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return dict_to_namespace(config_dict)

def main():
    parser = argparse.ArgumentParser(description="SAM3Agent Refactored")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--gpu", type=str, default=None, help="GPU IDs to use (e.g., '0' or '0,1')")
    parser.add_argument("--resume", type=str, default=None, help="Path to log directory to resume from")
    args = parser.parse_args()

    # Load Config
    cfg = load_config(args.config)
    print(f"[Main] Loaded config from {args.config}")

    # Default Setting
    if not hasattr(cfg, 'ablation'):
        cfg.ablation = SimpleNamespace()
    
    if not hasattr(cfg.ablation, 'use_arena'):
        cfg.ablation.use_arena = True
    if not hasattr(cfg.ablation, 'check_mode'):
        cfg.ablation.check_mode = "double"
    if not hasattr(cfg.ablation, 'output_mode'):
        cfg.ablation.output_mode = "hybrid"
    
    if args.resume:
        cfg.paths.log_dir = args.resume
        print(f"[Main] Resuming from existing directory: {cfg.paths.log_dir}")
    else:
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        cfg.paths.log_dir = os.path.join(cfg.paths.log_dir, f"{config_name}_{timestamp}")
    
    if hasattr(cfg, 'debug'):
        cfg.debug.output_dir = cfg.paths.log_dir
        
    print(f"[Main] Log Directory: {cfg.paths.log_dir}")

    gpu_ids = args.gpu
    if gpu_ids is None:
        if hasattr(cfg, 'system') and hasattr(cfg.system, 'gpu_ids'):
            gpu_ids = str(cfg.system.gpu_ids)

    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        print(f"[Main] Set CUDA_VISIBLE_DEVICES={gpu_ids}")
    
    import torch
    from src.models import QwenEngine, SAM3Engine
    from src.evaluator import Evaluator

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    
    qwen = QwenEngine(model_path=cfg.paths.qwen_model_path)
    sam = SAM3Engine(ckpt_path=cfg.paths.sam3_ckpt_path)
    
    evaluator = Evaluator(cfg, qwen, sam)
    
    evaluator.run()

if __name__ == "__main__":
    main()
