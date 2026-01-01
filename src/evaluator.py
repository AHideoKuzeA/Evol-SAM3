import os
import torch
import json
import datetime
from tqdm import tqdm
from PIL import Image

from src.utils import (
    ExperimentLogger, 
    AverageMeter, 
    intersectionAndUnion, 
    get_mask_from_json,
    save_checkpoint,
    load_checkpoint
)
from src.dataset import get_dataset_loader
from src.solver import Solver

class Evaluator:
    def __init__(self, cfg, qwen_engine, sam_engine):
        self.cfg = cfg
        self.qwen = qwen_engine
        self.sam = sam_engine
        
        self.log_dir = cfg.paths.log_dir
        self.dataset_root = cfg.paths.dataset_root
        self.split = cfg.dataset.split
        self.debug_mode = cfg.debug.mode
        self.debug_dir = cfg.debug.output_dir

        os.makedirs(self.log_dir, exist_ok=True)
        if self.debug_mode:
            os.makedirs(self.debug_dir, exist_ok=True)
       
        self.meters = {
            "all": {
                "inter": AverageMeter("I"), 
                "union": AverageMeter("U"), 
                "giou": AverageMeter("gIoU")
            }
        }
        if self.split == "test":
            self.meters["short"] = {
                "inter": AverageMeter("I"), 
                "union": AverageMeter("U"), 
                "giou": AverageMeter("gIoU")
            }
            self.meters["long"] = {
                "inter": AverageMeter("I"), 
                "union": AverageMeter("U"), 
                "giou": AverageMeter("gIoU")
            }

        self.ckpt_path = os.path.join(self.log_dir, "checkpoint.json")

    def run(self):
        loader = get_dataset_loader(self.cfg)
        dataset = loader.data
        
        start_index = 0
        if os.path.exists(self.ckpt_path):
            start_index = load_checkpoint(self.ckpt_path, self.meters) 
        
        print(f"[Evaluator] Starting from index {start_index} / {len(dataset)}")
        
        summary_log_path = os.path.join(self.log_dir, "summary.log")

        if not os.path.exists(summary_log_path):
            with open(summary_log_path, 'w', encoding='utf-8') as f:
                f.write("=== Experiment Summary === \n")
                f.write(f"Start Time: {datetime.datetime.now()}\n")
                f.write("--- Configuration ---\n")

                gpu_val = os.environ.get("CUDA_VISIBLE_DEVICES", "N/A")
                if gpu_val == "N/A" and hasattr(self.cfg, 'system') and hasattr(self.cfg.system, 'gpu_ids'):
                    gpu_val = self.cfg.system.gpu_ids


                f.write(f" gpu          : {gpu_val} \n")
                f.write(f" resume       : {os.path.exists(self.ckpt_path)} \n")
                f.write(f" debug        : {self.debug_mode} \n")
                f.write("-" * 20 + "\n")

                f.write(f" dataset_root : {self.dataset_root} \n")        
                d_name = getattr(self.cfg.dataset, 'name', None)
                d_split = getattr(self.cfg.dataset, 'split', None)
                d_type = getattr(self.cfg.dataset, 'type', None)
                if d_name:
                    f.write(f" data_name    : {d_name} \n")
                if d_split:
                    f.write(f" data_split   : {d_split} \n")
                if d_type:
                    f.write(f" data_type    : {d_type} \n")
                f.write("-" * 20 + "\n")
               
                f.write(f" qwen_path    : {self.cfg.paths.qwen_model_path} \n")
                f.write(f" sam_path     : {self.cfg.paths.sam3_ckpt_path} \n")
                f.write(f" log_dir      : {self.log_dir} \n")
                f.write(f" g_max        : {self.cfg.evolution.max_generations} \n")
                f.write(f" n_pop        : {self.cfg.evolution.n_pop} \n")
                f.write(f" use_arena    : {self.cfg.ablation.use_arena} \n")
                f.write(f" check_mode   : {self.cfg.ablation.check_mode} \n")
                
                f.write(" ==============================| \n")
        
        def log_summary(msg):
            with open(summary_log_path, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}\n")

        pbar = tqdm(range(start_index, len(dataset)), desc="Evaluating")
        for i in pbar:
            item = dataset[i]
            
            img_path = item['p']
            dataset_type = getattr(self.cfg.dataset, 'type', 'reason_seg')
            
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            
            img_pil = Image.open(img_path).convert("RGB")
            import numpy as np
            img_np = np.array(img_pil)
            h, w = img_np.shape[:2]

            if dataset_type == 'reason_seg':
                json_path = item['json_p']
                fname = item['fname']
                gt_mask, query, is_long = get_mask_from_json(json_path, img_np)
            elif dataset_type == 'res':
                original_fname = item['fname']
                fname_no_ext = os.path.splitext(original_fname)[0]
                ref_id = item.get('ref_id', 'unknown')
                fname = f"{fname_no_ext}_ref{ref_id}_idx{i}" 
                
                query = item['query']
                ann_id = item['ann_id']
                gt_mask = loader.get_gt_mask(ann_id, h, w)
                is_long = len(query.split()) > 3 
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")
            sample_logger = ExperimentLogger(self.log_dir, fname, resume=False) 
            
            sample_logger.log("Data", f"Processing [{i}/{len(dataset)}]: {fname}")
            sample_logger.log("Data", f"Query: {query}")
            
            # Run Solver
            solver = Solver(
                cfg=self.cfg,
                img_bytes=img_bytes,
                query=query,
                logger=sample_logger, # Pass individual logger
                mllm_engine=self.qwen,
                sam_engine=self.sam,
                fname=fname
            )
            
            final_ind = solver.run()
            
            pred_mask = None
            if final_ind and final_ind.M is not None:
                pred_mask = final_ind.M
            else:
                pred_mask = np.zeros_like(gt_mask)

            intersection, union, target = intersectionAndUnion(
                torch.from_numpy(pred_mask).long(), 
                torch.from_numpy(gt_mask).long(), 
                2, ignore_index=255
            )
            
            i_val = intersection[1].item()
            u_val = union[1].item()
            iou_val = i_val / (u_val + 1e-10)
            if u_val == 0: iou_val = 1.0 
            
            # Update Meters
            self.meters["all"]["inter"].update(i_val)
            self.meters["all"]["union"].update(u_val)
            self.meters["all"]["giou"].update(iou_val)
            
            if self.split == "test":
                k = "long" if is_long else "short"
                self.meters[k]["inter"].update(i_val)
                self.meters[k]["union"].update(u_val)
                self.meters[k]["giou"].update(iou_val)
            

            curr_giou = self.meters["all"]["giou"].avg * 100
            curr_ciou = (self.meters["all"]["inter"].sum / (self.meters["all"]["union"].sum + 1e-10)) * 100
            
            pbar.set_postfix({"gIoU": f"{curr_giou:.2f}", "cIoU": f"{curr_ciou:.2f}"})
            

            type_str = "long" if is_long else "short"
            log_msg = f"{fname} | IoU: {iou_val*100:.2f}% | {type_str} | cIoU: {curr_ciou:.2f}% | gIoU: {curr_giou:.2f}%"
            log_summary(log_msg)
            
            sample_logger.log("Metric", f"Sample IoU: {iou_val:.4f}")
            
            # Save Checkpoint
            if (i + 1) % 1 == 0: 
                save_checkpoint(self.ckpt_path, i, self.meters, self.split)

        print("\n" + "="*60)
        print(f"      ReasonSeg ({self.split}) Evaluation Results")
        print("="*60)
        print(f"{'Split':<10} | {'gIoU':<10} | {'cIoU':<10} | {'Count'}")
        print("-" * 60)
        
        keys_to_print = ["all"]
        if self.split == "test":
            keys_to_print.extend(["short", "long"])

        for key in keys_to_print:
            m = self.meters[key]
            if m["giou"].count > 0:
                giou = m["giou"].avg * 100
                ciou = (m["inter"].sum / (m["union"].sum + 1e-10)) * 100
                count = m["giou"].count
                print(f"{key.capitalize():<10} | {giou:.2f}       | {ciou:.2f}       | {count}")
                log_summary(f"Final {key}: gIoU={giou:.2f}, cIoU={ciou:.2f}")
            else:
                print(f"{key.capitalize():<10} | N/A        | N/A        | 0")
        print("="*60)

