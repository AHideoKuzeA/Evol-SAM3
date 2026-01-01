import torch
import os
import tempfile
from typing import List, Dict, Any, Union
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

class QwenEngine:
    _instance = None
    def __new__(cls, model_path=None):
        if cls._instance is None: 
            cls._instance = super(QwenEngine, cls).__new__(cls)
            cls._instance.model_path = model_path
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        print(f"[Qwen] Loading from {self.model_path}..."); 
        try:
            torch.cuda.empty_cache()
            device_map_config = "auto"
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path, 
                torch_dtype="auto", 
                device_map=device_map_config, 
                trust_remote_code=True
            )
        except Exception as e:
            print(f"[Qwen Error] Failed to load model. Error: {e}")
            raise e
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True); print("[Qwen] Ready.")

    def generate(self, image_paths: List[str], prompt: str) -> str:
        content = []
        for img_path in image_paths:
            content.append({"type": "image", "image": f"file://{img_path}"})
        
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.model.device) 
        
        with torch.no_grad(): 
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        
        full = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if "assistant\n" in full: return full.split("assistant\n")[-1].strip()
        elif "assistant" in full: return full.split("assistant")[-1].strip()
        return full.strip()

class SAM3Engine:
    _instance = None
    def __new__(cls, ckpt_path=None):
        if cls._instance is None: 
            cls._instance = super(SAM3Engine, cls).__new__(cls)
            cls._instance.ckpt_path = ckpt_path
            cls._instance._load_model()
        return cls._instance
    
    def _load_model(self):
        print(f"[SAM3] Loading from {self.ckpt_path}..."); 
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch.cuda.empty_cache()
        self.model = build_sam3_image_model(checkpoint_path=self.ckpt_path, load_from_HF=False, device=self.device)
        self.model = self.model.to(self.device)
        self.model.eval() 
        self.processor = Sam3Processor(self.model, device=self.device) 
        print(f"[SAM3] Ready on {self.device}.")
    
    def predict(self, image_pil: Image.Image, text_prompt: str) -> List[Dict[str, Any]]:
        if image_pil.mode != "RGB": image_pil = image_pil.convert("RGB")
        try:
            clean_prompt = text_prompt.split("\n")[0].strip()[:100]
            if not clean_prompt: return []
            with torch.cuda.device(self.device):
                inference_state = self.processor.set_image(image_pil)
                output = self.processor.set_text_prompt(state=inference_state, prompt=clean_prompt)
                masks, scores = output["masks"], output["scores"]
                results = []
                masks_np, scores_np = masks.cpu().numpy() > 0, scores.cpu().numpy()
                for i in range(len(scores_np)):
                    m, s = masks_np[i], float(scores_np[i])
                    if m.sum() < 50: continue 
                    while m.ndim > 2: m = m.squeeze(0)
                    results.append({'mask': m, 'conf': s})
                return results
        except Exception as e: 
            print(f"[SAM3 Error] Text predict failed: {e}")
            return []

    def predict_box(self, image_pil: Image.Image, box: List[int]) -> List[Dict[str, Any]]:
        if image_pil.mode != "RGB": image_pil = image_pil.convert("RGB")
        try:
            width, height = image_pil.size
            with torch.cuda.device(self.device):
                inference_state = self.processor.set_image(image_pil)
                x1, y1, x2, y2 = box
                x1 = max(0, min(x1, width - 1)); y1 = max(0, min(y1, height - 1))
                x2 = max(x1 + 1, min(x2, width)); y2 = max(y1 + 1, min(y2, height))
                
                cx = (x1 + (x2 - x1) / 2) / width
                cy = (y1 + (y2 - y1) / 2) / height
                nw = (x2 - x1) / width
                nh = (y2 - y1) / height
                normalized_box = [cx, cy, nw, nh]
                
                output = self.processor.add_geometric_prompt(box=normalized_box, label=True, state=inference_state)
                masks, scores = output["masks"], output["scores"]
                results = []
                masks_np, scores_np = masks.cpu().numpy() > 0, scores.cpu().numpy()
                
                for i in range(len(scores_np)):
                    m, s = masks_np[i], float(scores_np[i])
                    if m.sum() < 50: continue 
                    results.append({'mask': m, 'conf': s})
                return results
        except Exception as e:
            print(f"[SAM3 Error] Box predict failed: {e}")
            return []

class MLLMBase:
    def __init__(self, engine, img_bytes): 
        self.eng = engine; self.raw_bytes = img_bytes
    
    def query(self, prompt: str, image_data: Union[bytes, List[bytes], None] = None) -> str:
        target_data = []
        if image_data is None:
            target_data = [self.raw_bytes]
        elif isinstance(image_data, list):
            target_data = image_data
        else:
            target_data = [image_data] 

        temp_files = []
        temp_paths = []
        
        try:
            for i, data in enumerate(target_data):
                tfile = tempfile.NamedTemporaryFile(suffix=f"_{i}.jpg", delete=False)
                tfile.write(data)
                tfile.close()
                temp_files.append(tfile)
                temp_paths.append(tfile.name)
            
            res = self.eng.generate(temp_paths, prompt)
        finally:
            for tpath in temp_paths:
                if os.path.exists(tpath):
                    try: os.remove(tpath)
                    except: pass
        return res
