import os
import re
import random
import torch
import numpy as np
import gc
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from PIL import Image

from src.utils import calculate_iou, MaskRenderer, get_high_contrast_color
from src.prompts import RS_PROMPTS, RES_PROMPTS
from src.models import MLLMBase

class Individual:
    def __init__(self, p, origin="Init", r_init=0.0):
        self.P = p
        self.M = None
        self.score_gate = 0.0
        self.pei = r_init
        self.fitness = 0.0
        self.is_sam_confirmed = False
        self.is_mllm_confirmed = False
        self.origin = origin
        self.display_color = None
        self.color_name = "green"
        self.box = None
    
    def __repr__(self): 
        return f"'{self.P}' (Fit:{self.fitness:.1f}|{self.origin})"

class Solver:
    def __init__(self, cfg, img_bytes, query, logger, mllm_engine, sam_engine, fname="temp"):
        self.cfg = cfg
        self.img_bytes = img_bytes
        self.q_list = []
        try:
            if query.strip().startswith("[") and query.strip().endswith("]"):
                self.q_list = eval(query) 
            else:
                self.q_list = [query]
        except:
            self.q_list = [query]
            
        self.Q = self.q_list[0] 
        self.mllm = MLLMBase(mllm_engine, img_bytes)
        self.sam = sam_engine
        self.render = MaskRenderer()
        self.logger = logger
        self.pop = []
        self.T_final = ""
        self.mask_cache = {}
        
        self.fname = fname
        self.mask_result_cache = {}

        self.dataset_type = getattr(self.cfg.dataset, 'type', 'reason_seg')
        if self.dataset_type == 'res':
            self.prompts = RES_PROMPTS
        else:
            self.prompts = RS_PROMPTS


    def _hash_mask(self, mask): return hash(mask.tobytes())
    
    def _clean_prompt_output(self, raw):
        match = re.search(r'\{([^{}]+)\}', raw, re.DOTALL)
        if match: return match.group(1).strip()
        clean = raw.strip()
        if ":" in clean: parts = clean.split(":", 1); clean = parts[1] if len(parts[1])>1 else parts[0]
        clean = clean.replace('*', '').replace('_', '')
        clean = clean.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
        clean = clean.replace('"', '').replace("'", "")
        clean = re.sub(r'^[\d\.\-\s\•]+', '', clean)
        return clean.strip()
    
    def _split_model_output(self, raw_text):
        normalized_text = raw_text.replace(',', '\n')
        results = []
        for line in normalized_text.split('\n'):
            clean = self._clean_prompt_output(line)
            if len(clean) > 2 and "Here are" not in clean and "Output" not in clean:
                results.append(clean)
        return results
    
    def _deduplicate_prompts(self, prompts):
        seen = set()
        unique = []
        for p in prompts:
            if not isinstance(p, str): continue
            norm = p.lower().strip()
            if norm and norm not in seen:
                seen.add(norm)
                unique.append(p) 
        return unique

    def _parse_gate_score(self, text):
        if re.search(r'\b1\.0\b', text) or re.search(r'\b1\b', text) or "yes" in text.lower(): return 1.0
        return 0.0

    def _parse_arena_judge_3img(self, text):
        text = text.upper()
        score_a = 0
        score_b = 0
        if re.search(r'\bA\b', text): score_a += 1
        if re.search(r'\bB\b', text): score_b += 1
        if "IMAGE 2" in text: score_a += 1
        if "IMAGE 3" in text: score_b += 1
        
        if score_a > score_b: return "CANDIDATE_A" 
        if score_b > score_a: return "CANDIDATE_B"
        return "TIE"

    def _parse_box_coords(self, text, width, height):
        match = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', text)
        if match:
            coords = [int(match.group(i)) for i in range(1, 5)]
            x1, y1, x2, y2 = coords
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            return [x1, y1, x2, y2]
        return None

    def _filter_preds_by_box_overlap(self, preds, box, img_shape, threshold=0.5):
        if not preds:
            return []
        
        tx1, ty1, tx2, ty2 = box
        h, w = img_shape
        
        # Clip target box
        tx1, ty1 = int(max(0, tx1)), int(max(0, ty1))
        tx2, ty2 = int(min(w, tx2)), int(min(h, ty2))
        target_area = max(0, tx2 - tx1) * max(0, ty2 - ty1)
        
        self.logger.log("BoxFilter", f"--- Box-IoU Filtering {len(preds)} candidates (Thresh={threshold}) ---")
        
        valid_preds = []
        for i, p in enumerate(preds):
            mask = p['mask']
            # Ensure mask is 2D
            if mask.ndim > 2:
                mask = mask.squeeze()
                
            # Get mask bounding box
            # Note: numpy indices are [y, x]
            mask = mask > 0
            y_indices, x_indices = np.where(mask)
            
            if len(y_indices) == 0:
                self.logger.log("BoxFilter", f"  [Cand {i}] Empty Mask -> DROP")
                continue
                
            mx1, mx2 = np.min(x_indices), np.max(x_indices) + 1
            my1, my2 = np.min(y_indices), np.max(y_indices) + 1
            
            mask_area = max(0, mx2 - mx1) * max(0, my2 - my1)
            
            # Intersection
            ix1 = max(tx1, mx1)
            iy1 = max(ty1, my1)
            ix2 = min(tx2, mx2)
            iy2 = min(ty2, my2)
            
            inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            
            # Union
            union_area = target_area + mask_area - inter_area
            
            # Box IoU
            box_iou = inter_area / (union_area + 1e-6)
            
            p['box_iou'] = box_iou
            
            status = "DROP"
            if box_iou > threshold:
                valid_preds.append(p)
                status = "KEEP"
            
            self.logger.log("BoxFilter", f"  [Cand {i}] Conf={p['conf']:.2f} | MaskBox=[{mx1},{my1},{mx2},{my2}] | BoxIoU={box_iou:.4f} -> {status}")
                
        self.logger.log("BoxFilter", f"--- Result: {len(valid_preds)}/{len(preds)} candidates retained ---")
        return valid_preds

    def _phase0_meta_planning(self, initial_q_list):
        current_q = initial_q_list[0]
        dataset_type = getattr(self.cfg.dataset, 'type', 'reason_seg')
        
        if dataset_type == 'res':
            # RES Logic: Normalization
            self.logger.log("Phase0", f"Init Query (RES): '{current_q}'")
            p_init = self.prompts["PHASE0_RES_NORMALIZATION"].format(Q=current_q)
            raw_list = self.mllm.query(p_init)
            candidates = self._split_model_output(raw_list)
            current_best = candidates[0] if candidates else current_q
            self.logger.log("Phase0", f"Normalized: '{current_best}'")
            return [current_best]
            
        else:
            # RS Logic: Iterative Refine
            self.logger.log("Phase0", f"Init Query: '{current_q}'")
            
            p_init = self.prompts["PHASE0_RS_INIT"].format(Q=current_q)
            raw_list = self.mllm.query(p_init)
            candidates = self._split_model_output(raw_list)
            
            current_best = candidates[0] if candidates else "object"
            self.logger.log("Phase0", f"Initial Best: '{current_best}'")
            
            MAX_REFINE_STEPS = 3
            history = [current_best]
            
            for i in range(MAX_REFINE_STEPS):
                iter_q = random.choice(initial_q_list)
                prompt = self.prompts["PHASE0_ITERATIVE_REFINE"].format(
                    Q=iter_q, current_best=current_best
                )
                res = self.mllm.query(prompt) 
                clean_res = self._clean_prompt_output(res)
                
                if "NO" in clean_res.upper() or clean_res == 'B':
                    self.logger.log("Phase0", f"Step {i+1}: Model satisfied. Stop.")
                    break
                elif len(clean_res) > 2:
                    current_best = clean_res
                    history.append(current_best)
                    self.logger.log("Phase0", f"Step {i+1}: Refined -> '{current_best}'")
                else:
                    self.logger.log("Phase0", f"Step {i+1}: Invalid output '{clean_res}'. Stop.")
                    break
            return list(reversed(history))

    def _evaluate_step(self, gen_idx):
        SAM_HIGH_CONF = 0.85 
        SAM_LOW_CONF = 0.35  
        
        pending_mllm_checks = [] 
        img_pil = Image.open(BytesIO(self.img_bytes)).convert("RGB")
        img_np = np.array(img_pil)
        
        for ind in self.pop:
            if ind.P in self.mask_cache:
                ind.M = self.mask_cache[ind.P]
                if not hasattr(ind, 'sam_conf'): ind.sam_conf = 0.5 
            
            if ind.M is None:
                torch.cuda.empty_cache()
                preds = self.sam.predict(Image.open(BytesIO(self.img_bytes)), ind.P)
                
                if not preds: 
                    ind.score_gate = 0.0
                    ind.sam_conf = 0.0
                    continue
                
                combined_mask = np.zeros_like(preds[0]['mask'])
                best_conf = 0.0
                for p in preds:
                    combined_mask = np.logical_or(combined_mask, p['mask'])
                    if p['conf'] > best_conf: best_conf = p['conf']
                
                ind.M = combined_mask
                ind.sam_conf = best_conf
                self.mask_cache[ind.P] = ind.M

            if ind.M is not None and ind.display_color is None:
                ind.display_color, ind.color_name = get_high_contrast_color(img_np, ind.M)
                self.logger.log("Color", f"Gen {gen_idx}: '{ind.P}' -> color_name={ind.color_name}")
            
            if ind.score_gate == 0.0 and ind.fitness == 0.0:
                sam_conf = getattr(ind, 'sam_conf', 0.0)
                
                if sam_conf > SAM_HIGH_CONF:
                    ind.is_sam_confirmed = True
                    ind.is_mllm_confirmed = True 
                elif sam_conf < SAM_LOW_CONF:
                    ind.is_sam_confirmed = False
                    ind.is_mllm_confirmed = False
                else:
                    if sam_conf > self.cfg.evolution.sam_conf_threshold:
                        ind.is_sam_confirmed = True
                    
                    m_hash = self._hash_mask(ind.M)
                    if m_hash in self.mask_result_cache:
                        cached_result = self.mask_result_cache[m_hash]
                        if cached_result: ind.is_mllm_confirmed = True
                    else:
                        gate_bytes = self.render.to_bytes(self.render.overlay(self.img_bytes, ind.M, color=ind.display_color))
                        pending_mllm_checks.append((ind, [self.img_bytes, gate_bytes], m_hash, ind.color_name))

        if pending_mllm_checks:
            self.logger.log("Eval", f"Async checking {len(pending_mllm_checks)} unique masks...")
            def check_single(args):
                _, img_data_list, _, c_name = args 
                try:
                    prompt = self.prompts["SEMANTIC_GATE"].format(
                        Q=self.Q, T_final=self.T_final, color_name=c_name
                    )
                    res = self.mllm.query(prompt, image_data=img_data_list)
                    score = self._parse_gate_score(res)
                    return score == 1.0, res
                except Exception as e:
                    return False, "Error"

            MAX_WORKERS = 3 
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                results = list(executor.map(check_single, pending_mllm_checks))
            
            for i, (is_pass, raw_res) in enumerate(results):
                ind, _, m_hash, c_name = pending_mllm_checks[i]
                self.mask_result_cache[m_hash] = is_pass
                if is_pass: ind.is_mllm_confirmed = True

        for ind in self.pop:
            if ind.is_sam_confirmed or ind.is_mllm_confirmed:
                ind.score_gate = 1.0
            else:
                ind.score_gate = 0.0

    def _check_termination(self, valid_pop):
        if len(valid_pop) < 2: return False, "Insufficient elites"
        peis = [ind.pei for ind in valid_pop]
        pei_diff = max(peis) - min(peis)
        top_inds = valid_pop[:3]
        ious = [calculate_iou(top_inds[i].M, top_inds[j].M) for i in range(len(top_inds)) for j in range(i+1, len(top_inds))]
        avg_iou = sum(ious)/len(ious) if ious else 0.0
        if pei_diff <= self.cfg.evolution.theta_pei and avg_iou >= self.cfg.evolution.theta_iou: return True, f"Converged (IoU={avg_iou:.2f})"
        return False, "Running"
    
    def log_population_status(self, title):
        self.logger.log("Status", f"--- {title} ---")
        header = f"{'Idx':<3} | {'Prompt (Truncated)':<30} | {'Origin':<10} | {'SAM Conf':<8} | {'PEI':<5} | {'Gate':<4} | {'Fitness':<7}"
        sep_line = "-" * len(header)
        
        self.logger.log("Status", header)
        self.logger.log("Status", sep_line)
        
        for i, ind in enumerate(self.pop):
            p_str = (ind.P[:27] + '...') if len(ind.P) > 30 else ind.P
            conf_str = f"{getattr(ind, 'sam_conf', 0.0):.4f}"
            pei_str = f"{getattr(ind, 'pei', 0.0):.2f}"
            gate_str = "YES" if ind.score_gate == 1.0 else "NO"
            fit_str = f"{ind.fitness:.2f}"
            origin_str = getattr(ind, 'origin', 'Unknown')[:10]
            row = f"{i:<3} | {p_str:<30} | {origin_str:<10} | {conf_str:<8} | {pei_str:<5} | {gate_str:<4} | {fit_str:<7}"
            self.logger.log("Status", row)
            
        self.logger.log("Status", sep_line)

    def _final_arbitration(self, text_best_ind):
        final_ind = None
        img_pil = Image.open(BytesIO(self.img_bytes)).convert("RGB")
        img_np = np.array(img_pil)
        w, h = img_pil.size
        
        # --- 1. Generate Box Result ---
        box_ind = None
        if self.cfg.ablation.output_mode != 'text_only':
            self.logger.log("FinalChoice", "🔍 Generating Rescue Box for comparison...")
            prompt_box = self.prompts["PHASE_PRE_BOX_DETECT"].format(Q=self.Q, width=w, height=h)
            box_res = self.mllm.query(prompt_box)
            box_coords = self._parse_box_coords(box_res, w, h)
            
            if box_coords:
                preds = self.sam.predict_box(img_pil, box_coords)
                
                #If RES dataset, apply IoU filtering
                dataset_type = getattr(self.cfg.dataset, 'type', 'reason_seg')
                if dataset_type == 'res':
                    img_shape = (h, w) # h, w order
                    preds = self._filter_preds_by_box_overlap(preds, box_coords, img_shape, threshold=0.7)

                if preds:
                    best_pred = max(preds, key=lambda x: x['conf'])
                    box_ind = Individual(f"[Rescue Box: {box_coords}]", origin="Rescue_Box")
                    box_ind.M = best_pred['mask']
                    box_ind.sam_conf = best_pred['conf']
                    box_ind.display_color, box_ind.color_name = get_high_contrast_color(img_np, box_ind.M)
                    box_ind.box = box_coords
                    self.logger.log("FinalChoice", f"✅ Box Result: {box_coords}, Conf={best_pred['conf']:.2f}")
                else:
                    self.logger.log("FinalChoice", "❌ SAM failed on Box (or filtered out).")
            else:
                self.logger.log("FinalChoice", "❌ MLLM failed to predict Box.")

        # --- 2. Logic Branch ---
        
        # Case A: Only Box
        if (text_best_ind is None or text_best_ind.M is None) and box_ind is not None:
            self.logger.log("Result", "⚠️ Text Failed. Using Box Result.")
            final_ind = box_ind
            
        # Case B: Only Text
        elif (text_best_ind is not None) and box_ind is None:
            self.logger.log("Result", "⚠️ Box Failed. Using Text Result.")
            final_ind = text_best_ind
            
        # Case C: Both -> Arbitration
        elif (text_best_ind is not None) and (box_ind is not None):
            iou = calculate_iou(text_best_ind.M, box_ind.M)
            self.logger.log("FinalChoice", f"⚔️ ARBITRATION: Text vs Box (IoU={iou:.2f})")
            
            union_mask = np.logical_or(text_best_ind.M, box_ind.M)
            arena_color, arena_cname = get_high_contrast_color(img_np, union_mask)
            
            bytes_text = self.render.to_bytes(self.render.overlay(self.img_bytes, text_best_ind.M, color=arena_color))
            bytes_box = self.render.to_bytes(self.render.overlay(self.img_bytes, box_ind.M, color=arena_color,box=box_ind.box))             
            
            # Round 1: Text(A) vs Box(B)
            prompt1 = self.prompts["ARENA_JUDGE"].format(
                Q=self.Q, P_A=f"Text: {text_best_ind.P}", P_B=f"Box: {box_ind.P}", color_name=arena_cname
            )
            res1 = self.mllm.query(prompt1, image_data=[self.img_bytes, bytes_text, bytes_box])
            win1 = self._parse_arena_judge_3img(res1)
            
            check_mode = self.cfg.ablation.check_mode
            
            if check_mode == 'one':
                # One check: Trust Round 1
                if win1 == "CANDIDATE_A":
                    self.logger.log("FinalChoice", "🏆 Winner: TEXT (One Check)")
                    final_ind = text_best_ind
                else:
                    self.logger.log("FinalChoice", "🏆 Winner: BOX (One Check)")
                    final_ind = box_ind
            else:
                # Double Check
                # Round 2: Box(A) vs Text(B)
                prompt2 = self.prompts["ARENA_JUDGE"].format(
                    Q=self.Q, P_A=f"Box: {box_ind.P}", P_B=f"Text: {text_best_ind.P}", color_name=arena_cname
                )
                res2 = self.mllm.query(prompt2, image_data=[self.img_bytes, bytes_box, bytes_text])
                win2 = self._parse_arena_judge_3img(res2)
                
                self.logger.log("FinalChoice", f"   R1 [Orig, Text, Box]: {win1}")
                self.logger.log("FinalChoice", f"   R2 [Orig, Box, Text]: {win2}")
                
                if win1 == "CANDIDATE_A" and win2 == "CANDIDATE_B":
                    self.logger.log("FinalChoice", f"🏆 Winner: TEXT (Consistent)")
                    final_ind = text_best_ind
                elif win1 == "CANDIDATE_B" and win2 == "CANDIDATE_A":
                    self.logger.log("FinalChoice", f"🏆 Winner: BOX (Consistent)")
                    final_ind = box_ind
                else:
                    self.logger.log("FinalChoice", "⚠️ INCONSISTENT JUDGMENT (Hallucination Detected).")
                    
                    dataset_type = getattr(self.cfg.dataset, 'type', 'reason_seg')
                    if dataset_type == 'res':
                         self.logger.log("FinalChoice", "   -> [RES Rule] FALLBACK TO BOX (Rescue Priority).")
                         final_ind = box_ind
                    else:
                        self.logger.log("FinalChoice", "   -> [RS Rule] FALLBACK TO TEXT.")
                        final_ind = text_best_ind
        else:
            return None

        # --- Debug Visual ---
        if self.cfg.debug.mode and final_ind:
            safe_fname = os.path.basename(self.fname)
            box_to_draw = getattr(final_ind, 'box', None)
            vis_final = self.render.overlay(self.img_bytes, final_ind.M, color=final_ind.display_color, box=box_to_draw)
            save_path = os.path.join(self.cfg.debug.output_dir, f"{safe_fname}_final.jpg")
            vis_final.convert('RGB').save(save_path)
            self.logger.log("Debug", f"Saved: {save_path}")

        return final_ind

    def run(self):
        gc.collect()
        torch.cuda.empty_cache()

        # --- Ablation: Box Only ---
        if self.cfg.ablation.output_mode == 'box_only':
            return self._final_arbitration(None) # Pass None as text_ind to trigger Box Only logic

        # Phase 0
        
        dataset_type = getattr(self.cfg.dataset, 'type', 'reason_seg')
        if dataset_type == 'res':
            # RES: Add original query to population
            self.pop.append(Individual(self.Q, origin="Original_Q", r_init=self.cfg.evolution.r_init))
            
        candidate_prompts = self._phase0_meta_planning(self.q_list)
        if candidate_prompts:
            self.T_final = str(candidate_prompts[0])
            # For RES, candidate_prompts[0] is the normalized query
        else:
            self.T_final = self.Q
            candidate_prompts = [self.Q]

        for p in candidate_prompts:
            if len(self.pop) >= self.cfg.evolution.n_pop: break
            if isinstance(p, str) and p not in [ind.P for ind in self.pop]:
                self.pop.append(Individual(p, origin="Phase0", r_init=self.cfg.evolution.r_init))
        
        if len(self.pop) < self.cfg.evolution.n_pop:
            p1 = self.prompts["REFINE_TEMPLATE"].format(T_final=self.T_final, Q=self.Q)
            res_p1 = self.mllm.query(p1)
            for p in self._deduplicate_prompts(self._split_model_output(res_p1)):
                if len(self.pop) >= self.cfg.evolution.n_pop: break
                self.pop.append(Individual(p, origin="InitRefine", r_init=self.cfg.evolution.r_init))

        self._evaluate_step(0)
        self.log_population_status("Gen 0")

        best_text_ind = None
        img_pil = Image.open(BytesIO(self.img_bytes)).convert("RGB")
        img_np = np.array(img_pil)

        for g in range(self.cfg.evolution.max_generations):
            self.logger.log("Phase2", f"--- Gen {g+1} Start ---")
            elites = [ind for ind in self.pop if ind.score_gate == 1.0]
            
            # --- Selection / Ranking Strategy ---
            if self.cfg.ablation.use_arena:
                # Arena Mode
                if len(elites) >= 2:
                    elites.sort(key=lambda x: x.pei, reverse=True)
                    champ = elites[0]
                    potential_battles = []
                    for chal in elites[1:]:
                        iou_diff = calculate_iou(champ.M, chal.M)
                        if iou_diff > self.cfg.evolution.theta_arena_skip: continue
                        potential_battles.append((chal, iou_diff))
                    potential_battles.sort(key=lambda x: x[1])
                    
                    fights_count = 0
                    for chal, iou_val in potential_battles:
                        if fights_count >= self.cfg.evolution.max_arena_fights: break
                        union_mask = np.logical_or(champ.M, chal.M)
                        arena_color, arena_cname = get_high_contrast_color(img_np, union_mask)
                        self.logger.log("Arena", f"⚔️ Fight {fights_count+1}: '{champ.P}' vs '{chal.P}'")
                        
                        bytes_champ = self.render.to_bytes(self.render.overlay(self.img_bytes, champ.M, color=arena_color))
                        bytes_chal = self.render.to_bytes(self.render.overlay(self.img_bytes, chal.M, color=arena_color))
                        
                        prompt = self.prompts["ARENA_JUDGE"].format(
                            Q=self.Q, P_A=champ.P, P_B=chal.P, color_name=arena_cname
                        )
                        res = self.mllm.query(prompt, image_data=[self.img_bytes, bytes_champ, bytes_chal])
                        win = self._parse_arena_judge_3img(res)
                        self.logger.log("Arena", f"   -> Winner: {win}")
                        
                        if win == "CANDIDATE_A": 
                            champ.pei += self.cfg.evolution.pei_update_w; chal.pei -= self.cfg.evolution.pei_update_w
                        elif win == "CANDIDATE_B": 
                            champ.pei -= self.cfg.evolution.pei_update_w*2; chal.pei += self.cfg.evolution.pei_update_w*2
                        fights_count += 1
            else:
                # No Arena: Direct Scoring
                self.logger.log("Ablation", "No Arena. Using Direct Scoring.")
                for ind in elites:
                     bytes_ind = self.render.to_bytes(self.render.overlay(self.img_bytes, ind.M, color=ind.display_color))
                     prompt = self.prompts["DIRECT_SCORE"].format(
                         Q=self.Q, P=ind.P, color_name=ind.color_name
                     )
                     try:
                         res = self.mllm.query(prompt, image_data=[self.img_bytes, bytes_ind])
                         match = re.search(r'(\d+)', res)
                         score = float(match.group(1)) if match else 0.0
                         ind.pei = score / 10.0 # Scale to ~0-10 range to match fitness scale roughly
                         self.logger.log("Scoring", f"Ind '{ind.P}' -> Score: {score}")
                     except Exception as e:
                         ind.pei = 0.0
            
            # Update Fitness
            for ind in self.pop: 
                # If using arena, pei is around 0 +/- changes. 
                # If using direct score, pei is 0-10.
                ind.fitness = 10.0 + ind.pei
            
            valid_pop = [ind for ind in self.pop if ind.score_gate == 1.0]
            if not valid_pop:
                # Fallback to T_final
                self.pop = [Individual(self.T_final, origin="Fallback", r_init=self.cfg.evolution.r_init)]
                self._evaluate_step(g+1)
                valid_pop = [ind for ind in self.pop if ind.score_gate == 1.0]
            
            valid_pop.sort(key=lambda x: x.fitness, reverse=True)
            if valid_pop: 
                best_text_ind = valid_pop[0]
                self.logger.log("Evo", f"👑 Elite Kept: '{best_text_ind.P}'")

            if len(valid_pop) >= 2:
                stop, reason = self._check_termination(valid_pop)
                if stop: break

            # Reproduction
            if valid_pop:
                best = valid_pop[0]
                next_gen = [best] 
                if len(valid_pop) > 1: next_gen.append(valid_pop[1])
                
                mut_text = self._clean_prompt_output(self.mllm.query(self.prompts["REFINE_SIMPLIFY"].format(P_old=best.P, T_final=self.T_final, Q=self.Q)))
                if mut_text: next_gen.append(Individual(mut_text, origin="Mutate", r_init=self.cfg.evolution.r_init))
                
                fill_res = self.mllm.query(self.prompts["REFINE_TEMPLATE"].format(T_final=best.P, Q=self.Q))
                for cl in self._split_model_output(fill_res):
                    if len(cl)>2: next_gen.append(Individual(cl, origin="Refine", r_init=self.cfg.evolution.r_init))
                
                unique_pop = []
                seen_prompts = set()
                for ind in next_gen:
                    if ind.P not in seen_prompts:
                        seen_prompts.add(ind.P); unique_pop.append(ind)
                
                self.pop = unique_pop[:self.cfg.evolution.n_pop]
                self._evaluate_step(g+1)
            
            self.log_population_status(f"Gen {g+1}")

        valid_pop = [ind for ind in self.pop if ind.score_gate == 1.0]
        if valid_pop: 
            best_text_ind = max(valid_pop, key=lambda x: x.fitness)
        
        # Final Arbitration
        final_ind = self._final_arbitration(best_text_ind)
        
        if final_ind:
            self.logger.log("Result", f"🏆 FINAL: '{final_ind.P}'")
            return final_ind
        
        return None
