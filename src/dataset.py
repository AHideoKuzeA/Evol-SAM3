import os
import glob
import pickle
try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None

class RefCOCODatasetLoader:
    def __init__(self, root, dataset="refcoco", split="val"):
        """
        Args:
            root: 数据集根目录 (包含 refcoco, refcoco+, refcocog, images 等子目录)
            dataset: "refcoco", "refcoco+", "refcocog"
            split: "val", "testA", "testB", "val_google", "test_unc" 等 (取决于数据集)
        """
        if COCO is None:
            raise ImportError("pycocotools is required for RefCOCODatasetLoader. Please install it.")

        self.root = root
        self.dataset = dataset
        self.split = split
        self.data = []
        
        coco_anno_path = os.path.join(root, dataset, "instances.json")
        self.coco = COCO(coco_anno_path)
    
        if dataset == "refcocog":
            ref_path = os.path.join(root, dataset, "refs(umd).p")
            if not os.path.exists(ref_path):
                ref_path = os.path.join(root, dataset, "refs(google).p")
        elif dataset == "refcoco" or dataset == "refcoco+":
            ref_path = os.path.join(root, dataset, "refs(unc).p")
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
            
        print(f"[Data] Loading refs from {ref_path}...")
        with open(ref_path, 'rb') as f:
            self.refs = pickle.load(f)
            
        self._load_split()

    def _load_split(self):
        print(f"[Data] Filtering split: {self.split}...")
        count = 0
        for ref in self.refs:
            if ref['split'] != self.split:
                continue

            img_id = ref['image_id']
            img_info = self.coco.loadImgs(img_id)[0]
            file_name = img_info['file_name']
            
            img_path = os.path.join(self.root, "images", "train2014", file_name)

            ann_id = ref['ann_id']
            ann = self.coco.loadAnns(ann_id)[0]
            
            for sent in ref['sentences']:
                query = sent['sent'] # 引用文本
                
                self.data.append({
                    'p': img_path,
                    'query': query,
                    'ann_id': ann_id,
                    'img_id': img_id,
                    'fname': file_name,
                    'ref_id': ref['ref_id'], # <--- 确保这一项明确存在
                    'sent_id': sent['sent_id'], # <--- 如果想更细，甚至可以加 sent_id
                    'raw_ref': ref 
                })
                count += 1
                
        print(f"[Data] Loaded {len(self.data)} samples for {self.dataset}/{self.split}.")

    def get_gt_mask(self, ann_id, h, w):
        ann = self.coco.loadAnns(ann_id)[0]
        mask = self.coco.annToMask(ann)
        return mask

class ReasonSegDatasetLoader:
    def __init__(self, root, split="test"):
        self.root = root; self.split = split; self.data = []
        self._load()
    def _load(self):
        split_dir = os.path.join(self.root, self.split)
        print(f"[Data] Scanning {split_dir}...")
        if not os.path.exists(split_dir): return
        img_paths = sorted(glob.glob(os.path.join(split_dir, "*.jpg")) + glob.glob(os.path.join(split_dir, "*.png")))
        for p in img_paths:
            base = os.path.splitext(p)[0]
            if os.path.exists(base + ".json"):
                self.data.append({'p': p, 'json_p': base + ".json", 'fname': os.path.basename(p)})
        print(f"[Data] Loaded {len(self.data)} pairs.")

def get_dataset_loader(cfg):
    dataset_type = getattr(cfg.dataset, 'type', 'reason_seg')
    root = cfg.paths.dataset_root
    split = cfg.dataset.split
    
    if dataset_type == 'reason_seg':
        return ReasonSegDatasetLoader(root, split=split)
    elif dataset_type == 'res':
        dataset_name = getattr(cfg.dataset, 'name', 'refcoco')
        return RefCOCODatasetLoader(root, dataset=dataset_name, split=split)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
