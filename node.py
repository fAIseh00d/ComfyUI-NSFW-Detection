# NSFW Detection supporting multiple models
from PIL import Image
import timm
import timm.data
from timm.models import load_checkpoint
from transformers import AutoModelForImageClassification, ViTImageProcessor
import torch
import numpy
import os
import json
import folder_paths

try:
    from server import PromptServer
except:
    PromptServer = None

AVAILABLE_MODELS = [
    "Marqo/nsfw-image-detection-384",
    "Falconsai/nsfw_image_detection",
    "AdamCodd/vit-base-nsfw-detector",
]


def tensor2pil(image):
    return Image.fromarray(numpy.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(numpy.uint8))


def pil2tensor(image):
    return torch.from_numpy(numpy.array(image).astype(numpy.float32) / 255.0).unsqueeze(0)


class NSFWDetection:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "score": ("FLOAT", {
                    "default": 0.80,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "nsfw_threshold"}),
                "model": (AVAILABLE_MODELS,),
            },
            "optional": {
                "alternative_image": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "run"

    CATEGORY = "NSFWDetection"

    def run(self, image, score, model, alternative_image=None, unique_id=None):
        if model == "Marqo/nsfw-image-detection-384":
            classifier = self._load_marqo_model()
        else:
            # Both Falconsai and AdamCodd use transformers ViT
            folder_map = {
                "Falconsai/nsfw_image_detection": "nsfw_image_detection",
                "AdamCodd/vit-base-nsfw-detector": "vit-base-nsfw-detector",
            }
            classifier = self._load_vit_model(folder_map[model])
        
        max_nsfw_score = 0.0
        for i in range(len(image)):
            pil_img = tensor2pil(image[i])
            image_size = image[i].size()
            width, height = image_size[1], image_size[0]
            
            nsfw_score = classifier(pil_img)
            max_nsfw_score = max(max_nsfw_score, nsfw_score)
            
            if nsfw_score > score:
                if alternative_image is None:
                    raise Exception(f"NSFW content detected in image {i} with score {nsfw_score:.3f} (threshold: {score})")
                alt_pil = tensor2pil(alternative_image[0])
                image[i] = pil2tensor(alt_pil.resize((width, height), resample=Image.Resampling.BILINEAR))
        
        # Display max NSFW probability on the node
        if unique_id and PromptServer is not None:
            try:
                status = "🔴 NSFW" if max_nsfw_score > score else "🟢 SFW"
                msg = f"<tr><td>NSFW Detection</td><td>{status} ({max_nsfw_score:.1%})</td></tr>"
                PromptServer.instance.send_progress_text(msg, unique_id)
            except:
                pass

        return (image,)
    
    def _load_marqo_model(self):
        model_path = os.path.join(folder_paths.models_dir, "nsfw_detector", "nsfw-image-detection-384")
        
        config_file = os.path.join(model_path, "config.json")
        with open(config_file, "r") as f:
            config = json.load(f)
        
        architecture = config["architecture"]
        pretrained_cfg = config["pretrained_cfg"]
        num_classes = config.get("num_classes", pretrained_cfg.get("num_classes", 2))
        
        model = timm.create_model(architecture, pretrained=False, num_classes=num_classes, pretrained_cfg=pretrained_cfg)
        checkpoint_file = os.path.join(model_path, "model.safetensors")
        load_checkpoint(model, checkpoint_file)
        model = model.eval()
        
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        
        class_names = config["label_names"]
        nsfw_idx = class_names.index("NSFW") if "NSFW" in class_names else 0
        
        def classify(pil_img):
            with torch.no_grad():
                output = model(transforms(pil_img).unsqueeze(0)).softmax(dim=-1).cpu()
            return output[0][nsfw_idx].item()
        
        return classify
    
    def _load_vit_model(self, folder_name):
        """Load transformers-based ViT models (Falconsai, AdamCodd)"""
        model_path = os.path.join(folder_paths.models_dir, "nsfw_detector", folder_name)
        
        model = AutoModelForImageClassification.from_pretrained(model_path, local_files_only=True)
        processor = ViTImageProcessor.from_pretrained(model_path, local_files_only=True)
        model = model.eval()
        
        # Find NSFW label index (handles both "nsfw" and "NSFW")
        id2label = model.config.id2label
        nsfw_idx = None
        for idx, label in id2label.items():
            if label.lower() == "nsfw":
                nsfw_idx = int(idx)
                break
        
        def classify(pil_img):
            with torch.no_grad():
                inputs = processor(images=pil_img, return_tensors="pt")
                outputs = model(**inputs)
                probs = outputs.logits.softmax(dim=-1).cpu()
            return probs[0][nsfw_idx].item() if nsfw_idx is not None else 0.0
        
        return classify


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "NSFWDetection": NSFWDetection
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "NSFWDetection": "NSFW Detection"
}
