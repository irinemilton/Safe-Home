"""
Florence-2 Defect Detector
Advanced vision model for concrete/structural defect detection
No training required - works out of the box
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock

# Mock flash_attn because it's hard to install on Windows but we don't need it for CPU/eager mode
# We use ModuleType and set __spec__ to satisfy importlib checks
m = ModuleType('flash_attn')
m.__spec__ = MagicMock()
sys.modules['flash_attn'] = m

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class Florence2Detector:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_model(self):
        """Load Florence-2 model (downloads ~2GB on first run)"""
        try:
            print(f"[Florence-2] Loading model to {self.device}...")
            model_id = 'microsoft/Florence-2-base'
            
            # Use eager attention to avoid flash_attn dependency (not needed for CPU)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                trust_remote_code=True,
                attn_implementation='eager'  # Bypass flash_attn requirement
            ).to(self.device).eval()
            
            self.processor = AutoProcessor.from_pretrained(
                model_id, 
                trust_remote_code=True
            )
            
            print("[Florence-2] Model loaded successfully!")
            return True
        except Exception as e:
            print(f"[Florence-2] Error loading model: {e}")
            return False
    
    def analyze_image(self, image_path_or_pil):
        """
        Analyze image for defects
        Returns: (defect_type, confidence, description, bounding_boxes)
        """
        if self.model is None:
            return None, 0, "Model not loaded", []
        
        try:
            # Load image
            if isinstance(image_path_or_pil, str):
                image = Image.open(image_path_or_pil).convert("RGB")
            else:
                image = image_path_or_pil.convert("RGB")
            
            # Task 1: Get detailed description
            description = self._get_description(image)
            
            # Task 2: Detect defects (phrase grounding)
            defects = self._detect_defects(image)
            
            # Analyze results
            defect_type, confidence = self._analyze_results(description, defects)
            
            return defect_type, confidence, description, defects
            
        except Exception as e:
            return None, 0, f"Analysis error: {str(e)}", []
    
    def _get_description(self, image):
        """Get detailed caption of the image"""
        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )
        
        result_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            result_text, 
            task=prompt, 
            image_size=(image.width, image.height)
        )
        
        return parsed.get('<MORE_DETAILED_CAPTION>', 'No description available')
    
    def _detect_defects(self, image):
        """Detect specific defects using phrase grounding"""
        # Look for common structural defects
        prompt = "<CAPTION_TO_PHRASE_GROUNDING> Find cracks, damage, holes, water stains, rust, or structural defects."
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )
        
        result_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            result_text, 
            task=prompt, 
            image_size=(image.width, image.height)
        )
        
        return parsed.get('<CAPTION_TO_PHRASE_GROUNDING>', {})
    
    def _analyze_results(self, description, defects):
        """Analyze description and defects to determine severity"""
        description_lower = description.lower()
        
        # Check for critical keywords
        critical_keywords = ['crack', 'hole', 'damage', 'broken', 'collapse', 'severe']
        warning_keywords = ['stain', 'discolor', 'minor', 'small']
        
        has_critical = any(kw in description_lower for kw in critical_keywords)
        has_warning = any(kw in description_lower for kw in warning_keywords)
        
        # Check if defects were detected
        has_defects = bool(defects.get('bboxes', []))
        
        if has_critical or has_defects:
            if 'crack' in description_lower:
                return 'major_crack', 85.0
            elif 'hole' in description_lower or 'damage' in description_lower:
                return 'spalling', 80.0
            else:
                return 'major_crack', 75.0
        elif has_warning:
            return 'stain', 70.0
        else:
            return 'normal', 60.0
    
    def draw_bounding_boxes(self, image, defects, output_path=None):
        """Draw bounding boxes on image"""
        if not defects or 'bboxes' not in defects:
            return image
        
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        bboxes = defects.get('bboxes', [])
        labels = defects.get('labels', [])
        
        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = bbox
            # Draw red box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
            # Draw label background
            draw.rectangle([x1, y1, x1 + 100, y1 + 25], fill="red")
            # Draw text
            draw.text((x1 + 5, y1 + 5), label, fill="white", font=font)
        
        if output_path:
            img_copy.save(output_path)
        
        return img_copy

# Global instance
florence_detector = None

def get_florence_detector():
    """Get or create Florence-2 detector instance"""
    global florence_detector
    if florence_detector is None:
        florence_detector = Florence2Detector()
        florence_detector.load_model()
    return florence_detector
