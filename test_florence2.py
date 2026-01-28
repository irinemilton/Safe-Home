"""
Test Florence-2 model loading independently
"""
import sys
from types import ModuleType
from unittest.mock import MagicMock

# Mock flash_attn because it's hard to install on Windows but we don't need it for CPU/eager mode
# We use ModuleType and set __spec__ to satisfy importlib checks
m = ModuleType('flash_attn')
m.__spec__ = MagicMock()
sys.modules['flash_attn'] = m

print("Testing Florence-2...")

try:
    import torch
    print(f"[OK] PyTorch version: {torch.__version__}")
    print(f"[OK] CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"[ERROR] PyTorch import failed: {e}")
    sys.exit(1)

try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    print("[OK] Transformers imported")
except Exception as e:
    print(f"[ERROR] Transformers import failed: {e}")
    sys.exit(1)

try:
    print("\n[INFO] Loading Florence-2 model (this will download ~2GB on first run)...")
    model_id = 'microsoft/Florence-2-base'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading model from: {model_id}")
    
    # Use eager attention to avoid flash_attn dependency
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True,
        attn_implementation='eager'  # Bypass flash_attn requirement
    ).to(device).eval()
    
    processor = AutoProcessor.from_pretrained(
        model_id, 
        trust_remote_code=True
    )
    
    print("[SUCCESS] Florence-2 model loaded successfully!")
    print(f"[INFO] Model type: {type(model)}")
    print(f"[INFO] Processor type: {type(processor)}")
    
    # Test Inference with dummy image
    print("\n[INFO] Testing inference with dummy image...")
    from PIL import Image
    import numpy as np
    
    # Create black image
    dummy_image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    
    prompt = "<MORE_DETAILED_CAPTION>"
    inputs = processor(text=prompt, images=dummy_image, return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=50,
        do_sample=False,
        num_beams=3,
    )
    
    result_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    print(f"[SUCCESS] Inference result: {result_text}")
    
except Exception as e:
    print(f"[ERROR] Florence-2 loading/inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[SUCCESS] All tests passed! Florence-2 is ready to use.")
