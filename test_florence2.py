"""
Test Florence-2 model loading independently
"""
import sys
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
    
except Exception as e:
    print(f"[ERROR] Florence-2 loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[SUCCESS] All tests passed! Florence-2 is ready to use.")
