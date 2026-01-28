"""
Fix Keras model by removing quantization_config parameter from the saved model file.
This script extracts the model, removes the incompatible parameter, and resaves it.
"""
import zipfile
import json
import os
import shutil

def fix_keras_model(model_path):
    """Remove quantization_config from a Keras model file"""
    print(f"[INFO] Fixing model: {model_path}")
    
    # Create backup
    backup_path = model_path + ".backup"
    if not os.path.exists(backup_path):
        shutil.copy2(model_path, backup_path)
        print(f"[INFO] Backup created: {backup_path}")
    
    # Extract the .keras file (it's a zip)
    extract_dir = model_path + "_temp"
    os.makedirs(extract_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"[INFO] Extracted model to: {extract_dir}")
        
        # Find and fix config.json
        config_path = os.path.join(extract_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Recursively remove quantization_config
            def remove_quantization_config(obj):
                if isinstance(obj, dict):
                    # Remove the key if it exists
                    if 'quantization_config' in obj:
                        del obj['quantization_config']
                    # Recursively process all values
                    for value in obj.values():
                        remove_quantization_config(value)
                elif isinstance(obj, list):
                    for item in obj:
                        remove_quantization_config(item)
            
            remove_quantization_config(config)
            
            # Save fixed config
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            print(f"[OK] Removed quantization_config from config.json")
            
            # Repackage the model
            fixed_model_path = model_path.replace('.keras', '_fixed.keras')
            with zipfile.ZipFile(fixed_model_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(extract_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, extract_dir)
                        zipf.write(file_path, arcname)
            
            print(f"[OK] Fixed model saved to: {fixed_model_path}")
            
            # Clean up
            shutil.rmtree(extract_dir)
            print(f"[INFO] Cleaned up temporary files")
            
            return fixed_model_path
        else:
            print(f"[ERROR] config.json not found in model")
            return None
            
    except Exception as e:
        print(f"[ERROR] Failed to fix model: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Clean up temp dir if it exists
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)

if __name__ == "__main__":
    model_path = "models/bd3_model_modern.keras"
    fixed_path = fix_keras_model(model_path)
    
    if fixed_path:
        print(f"\n[SUCCESS] Model fixed successfully!")
        print(f"Original: {model_path}")
        print(f"Fixed: {fixed_path}")
        print(f"Backup: {model_path}.backup")
        print(f"\nYou can now use the fixed model in your app.")
    else:
        print(f"\n[FAILED] Could not fix the model")
