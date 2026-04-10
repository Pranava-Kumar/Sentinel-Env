"""
Force clear all Python caches and recompile inference.py
Run this BEFORE submission to ensure clean state
"""
import os
import shutil
import py_compile
from pathlib import Path

def clear_all_caches():
    """Remove all Python cache files"""
    print("🧹 Clearing Python caches...")
    
    # Remove __pycache__ directories
    for pycache in Path(".").rglob("__pycache__"):
        if pycache.is_dir():
            shutil.rmtree(pycache)
            print(f"  Removed: {pycache}")
    
    # Remove .pyc files
    for pyc in Path(".").rglob("*.pyc"):
        pyc.unlink()
        print(f"  Removed: {pyc}")
    
    # Remove .pyo files
    for pyo in Path(".").rglob("*.pyo"):
        pyo.unlink()
        print(f"  Removed: {pyo}")
    
    print("✓ All caches cleared")

def verify_inference():
    """Verify inference.py has the fix"""
    print("\n🔍 Verifying inference.py...")
    
    with open("inference.py", "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Check line 74 area
    for i in range(70, 90):
        if i < len(lines):
            line = lines[i]
            if "MAX_STEPS" in line:
                print(f"  Line {i+1}: {line.rstrip()}")
                if 'os.getenv("MAX_STEPS"' not in line and "int(os.getenv" in line:
                    print("  ❌ ERROR: Line has int() without default!")
                    return False
                elif 'os.getenv("MAX_STEPS", "20")' in line or "try:" in line:
                    print("  ✅ Fix is present!")
                    return True
    
    print("  ⚠️  Could not verify fix location")
    return False

def recompile_inference():
    """Force recompile inference.py"""
    print("\n🔨 Recompiling inference.py...")
    try:
        py_compile.compile("inference.py", doraise=True)
        print("  ✅ Compilation successful")
        return True
    except py_compile.PyCompileError as e:
        print(f"  ❌ Compilation failed: {e}")
        return False

if __name__ == "__main__":
    clear_all_caches()
    
    if verify_inference():
        recompile_inference()
        print("\n✅ Ready for submission!")
        print("   The fix is in place and caches are cleared.")
    else:
        print("\n❌ Fix not verified! Check inference.py manually.")
