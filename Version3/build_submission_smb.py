"""
build_submission_smb.py
=======================
Ghép tất cả file trong smb/ thành 1 file submission duy nhất để nộp Kaggle.
Thứ tự ghép theo dependency: config -> movement -> garrison -> kinematics -> planner -> runtime

Cách dùng:
    python build_submission_smb.py [output_filename]
    
Mặc định output: submission_smb.py
"""
import re
import sys
import os

# File order theo dependency (từ base -> top)
MODULE_ORDER = [
    "config.py",
    "movement.py",
    "garrison.py",
    "kinematics.py",
    "planner.py",
    "runtime.py",
]

# Thư mục chứa package smb/
SMB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smb")

# Output filename
OUTPUT = sys.argv[1] if len(sys.argv) > 1 else "submission_smb.py"

def strip_relative_imports(code: str) -> str:
    """Xóa tất cả 'from .xxx import ...' blocks (bao gồm cả multi-line).
    Khi ghép thành 1 file, tất cả symbols đều nằm trong cùng namespace."""
    lines = code.split("\n")
    filtered = []
    in_relative_import = False
    for line in lines:
        stripped = line.strip()
        # Bắt đầu relative import block
        if re.match(r"^from \.\w+ import", stripped):
            if "(" in stripped and ")" not in stripped:
                # Multi-line import: from .xxx import (\n...\n)
                in_relative_import = True
            # Dòng đơn hoặc dòng kết thúc ngay: bỏ qua
            continue
        # Đang trong multi-line relative import block
        if in_relative_import:
            if ")" in stripped:
                in_relative_import = False
            continue
        filtered.append(line)
    return "\n".join(filtered)

def strip_duplicate_stdlib_imports(code: str, seen_imports: set) -> str:
    """Xóa các import stdlib đã xuất hiện ở module trước."""
    lines = code.split("\n")
    filtered = []
    for line in lines:
        stripped = line.strip()
        # Detect: 'from __future__ import annotations', 'import math', 'from dataclasses import dataclass', etc.
        if stripped.startswith("import ") or stripped.startswith("from "):
            # Bỏ relative imports (đã xử lý ở trên)
            if stripped.startswith("from ."):
                continue
            if stripped in seen_imports:
                continue
            seen_imports.add(stripped)
        filtered.append(line)
    return "\n".join(filtered)

def build():
    parts = []
    seen_imports = set()
    
    for module_name in MODULE_ORDER:
        filepath = os.path.join(SMB_DIR, module_name)
        if not os.path.exists(filepath):
            print(f"WARNING: {filepath} not found, skipping!")
            continue
        
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()
        
        # Bước 1: Xóa relative imports
        code = strip_relative_imports(code)
        
        # Bước 2: Xóa duplicate stdlib imports
        code = strip_duplicate_stdlib_imports(code, seen_imports)
        
        # Bước 3: Xóa blank lines ở đầu
        code = code.lstrip("\n")
        
        # Thêm separator comment
        parts.append(f"\n# {'='*60}")
        parts.append(f"# Module: {module_name}")
        parts.append(f"# {'='*60}\n")
        parts.append(code)
    
    # Ghép tất cả
    full_code = "\n".join(parts)
    
    # Xóa dòng trống liên tiếp (> 2 blank lines -> 2 blank lines)
    full_code = re.sub(r"\n{4,}", "\n\n\n", full_code)
    
    output_path = os.path.join(os.path.dirname(SMB_DIR), OUTPUT)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_code)
    
    # Statistics
    line_count = full_code.count("\n") + 1
    byte_count = len(full_code.encode("utf-8"))
    print(f"Build successful!")
    print(f"  Output: {output_path}")
    print(f"  Lines:  {line_count}")
    print(f"  Bytes:  {byte_count}")
    
    return output_path

if __name__ == "__main__":
    build()
