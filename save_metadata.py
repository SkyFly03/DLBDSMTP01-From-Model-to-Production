# save_metadata.py
# ----------------------------------------------------
# Scans the project folder and creates a metadata file
# with filename, relative path, size, and modification date.
# Saves output as project_metadata.json.
# ----------------------------------------------------

import os
import json
from datetime import datetime

# Project root directory
project_root = os.getcwd()
output_path = os.path.join(project_root, "project_metadata.json")

# Collect metadata for relevant file types
metadata = []
for root, _, files in os.walk(project_root):
    for fname in files:
        if fname.endswith((".py", ".pt", ".json", ".txt", ".md", ".zip")):
            full_path = os.path.join(root, fname)
            metadata.append({
                "file": fname,
                "relative_path": os.path.relpath(full_path, project_root),
                "size_kb": round(os.path.getsize(full_path) / 1024, 2),
                "last_modified": datetime.fromtimestamp(os.path.getmtime(full_path)).isoformat()
            })

# Write metadata to JSON file
with open(output_path, "w") as f:
    json.dump(metadata, f, indent=4)

print("project_metadata.json saved.")
