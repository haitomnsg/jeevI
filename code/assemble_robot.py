"""
SpdrBot URDF Mesh Assembler
============================
This script combines all 13 STL mesh files from the URDF into a single
assembled STL file that can be imported into Fusion 360.

GOOD NEWS: The STL files exported by fusion2urdf are already positioned
in world coordinates — the visual origins in the URDF exactly compensate
for the joint chain transforms. So this script simply merges them as-is.

Usage:
  1. Install Python 3.8+ from https://www.python.org/downloads/
  2. pip install numpy-stl
  3. python assemble_robot.py

Output:
  - SpdrBot_assembled.stl  (single combined mesh, ready for Fusion 360)
"""

import os
import sys

try:
    from stl import mesh as stl_mesh
    import numpy as np
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Run:  pip install numpy-stl numpy")
    sys.exit(1)

# Path to the URDF meshes folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MESHES_DIR = os.path.join(SCRIPT_DIR, "spyderbot_minimal URDF", "SpdrBot_description", "meshes")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "SpdrBot_assembled.stl")

# All 13 mesh files in the URDF (already in world coordinates)
MESH_FILES = [
    "base_link.stl",
    "arm_a_1_1.stl", "arm_b_1_1.stl", "arm_c_1_1.stl",  # Leg 1 (FR)
    "arm_a_2_1.stl", "arm_b_2_1.stl", "arm_c_2_1.stl",  # Leg 2 (RL)
    "arm_a_3_1.stl", "arm_b_3_1.stl", "arm_c_3_1.stl",  # Leg 3 (RR)
    "arm_a_4_1.stl", "arm_b_4_1.stl", "arm_c_4_1.stl",  # Leg 4 (FL)
]


def combine_stl_files():
    """Load all STL mesh files and combine into a single mesh."""
    meshes = []
    total_triangles = 0

    print(f"Loading meshes from: {MESHES_DIR}\n")

    for filename in MESH_FILES:
        filepath = os.path.join(MESHES_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  WARNING: {filename} not found, skipping")
            continue

        m = stl_mesh.Mesh.from_file(filepath)
        num_triangles = len(m.vectors)
        total_triangles += num_triangles
        meshes.append(m)
        print(f"  Loaded: {filename:20s} ({num_triangles:,} triangles)")

    if not meshes:
        print("\nERROR: No mesh files found!")
        sys.exit(1)

    # Combine all meshes into one
    combined = stl_mesh.Mesh(np.zeros(total_triangles, dtype=stl_mesh.Mesh.dtype))
    offset = 0
    for m in meshes:
        n = len(m.vectors)
        combined.vectors[offset:offset + n] = m.vectors
        offset += n

    # Update normals
    combined.update_normals()

    return combined


def main():
    print("=" * 55)
    print("  SpdrBot URDF Mesh Assembler")
    print("=" * 55)
    print()

    combined = combine_stl_files()

    # Save combined mesh
    combined.save(OUTPUT_FILE)
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)

    print(f"\n  Combined mesh saved to: {OUTPUT_FILE}")
    print(f"  Total triangles: {len(combined.vectors):,}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print()
    print("Next steps:")
    print("  1. Open Fusion 360")
    print("  2. Insert > Insert Mesh > select 'SpdrBot_assembled.stl'")
    print("  3. Units: Millimeters")
    print("  4. Done! The robot will appear fully assembled.")
    print()
    print("To export a render/image from Fusion 360:")
    print("  - Design > Render workspace > Capture Image")


if __name__ == "__main__":
    main()
