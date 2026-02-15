# How to Get a Fully Assembled SpdrBot in Fusion 360

## What You Have

| Resource | Location | What It Contains |
|----------|----------|-----------------|
| **URDF Meshes** | `spyderbot_minimal URDF/SpdrBot_description/meshes/` | 13 STL files — **already positioned in world coordinates** |
| **URDF Definition** | `spyderbot_minimal URDF/SpdrBot_description/urdf/SpdrBot.xacro` | Joint/link hierarchy, transforms, physics properties |
| **USD Files** | `spdr.usd`, `spdr_stage.usd` | Binary USD for NVIDIA Isaac Sim (not directly usable in Fusion 360) |
| **Print Files** | `printFiles/` | Individual 3D-printable STL parts (**not positioned for assembly**) |

## Key Insight

The 13 STL meshes in the URDF `meshes/` folder were exported from Fusion 360 using the `fusion2urdf` plugin. **They are already positioned in world coordinates** — meaning their vertices are already at the correct assembled positions. You don't need to manually position anything!

### Robot Structure (4-legged spider)
```
base_link (body plate)
├── Leg 1 (FR): arm_a_1_1 → arm_b_1_1 → arm_c_1_1
├── Leg 2 (RL): arm_a_2_1 → arm_b_2_1 → arm_c_2_1
├── Leg 3 (RR): arm_a_3_1 → arm_b_3_1 → arm_c_3_1
└── Leg 4 (FL): arm_a_4_1 → arm_b_4_1 → arm_c_4_1
```
Each leg has 3 joints: hip (a), mid (b), tip (c) — 12 joints total.

---

## Option A: Import All 13 STLs Directly into Fusion 360 (Easiest)

1. **Open Fusion 360** → Create a new design
2. **Insert → Insert Mesh** (or drag & drop)
3. Navigate to: `spyderbot_minimal URDF/SpdrBot_description/meshes/`
4. Select **all 13 STL files** (Ctrl+A or select them all)
5. When prompted for units, choose **Millimeters**
6. Click **OK** — all parts will appear in the correct assembled positions
7. Each part becomes a separate Mesh Body in the Bodies folder

### To get a nice render:
- Switch to the **Render** workspace (top-left dropdown)
- Add materials/appearances if desired
- **Render → In-Canvas Render** or **Capture Image**

### To convert meshes to solid bodies (optional):
- Right-click each mesh body → **Mesh to BRep** (works for simpler meshes)
- Or use **Insert → Insert Mesh → then Edit Mesh** tools

---

## Option B: Use the Python Script to Create a Single Combined STL

If you want a single file instead of 13 separate bodies:

### Prerequisites
```powershell
# Install Python from https://www.python.org/downloads/
# Then install the required package:
pip install numpy-stl numpy
```

### Run the script
```powershell
cd "d:\jeevI\code"
python assemble_robot.py
```

This creates `SpdrBot_assembled.stl` — a single combined mesh file.

### Import into Fusion 360
1. **Insert → Insert Mesh** → select `SpdrBot_assembled.stl`
2. Units: **Millimeters**
3. Done!

---

## Option C: Use the USD Files (Requires NVIDIA Tools)

The `.usd` files are binary Universal Scene Description files for NVIDIA Isaac Sim. To use them:

1. **If you have NVIDIA Omniverse/Isaac Sim installed:**
   - Open `spdr_stage.usd` in Isaac Sim or Omniverse USD Composer
   - Export as FBX or OBJ: File → Export → choose format
   - Import the exported file into Fusion 360

2. **Using the free NVIDIA Omniverse Launcher:**
   - Download from https://www.nvidia.com/en-us/omniverse/
   - Install USD Composer (free)
   - Open the USD file and export to a Fusion-compatible format

---

## Option D: Buy the Original Fusion 360 Files

Per the project README, the original Fusion 360 source files are available as a paid download at:
https://indystry.cc/product/3d-printable-4-legged-spider-robot/

---

## FAQ

**Q: Why can't I just import the printFiles/ STLs?**
A: Those are individual parts for 3D printing — they're NOT positioned for assembly. They're centered at origin individually and would all overlap. The URDF meshes are the ones with correct world-frame positioning.

**Q: The mesh looks faceted/low-quality in Fusion 360?**
A: STL files are triangulated meshes, not smooth CAD surfaces. For higher quality, consider Option D (original Fusion 360 files) or use Fusion's mesh refinement tools.

**Q: Can I animate the joints in Fusion 360?**
A: Not directly from the STL import. You'd need to set up joints manually in Fusion 360's Assembly features, or use the URDF in a robotics simulator (Isaac Sim, Gazebo, etc.).

**Q: What's the scale?**
A: The STL meshes are in **millimeters**. The URDF uses a scale of 0.001 to convert to meters (standard for ROS/robotics). Fusion 360 works in mm natively, so just import with mm units.
