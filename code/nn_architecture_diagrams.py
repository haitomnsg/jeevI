"""
Neural Network Architecture Diagrams for SpdrBot3 PPO Agent
============================================================
Generates multiple presentation-ready versions of the Actor-Critic architecture.

Architecture (from codebase):
  - Observation space: 48 dimensions
  - Action space: 12 dimensions (joint position targets)
  - Actor:  Input(48) → FC(64)+ELU → FC(64)+ELU → Output(12) + Gaussian noise (σ)
  - Critic: Input(48) → FC(64)+ELU → FC(64)+ELU → Output(1) [V(s)]
  - Algorithm: PPO (Proximal Policy Optimization)

Run:  python nn_architecture_diagrams.py
Output: Saves PNG files in the current directory.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Colour palette (professional, presentation-friendly) ────────────────────
C = {
    "bg":           "#FFFFFF",
    "actor_fill":   "#3B82F6",   # blue
    "actor_dark":   "#1E40AF",
    "critic_fill":  "#10B981",   # green
    "critic_dark":  "#047857",
    "input_fill":   "#F59E0B",   # amber
    "input_dark":   "#B45309",
    "output_fill":  "#EF4444",   # red
    "output_dark":  "#B91C1C",
    "hidden_fill":  "#8B5CF6",   # purple
    "hidden_dark":  "#5B21B6",
    "text_dark":    "#1F2937",
    "text_light":   "#FFFFFF",
    "arrow":        "#6B7280",
    "border":       "#374151",
    "env_fill":     "#F3F4F6",
    "env_border":   "#9CA3AF",
    "noise_fill":   "#EC4899",   # pink
    "ppo_fill":     "#6366F1",   # indigo
}

# ═══════════════════════════════════════════════════════════════════════════════
#  VERSION 1 — Clean Horizontal Block Diagram (Presentation Hero Slide)
# ═══════════════════════════════════════════════════════════════════════════════
def version1_horizontal_block():
    fig, ax = plt.subplots(figsize=(16, 6), dpi=200)
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-2.5, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])

    def draw_box(x, y, w, h, color, border_color, label, sublabel=None, fontsize=12):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                             facecolor=color, edgecolor=border_color, linewidth=2)
        ax.add_patch(box)
        if sublabel:
            ax.text(x + w/2, y + h/2 + 0.2, label, ha="center", va="center",
                    fontsize=fontsize, fontweight="bold", color=C["text_light"])
            ax.text(x + w/2, y + h/2 - 0.25, sublabel, ha="center", va="center",
                    fontsize=fontsize - 3, color=C["text_light"], style="italic")
        else:
            ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                    fontsize=fontsize, fontweight="bold", color=C["text_light"])

    def draw_arrow(x1, y1, x2, y2, label=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=C["arrow"], lw=2))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.25, label, ha="center", va="bottom",
                    fontsize=8, color=C["text_dark"])

    # Title
    ax.text(8, 5.2, "SpdrBot3 PPO Actor-Critic Architecture",
            ha="center", fontsize=18, fontweight="bold", color=C["text_dark"])

    # ── ACTOR path (top row, y≈3) ──
    y_a = 2.7
    # Input
    draw_box(0, y_a, 2.2, 1.2, C["input_fill"], C["input_dark"], "Observations", "48-dim", 11)
    # Hidden 1
    draw_box(3.2, y_a, 2.2, 1.2, C["hidden_fill"], C["hidden_dark"], "FC + ELU", "64 neurons", 11)
    # Hidden 2
    draw_box(6.4, y_a, 2.2, 1.2, C["hidden_fill"], C["hidden_dark"], "FC + ELU", "64 neurons", 11)
    # Output (mean)
    draw_box(9.6, y_a, 2.2, 1.2, C["actor_fill"], C["actor_dark"], "μ (mean)", "12-dim", 11)
    # Gaussian
    draw_box(12.6, y_a, 2.5, 1.2, C["output_fill"], C["output_dark"], "Actions", "12 joints", 11)

    # Noise sigma
    draw_box(10.4, y_a - 1.6, 1.6, 0.8, C["noise_fill"], "#BE185D", "σ (std)", fontsize=9)

    draw_arrow(2.2, y_a + 0.6, 3.2, y_a + 0.6)
    draw_arrow(5.4, y_a + 0.6, 6.4, y_a + 0.6)
    draw_arrow(8.6, y_a + 0.6, 9.6, y_a + 0.6)
    draw_arrow(11.8, y_a + 0.6, 12.6, y_a + 0.6, "sample")
    ax.annotate("", xy=(12.6, y_a + 0.3), xytext=(11.6, y_a - 0.6),
                arrowprops=dict(arrowstyle="-|>", color=C["arrow"], lw=1.5, ls="--"))

    ax.text(7.5, y_a + 1.55, "Actor Network (Policy π)", ha="center",
            fontsize=13, fontweight="bold", color=C["actor_dark"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#DBEAFE", edgecolor=C["actor_dark"], lw=1.5))

    # ── CRITIC path (bottom row, y≈0) ──
    y_c = 0.2
    draw_box(0, y_c, 2.2, 1.2, C["input_fill"], C["input_dark"], "Observations", "48-dim", 11)
    draw_box(3.2, y_c, 2.2, 1.2, C["hidden_fill"], C["hidden_dark"], "FC + ELU", "64 neurons", 11)
    draw_box(6.4, y_c, 2.2, 1.2, C["hidden_fill"], C["hidden_dark"], "FC + ELU", "64 neurons", 11)
    draw_box(9.6, y_c, 2.2, 1.2, C["critic_fill"], C["critic_dark"], "V(s)", "scalar", 11)

    draw_arrow(2.2, y_c + 0.6, 3.2, y_c + 0.6)
    draw_arrow(5.4, y_c + 0.6, 6.4, y_c + 0.6)
    draw_arrow(8.6, y_c + 0.6, 9.6, y_c + 0.6)

    ax.text(5.5, y_c + 1.55, "Critic Network (Value V)", ha="center",
            fontsize=13, fontweight="bold", color=C["critic_dark"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#D1FAE5", edgecolor=C["critic_dark"], lw=1.5))

    # ── Dashed separator ──
    ax.plot([-.3, 15.3], [2.1, 2.1], ls="--", color=C["env_border"], lw=1)

    plt.tight_layout()
    plt.savefig("nn_arch_v1_horizontal_block.png", dpi=200, bbox_inches="tight",
                facecolor=C["bg"], edgecolor="none")
    plt.close()
    print("✓ Saved nn_arch_v1_horizontal_block.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  VERSION 2 — Detailed Neuron-Level Diagram (vertical, shows individual nodes)
# ═══════════════════════════════════════════════════════════════════════════════
def version2_neuron_diagram():
    fig, axes = plt.subplots(1, 2, figsize=(18, 10), dpi=200)
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("SpdrBot3 PPO Neural Network — Neuron-Level View",
                 fontsize=18, fontweight="bold", color=C["text_dark"], y=0.97)

    def draw_network(ax, title, layer_sizes, layer_labels, colors, title_color):
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold", color=title_color, pad=15)

        n_layers = len(layer_sizes)
        max_neurons = max(layer_sizes)
        x_spacing = 3.0
        y_spacing = 0.55
        radius = 0.2

        positions = []
        for i, n in enumerate(layer_sizes):
            x = i * x_spacing
            y_start = -(n - 1) * y_spacing / 2
            layer_pos = [(x, y_start + j * y_spacing) for j in range(n)]
            positions.append(layer_pos)

        # Draw connections
        for i in range(n_layers - 1):
            n1 = min(layer_sizes[i], 10)
            n2 = min(layer_sizes[i + 1], 10)
            for p1 in positions[i][:n1]:
                for p2 in positions[i + 1][:n2]:
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                            color="#E5E7EB", lw=0.3, zorder=0)

        # Draw neurons
        for i, (layer_pos, n, label, color) in enumerate(
                zip(positions, layer_sizes, layer_labels, colors)):
            display_n = min(n, 8)
            for j, (x, y) in enumerate(layer_pos[:display_n]):
                circle = plt.Circle((x, y), radius, facecolor=color,
                                    edgecolor=C["border"], lw=1.2, zorder=2)
                ax.add_patch(circle)
            if n > 8:
                # show ellipsis
                x = layer_pos[0][0]
                y_bottom = layer_pos[display_n - 1][1] - 0.4
                ax.text(x, y_bottom, "⋮", ha="center", va="center",
                        fontsize=16, color=C["text_dark"], zorder=3)
                # Show total count
                ax.text(x, y_bottom - 0.5, f"({n} total)", ha="center",
                        va="center", fontsize=7, color=C["arrow"])

            # Layer label below
            x = layer_pos[0][0]
            y_top = layer_pos[0][1] + 0.6
            ax.text(x, y_top, label, ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=C["text_dark"])

        # Activation arrows
        for i in range(1, n_layers - 1):
            x = positions[i][0][0]
            y_bottom = positions[i][-1][1] if len(positions[i]) <= 8 else positions[i][7][1] - 1.2
            ax.text(x, y_bottom - 0.7, "ELU", ha="center", va="center",
                    fontsize=8, color=C["hidden_dark"], style="italic",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#EDE9FE",
                              edgecolor=C["hidden_dark"], lw=0.8))

        ax.autoscale()
        ax.set_xlim(ax.get_xlim()[0] - 0.5, ax.get_xlim()[1] + 0.5)
        ax.set_ylim(ax.get_ylim()[0] - 1.5, ax.get_ylim()[1] + 1.2)

    # Actor
    draw_network(axes[0], "Actor Network (Policy π)",
                 [48, 64, 64, 12],
                 ["Input\n(48-dim obs)", "Hidden 1\n(64)", "Hidden 2\n(64)", "Output\n(12 actions)"],
                 [C["input_fill"], C["hidden_fill"], C["hidden_fill"], C["actor_fill"]],
                 C["actor_dark"])

    # Critic
    draw_network(axes[1], "Critic Network (Value V)",
                 [48, 64, 64, 1],
                 ["Input\n(48-dim obs)", "Hidden 1\n(64)", "Hidden 2\n(64)", "Output\nV(s)"],
                 [C["input_fill"], C["hidden_fill"], C["hidden_fill"], C["critic_fill"]],
                 C["critic_dark"])

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("nn_arch_v2_neuron_level.png", dpi=200, bbox_inches="tight",
                facecolor=C["bg"], edgecolor="none")
    plt.close()
    print("✓ Saved nn_arch_v2_neuron_level.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  VERSION 3 — Minimalist Single-Column (good for vertical slide / poster)
# ═══════════════════════════════════════════════════════════════════════════════
def version3_minimalist():
    fig, ax = plt.subplots(figsize=(7, 12), dpi=200)
    ax.set_xlim(0, 7)
    ax.set_ylim(-0.5, 13)
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])

    def rbox(x, y, w, h, fc, ec, label, sub=None, fs=12):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                             facecolor=fc, edgecolor=ec, linewidth=2, zorder=2)
        ax.add_patch(box)
        if sub:
            ax.text(x + w/2, y + h/2 + 0.15, label, ha="center", va="center",
                    fontsize=fs, fontweight="bold", color=C["text_light"], zorder=3)
            ax.text(x + w/2, y + h/2 - 0.2, sub, ha="center", va="center",
                    fontsize=fs - 3, color=C["text_light"], style="italic", zorder=3)
        else:
            ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                    fontsize=fs, fontweight="bold", color=C["text_light"], zorder=3)

    def varrow(x, y1, y2, label=None):
        ax.annotate("", xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle="-|>", color=C["arrow"], lw=2))
        if label:
            ax.text(x + 0.15, (y1 + y2) / 2, label, ha="left", va="center",
                    fontsize=8, color=C["text_dark"])

    ax.text(3.5, 12.6, "Actor Network", ha="center",
            fontsize=16, fontweight="bold", color=C["actor_dark"])

    # Observation breakdown
    obs_items = [
        ("Linear Velocity", "3"),
        ("Angular Velocity", "3"),
        ("Projected Gravity", "3"),
        ("Commands (vx, vy, ωz)", "3"),
        ("Joint Position Error", "12"),
        ("Joint Velocities", "12"),
        ("Previous Actions", "12"),
    ]
    obs_y = 11.8
    ax.text(3.5, obs_y, "Observation Vector (48-dim)", ha="center",
            fontsize=10, fontweight="bold", color=C["input_dark"])
    for i, (name, dim) in enumerate(obs_items):
        y = obs_y - 0.35 * (i + 1)
        ax.text(1.2, y, f"• {name}", fontsize=7.5, color=C["text_dark"], va="center")
        ax.text(5.8, y, f"[{dim}]", fontsize=7.5, color=C["arrow"], va="center", ha="right")

    # Input box
    rbox(1, 8.3, 5, 0.85, C["input_fill"], C["input_dark"], "Input Layer", "48 neurons", 11)
    varrow(3.5, 8.3, 7.6)

    # Hidden 1
    rbox(1, 6.7, 5, 0.85, C["hidden_fill"], C["hidden_dark"], "Hidden Layer 1", "64 neurons · ELU", 11)
    varrow(3.5, 6.7, 6.05)

    # Hidden 2
    rbox(1, 5.1, 5, 0.85, C["hidden_fill"], C["hidden_dark"], "Hidden Layer 2", "64 neurons · ELU", 11)
    varrow(3.5, 5.1, 4.45)

    # Output
    rbox(1, 3.5, 5, 0.85, C["actor_fill"], C["actor_dark"], "Output (μ)", "12 action means", 11)
    varrow(3.5, 3.5, 2.85)

    # Gaussian sampling
    rbox(1, 1.9, 5, 0.85, C["output_fill"], C["output_dark"],
         "Gaussian Sampling", "μ + σ·ε  →  12 joint targets", 10)

    # σ annotation
    rbox(0.1, 3.0, 0.7, 0.5, C["noise_fill"], "#BE185D", "σ", fs=9)
    ax.annotate("", xy=(1.0, 2.5), xytext=(0.8, 3.0),
                arrowprops=dict(arrowstyle="-|>", color="#BE185D", lw=1.5, ls="--"))

    # Separator
    ax.plot([0.5, 6.5], [1.3, 1.3], ls="--", color=C["env_border"], lw=1)
    ax.text(3.5, 1.05, "Critic Network (same hidden structure)", ha="center",
            fontsize=10, fontweight="bold", color=C["critic_dark"])
    rbox(1, 0.1, 5, 0.75, C["critic_fill"], C["critic_dark"], "Value V(s)", "scalar output", 10)

    plt.tight_layout()
    plt.savefig("nn_arch_v3_minimalist.png", dpi=200, bbox_inches="tight",
                facecolor=C["bg"], edgecolor="none")
    plt.close()
    print("✓ Saved nn_arch_v3_minimalist.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  VERSION 4 — Full PPO Training Loop System Diagram
# ═══════════════════════════════════════════════════════════════════════════════
def version4_ppo_system():
    fig, ax = plt.subplots(figsize=(18, 9), dpi=200)
    ax.set_xlim(-1, 19)
    ax.set_ylim(-1, 9.5)
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])

    def rbox(x, y, w, h, fc, ec, lines, fs=11):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                             facecolor=fc, edgecolor=ec, linewidth=2, zorder=2)
        ax.add_patch(box)
        n = len(lines)
        for i, (text, weight, size) in enumerate(lines):
            offset = (n - 1) / 2 - i
            ax.text(x + w/2, y + h/2 + offset * 0.3, text, ha="center", va="center",
                    fontsize=size, fontweight=weight, color=C["text_light"], zorder=3)

    def arrow(x1, y1, x2, y2, label=None, curved=False, color=C["arrow"]):
        style = "arc3,rad=0.2" if curved else None
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2,
                                    connectionstyle=style))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.2, label, ha="center", va="bottom",
                    fontsize=8, color=C["text_dark"],
                    bbox=dict(boxstyle="round,pad=0.15", fc="#F9FAFB", ec="#D1D5DB", lw=0.5))

    ax.text(9, 9.2, "SpdrBot3 — PPO Training Loop & Architecture",
            ha="center", fontsize=18, fontweight="bold", color=C["text_dark"])

    # Environment
    rbox(0, 3, 3.5, 2.5, "#1E293B", "#0F172A", [
        ("Isaac Lab", "bold", 12),
        ("Simulation", "normal", 10),
        ("Environment", "normal", 10),
        ("(4096 parallel envs)", "normal", 8),
    ])

    # Observation breakdown box
    rbox(4.5, 5.5, 3.5, 2.8, C["input_fill"], C["input_dark"], [
        ("Observations (48)", "bold", 11),
        ("───────────", "normal", 6),
        ("lin_vel [3]  ang_vel [3]", "normal", 7),
        ("gravity [3]  cmd [3]", "normal", 7),
        ("joint_pos_err [12]", "normal", 7),
        ("joint_vel [12]", "normal", 7),
        ("prev_actions [12]", "normal", 7),
    ])

    # Actor box
    rbox(9, 5.5, 3.5, 2.8, C["actor_fill"], C["actor_dark"], [
        ("Actor (Policy π)", "bold", 12),
        ("───────────", "normal", 6),
        ("FC(48→64) + ELU", "normal", 9),
        ("FC(64→64) + ELU", "normal", 9),
        ("FC(64→12) → μ", "normal", 9),
        ("+ σ (learnable)", "normal", 8),
    ])

    # Actions box
    rbox(13.5, 5.5, 3, 2.8, C["output_fill"], C["output_dark"], [
        ("Actions (12)", "bold", 11),
        ("───────────", "normal", 6),
        ("Joint Position", "normal", 9),
        ("Targets", "normal", 9),
        ("(12 DOF)", "normal", 8),
    ])

    # Critic box
    rbox(9, 1.0, 3.5, 2.8, C["critic_fill"], C["critic_dark"], [
        ("Critic (Value V)", "bold", 12),
        ("───────────", "normal", 6),
        ("FC(48→64) + ELU", "normal", 9),
        ("FC(64→64) + ELU", "normal", 9),
        ("FC(64→1) → V(s)", "normal", 9),
    ])

    # PPO Update box
    rbox(14.5, 1.0, 3.5, 2.8, C["ppo_fill"], "#4338CA", [
        ("PPO Update", "bold", 12),
        ("───────────", "normal", 6),
        ("Clip ratio ε=0.2", "normal", 8),
        ("GAE (γ=0.99, λ=0.95)", "normal", 8),
        ("LR=1e-4 (adaptive)", "normal", 8),
        ("5 epochs, 4 mini-batch", "normal", 8),
    ])

    # Reward
    rbox(4.5, 1.0, 3.5, 2.0, "#F59E0B", C["input_dark"], [
        ("Rewards", "bold", 11),
        ("(multi-term)", "normal", 9),
    ])

    # Arrows  — main flow
    arrow(3.5, 4.8, 4.5, 6.9, "obs (48)")
    arrow(8.0, 6.9, 9.0, 6.9, "")
    arrow(12.5, 6.9, 13.5, 6.9, "μ + σε")
    arrow(15.0, 5.5, 2.5, 5.5, "actions (12)")

    # Observations → critic
    arrow(6.25, 5.5, 9.0, 3.4, "obs", curved=True)

    # Env → Reward
    arrow(2.5, 3.0, 4.5, 2.0, "reward")

    # Critic → PPO
    arrow(12.5, 2.4, 14.5, 2.4, "V(s)")
    # Reward → PPO
    arrow(8.0, 2.0, 14.5, 2.0, "r")

    # PPO → both networks (gradient)
    arrow(16.25, 3.8, 10.75, 5.5, "∇ update", curved=True, color=C["ppo_fill"])
    arrow(14.5, 2.0, 12.5, 2.0, "∇ update", color=C["ppo_fill"])

    plt.tight_layout()
    plt.savefig("nn_arch_v4_ppo_system.png", dpi=200, bbox_inches="tight",
                facecolor=C["bg"], edgecolor="none")
    plt.close()
    print("✓ Saved nn_arch_v4_ppo_system.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  VERSION 5 — Dark-Theme Sleek Diagram (modern presentation style)
# ═══════════════════════════════════════════════════════════════════════════════
def version5_dark_theme():
    D = {
        "bg": "#0F172A",
        "text": "#F8FAFC",
        "text_dim": "#94A3B8",
        "accent1": "#38BDF8",  # sky-400
        "accent2": "#A78BFA",  # violet-400
        "accent3": "#34D399",  # emerald-400
        "accent4": "#FB923C",  # orange-400
        "accent5": "#F472B6",  # pink-400
        "border": "#475569",
        "card": "#1E293B",
    }

    fig, ax = plt.subplots(figsize=(16, 8), dpi=200)
    ax.set_xlim(-0.5, 16)
    ax.set_ylim(-1, 8.5)
    ax.axis("off")
    fig.patch.set_facecolor(D["bg"])

    def card(x, y, w, h, accent, label, sub, fs=12):
        # background card
        bg = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                            facecolor=D["card"], edgecolor=accent, linewidth=2.5, zorder=2)
        ax.add_patch(bg)
        # accent bar at top
        bar = FancyBboxPatch((x + 0.05, y + h - 0.35), w - 0.1, 0.30,
                             boxstyle="round,pad=0.05",
                             facecolor=accent, edgecolor="none", alpha=0.9, zorder=3)
        ax.add_patch(bar)
        ax.text(x + w/2, y + h - 0.2, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=D["bg"], zorder=4)
        ax.text(x + w/2, y + h/2 - 0.15, sub, ha="center", va="center",
                fontsize=fs - 3, color=D["text_dim"], zorder=4)

    def glow_arrow(x1, y1, x2, y2, color, label=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.25, label, ha="center", va="bottom",
                    fontsize=8, color=color, fontweight="bold")

    ax.text(8, 8.1, "SpdrBot3 — Neural Network Architecture",
            ha="center", fontsize=19, fontweight="bold", color=D["text"])
    ax.text(8, 7.6, "PPO Actor-Critic  |  RSL-RL  |  Isaac Lab",
            ha="center", fontsize=10, color=D["text_dim"])

    # ═══ ACTOR ROW (y≈5)
    y_a = 4.8
    card(0, y_a, 2.5, 1.5, D["accent4"], "Observation", "48-dimensional\nstate vector")
    card(3.5, y_a, 2.5, 1.5, D["accent2"], "Hidden 1", "FC(48→64)\nELU activation")
    card(7, y_a, 2.5, 1.5, D["accent2"], "Hidden 2", "FC(64→64)\nELU activation")
    card(10.5, y_a, 2.5, 1.5, D["accent1"], "Output μ", "FC(64→12)\n12 action means")
    card(13.5, y_a, 2.3, 1.5, D["accent5"], "Actions", "Gaussian\nN(μ, σ²)")

    glow_arrow(2.5, y_a + 0.75, 3.5, y_a + 0.75, D["accent4"])
    glow_arrow(6.0, y_a + 0.75, 7.0, y_a + 0.75, D["accent2"])
    glow_arrow(9.5, y_a + 0.75, 10.5, y_a + 0.75, D["accent2"])
    glow_arrow(13.0, y_a + 0.75, 13.5, y_a + 0.75, D["accent1"], "sample")

    ax.text(8, y_a + 2.0, "ACTOR  (Policy Network)", ha="center",
            fontsize=13, fontweight="bold", color=D["accent1"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=D["card"],
                      edgecolor=D["accent1"], lw=1.5))

    # ═══ CRITIC ROW (y≈1.5)
    y_c = 1.5
    card(0, y_c, 2.5, 1.5, D["accent4"], "Observation", "48-dimensional\nstate vector")
    card(3.5, y_c, 2.5, 1.5, D["accent2"], "Hidden 1", "FC(48→64)\nELU activation")
    card(7, y_c, 2.5, 1.5, D["accent2"], "Hidden 2", "FC(64→64)\nELU activation")
    card(10.5, y_c, 2.5, 1.5, D["accent3"], "Value V(s)", "FC(64→1)\nscalar estimate")

    glow_arrow(2.5, y_c + 0.75, 3.5, y_c + 0.75, D["accent4"])
    glow_arrow(6.0, y_c + 0.75, 7.0, y_c + 0.75, D["accent2"])
    glow_arrow(9.5, y_c + 0.75, 10.5, y_c + 0.75, D["accent2"])

    ax.text(6.75, y_c + 2.0, "CRITIC  (Value Network)", ha="center",
            fontsize=13, fontweight="bold", color=D["accent3"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=D["card"],
                      edgecolor=D["accent3"], lw=1.5))

    # ── separator
    ax.plot([-.3, 15.8], [4.3, 4.3], ls="--", color=D["border"], lw=1)

    # ── Parameter summary
    params = [
        "Total Actor Params: 48×64 + 64 + 64×64 + 64 + 64×12 + 12 = 7,436",
        "Total Critic Params: 48×64 + 64 + 64×64 + 64 + 64×1 + 1 = 7,361",
    ]
    for i, p in enumerate(params):
        ax.text(8, 0.6 - i * 0.35, p, ha="center", fontsize=8, color=D["text_dim"])

    plt.tight_layout()
    plt.savefig("nn_arch_v5_dark_theme.png", dpi=200, bbox_inches="tight",
                facecolor=D["bg"], edgecolor="none")
    plt.close()
    print("✓ Saved nn_arch_v5_dark_theme.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  VERSION 6 — Transfer Learning Pipeline (shows 3-stage training)
# ═══════════════════════════════════════════════════════════════════════════════
def version6_transfer_pipeline():
    fig, ax = plt.subplots(figsize=(17, 6), dpi=200)
    ax.set_xlim(-0.5, 17)
    ax.set_ylim(-0.5, 6)
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])

    ax.text(8.25, 5.6, "SpdrBot3 — Transfer Learning Pipeline",
            ha="center", fontsize=17, fontweight="bold", color=C["text_dark"])

    stages = [
        {
            "x": 0, "color": "#3B82F6", "border": "#1E40AF",
            "title": "Stage 1: Flat Terrain",
            "sub": "500 iterations · LR=1e-4",
            "net": "48 → [64·ELU] → [64·ELU] → 12",
            "file": "model_499.pt",
            "desc": "Learn basic\nlocomotion gait"
        },
        {
            "x": 5.8, "color": "#10B981", "border": "#047857",
            "title": "Stage 2: Rough Terrain",
            "sub": "98 iters (fine-tune) · LR=1e-4",
            "net": "48 → [64·ELU] → [64·ELU] → 12",
            "file": "model_598.pt",
            "desc": "Adapt to uneven\nterrain + perturbations"
        },
        {
            "x": 11.6, "color": "#F59E0B", "border": "#B45309",
            "title": "Stage 3: Box Obstacles",
            "sub": "98 iters (fine-tune) · LR=1e-4",
            "net": "48 → [64·ELU] → [64·ELU] → 12",
            "file": "model_598.pt",
            "desc": "Navigate over\nbox obstacles"
        },
    ]

    for i, s in enumerate(stages):
        x = s["x"]
        # main card
        box = FancyBboxPatch((x, 0.5), 5, 4.2, boxstyle="round,pad=0.2",
                             facecolor="#F9FAFB", edgecolor=s["border"], linewidth=2.5)
        ax.add_patch(box)

        # header bar
        hdr = FancyBboxPatch((x + 0.05, 3.85), 4.9, 0.8, boxstyle="round,pad=0.1",
                             facecolor=s["color"], edgecolor="none")
        ax.add_patch(hdr)
        ax.text(x + 2.5, 4.25, s["title"], ha="center", va="center",
                fontsize=11, fontweight="bold", color=C["text_light"])

        # Content
        ax.text(x + 2.5, 3.4, s["sub"], ha="center", fontsize=8, color=C["arrow"])

        # Network diagram mini
        net_box = FancyBboxPatch((x + 0.3, 2.3), 4.4, 0.8, boxstyle="round,pad=0.1",
                                 facecolor=s["color"], edgecolor=s["border"],
                                 alpha=0.15, lw=1)
        ax.add_patch(net_box)
        ax.text(x + 2.5, 2.7, s["net"], ha="center", fontsize=9,
                fontweight="bold", color=s["border"], family="monospace")

        ax.text(x + 2.5, 1.7, s["desc"], ha="center", fontsize=9, color=C["text_dark"])
        ax.text(x + 2.5, 0.85, s["file"], ha="center", fontsize=8,
                color=C["arrow"], family="monospace",
                bbox=dict(boxstyle="round,pad=0.15", fc="#F3F4F6", ec=C["env_border"]))

        # Transfer arrows
        if i < len(stages) - 1:
            nx = stages[i + 1]["x"]
            ax.annotate("", xy=(nx, 2.6), xytext=(x + 5, 2.6),
                        arrowprops=dict(arrowstyle="-|>", color=s["border"],
                                        lw=3, ls="-"))
            ax.text((x + 5 + nx) / 2, 2.95, "Transfer\nWeights",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color=s["border"])

    # Architecture badge
    ax.text(8.25, 0.0, "Same architecture [64, 64] across all stages enables direct weight transfer",
            ha="center", fontsize=9, style="italic", color=C["arrow"])

    plt.tight_layout()
    plt.savefig("nn_arch_v6_transfer_pipeline.png", dpi=200, bbox_inches="tight",
                facecolor=C["bg"], edgecolor="none")
    plt.close()
    print("✓ Saved nn_arch_v6_transfer_pipeline.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating SpdrBot3 Neural Network Architecture Diagrams...")
    print("=" * 60)
    version1_horizontal_block()
    version2_neuron_diagram()
    version3_minimalist()
    version4_ppo_system()
    version5_dark_theme()
    version6_transfer_pipeline()
    print("=" * 60)
    print("Done! 6 diagram versions saved as PNG files.")
    print("\nVersions:")
    print("  v1  Horizontal block diagram   — clean hero slide")
    print("  v2  Neuron-level diagram        — detailed technical view")
    print("  v3  Minimalist vertical         — poster / single column")
    print("  v4  PPO system diagram          — full training loop")
    print("  v5  Dark theme                  — modern presentation")
    print("  v6  Transfer learning pipeline  — 3-stage training flow")
