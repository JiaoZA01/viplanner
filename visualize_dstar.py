"""
Standalone visualizer for D* Lite perception and planning.
Runs without Isaac Sim — uses synthetic depth scenarios.

Usage:
    python visualize_dstar.py
"""

import math
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "d-star-lite")))
from d_star_lite import (
    computeShortestPath,
    initDStarLite,
    nextInShortestPath,
    scanForObstacles,
    stateNameToCoords,
)
from grid import GridWorld

# ---------------------------------------------------------------------------
# Grid config — must match get_d_star_path.py
# ---------------------------------------------------------------------------
X_DIM, Y_DIM = 40, 40
CELL_RES = 0.25  # meters per cell

# Depth camera intrinsics (matches viplanner_demo.py)
FX = 430.31607
CX = 428.28408
CY = 244.00695
IMG_H, IMG_W = 480, 856


# ---------------------------------------------------------------------------
# Synthetic depth image generators
# ---------------------------------------------------------------------------

def _empty_depth():
    """All depth values out of range — no obstacles detected."""
    return np.full((IMG_H, IMG_W), 15.0, dtype=np.float32)


def make_clear_path():
    """Open corridor — no obstacles."""
    return _empty_depth(), [0.0, 0.0, 5.0], "Scenario 1: Clear path"


def make_wall_with_gap():
    """Wall at 2 m across the full width, with a 1 m gap in the centre."""
    depth = _empty_depth()
    wall_z = 2.0
    gap_half_px = int(0.5 / wall_z * FX)       # ±0.5 m gap → pixel half-width
    gap_centre = int(CX)
    for v in range(IMG_H):
        for u in range(IMG_W):
            if abs(u - gap_centre) > gap_half_px:
                depth[v, u] = wall_z
    return depth, [0.0, 0.0, 4.0], "Scenario 2: Wall with centre gap"


def make_wall_left_open_right():
    """Wall on the left half, open on the right."""
    depth = _empty_depth()
    wall_z = 1.5
    for v in range(IMG_H):
        for u in range(0, int(CX)):
            depth[v, u] = wall_z
    return depth, [1.5, 0.0, 4.0], "Scenario 3: Left wall, open right"


def make_diagonal_wall():
    """Diagonal wall forcing the robot to route around."""
    depth = _empty_depth()
    for v in range(IMG_H):
        for u in range(IMG_W):
            # Wall sweeps from z=1m on the left to z=4m on the right
            wall_z = 1.0 + (u / IMG_W) * 3.0
            if wall_z < 9.0:
                depth[v, u] = wall_z
    return depth, [0.0, 0.0, 6.0], "Scenario 4: Diagonal wall"


def make_fully_blocked():
    """Obstacle fills the entire view — D* Lite should return no path."""
    depth = np.full((IMG_H, IMG_W), 0.5, dtype=np.float32)
    return depth, [0.0, 0.0, 5.0], "Scenario 5: Fully blocked"


# ---------------------------------------------------------------------------
# Core: depth → grid → D* Lite plan
# ---------------------------------------------------------------------------

def depth_to_grid(depth_np):
    graph = GridWorld(X_DIM, Y_DIM)
    step = 8
    for v in range(0, IMG_H, step):
        for u in range(0, IMG_W, step):
            z = float(depth_np[v, u])
            if z <= 0.1 or z > 9.0:
                continue
            x = (u - CX) * z / FX
            grid_x = int(x / CELL_RES + X_DIM / 2)
            grid_y = int(z / CELL_RES)
            if 0 <= grid_x < X_DIM and 0 <= grid_y < Y_DIM:
                graph.cells[grid_y][grid_x] = -1
    return graph


def run_dstar(graph, goal_cam):
    """
    Run D* Lite on the graph.
    goal_cam: [x, y, z] in camera frame (y ignored, z=forward, x=right)
    Returns (path_coords, goal_grid, start_grid) — all in grid index space.
    """
    gx = int(goal_cam[0] / CELL_RES + X_DIM / 2)
    gy = int(goal_cam[2] / CELL_RES)
    gx = max(0, min(X_DIM - 1, gx))
    gy = max(0, min(Y_DIM - 1, gy))

    sx = int(X_DIM / 2)
    sy = 0
    s_start = f"x{sx}y{sy}"
    s_goal  = f"x{gx}y{gy}"

    graph.setStart(s_start)
    graph.setGoal(s_goal)

    queue = []
    k_m = 0
    graph, queue, k_m = initDStarLite(graph, queue, s_start, s_goal, k_m)
    scanForObstacles(graph, queue, s_start, 100, k_m)
    computeShortestPath(graph, queue, s_start, k_m)

    path_coords = []
    curr = s_start
    for _ in range(200):
        if curr == s_goal:
            break
        try:
            curr = nextInShortestPath(graph, curr)
            path_coords.append(stateNameToCoords(curr))
        except Exception:
            break

    return path_coords, (gx, gy), (sx, sy)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_scenario(ax_grid, ax_metric, graph, path_coords, goal_grid, start_grid, title):
    """Draw one scenario into a pair of axes."""

    # --- Build obstacle image (grid index space) ---
    obstacle_img = np.zeros((Y_DIM, X_DIM, 3))
    obstacle_img[:, :] = [0.95, 0.95, 0.95]            # free = light grey
    cells = np.array(graph.cells)
    obstacle_img[cells < 0] = [0.15, 0.15, 0.15]       # obstacle = dark

    # Grid-index view
    ax_grid.imshow(
        obstacle_img, origin="lower",
        extent=[0, X_DIM, 0, Y_DIM], interpolation="nearest"
    )
    if path_coords:
        px = [p[0] for p in path_coords]
        py = [p[1] for p in path_coords]
        ax_grid.plot(px, py, color="royalblue", linewidth=2, zorder=3)
        ax_grid.plot(px, py, "o", color="royalblue", markersize=3, zorder=3)

    ax_grid.plot(*start_grid, "g^", markersize=10, label="Robot", zorder=5)
    ax_grid.plot(*goal_grid,  "r*", markersize=12, label="Goal",  zorder=5)
    ax_grid.set_title(title + "\n(grid cells)", fontsize=9)
    ax_grid.set_xlabel("Grid X  →  right")
    ax_grid.set_ylabel("Grid Y  →  forward")
    ax_grid.legend(fontsize=7, loc="upper right")
    # draw grid lines lightly
    ax_grid.set_xticks(range(0, X_DIM + 1, 5))
    ax_grid.set_yticks(range(0, Y_DIM + 1, 5))
    ax_grid.grid(True, color="white", linewidth=0.4)

    # Metric view (metres)
    ax_metric.set_facecolor("#f0f0f0")
    ax_metric.set_xlim(-X_DIM / 2 * CELL_RES, X_DIM / 2 * CELL_RES)
    ax_metric.set_ylim(-CELL_RES, Y_DIM * CELL_RES)

    for row in range(Y_DIM):
        for col in range(X_DIM):
            if graph.cells[row][col] < 0:
                wx = (col - X_DIM / 2) * CELL_RES
                wz = row * CELL_RES
                rect = mpatches.FancyBboxPatch(
                    (wx - CELL_RES / 2, wz - CELL_RES / 2),
                    CELL_RES, CELL_RES,
                    boxstyle="square,pad=0",
                    linewidth=0, facecolor="#222222",
                )
                ax_metric.add_patch(rect)

    if path_coords:
        mx = [(p[0] - X_DIM / 2) * CELL_RES for p in path_coords]
        mz = [p[1] * CELL_RES for p in path_coords]
        ax_metric.plot(mx, mz, color="royalblue", linewidth=2, label="D* path", zorder=3)
        ax_metric.plot(mx, mz, "o", color="royalblue", markersize=3, zorder=3)
    else:
        ax_metric.text(
            0, Y_DIM * CELL_RES / 2, "NO PATH FOUND",
            ha="center", va="center", fontsize=11,
            color="red", fontweight="bold"
        )

    goal_m_x = (goal_grid[0] - X_DIM / 2) * CELL_RES
    goal_m_z = goal_grid[1] * CELL_RES
    ax_metric.plot(0, 0, "g^", markersize=10, label="Robot (0, 0)", zorder=5)
    ax_metric.plot(goal_m_x, goal_m_z, "r*", markersize=12,
                   label=f"Goal ({goal_m_x:.1f} m, {goal_m_z:.1f} m)", zorder=5)

    ax_metric.axhline(0, color="grey", linewidth=0.5)
    ax_metric.axvline(0, color="grey", linewidth=0.5)
    ax_metric.set_title(title + "\n(metres)", fontsize=9)
    ax_metric.set_xlabel("X (m)  →  right")
    ax_metric.set_ylabel("Z (m)  →  forward")
    ax_metric.legend(fontsize=7, loc="upper right")
    ax_metric.grid(True, alpha=0.3)


def main():
    scenarios = [
        make_clear_path(),
        make_wall_with_gap(),
        make_wall_left_open_right(),
        make_diagonal_wall(),
        make_fully_blocked(),
    ]

    n = len(scenarios)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    fig.suptitle("D* Lite — Perception & Planning Visualizer", fontsize=13, fontweight="bold")

    for i, (depth_np, goal_cam, title) in enumerate(scenarios):
        print(f"\n[{title}]")
        graph = depth_to_grid(depth_np)
        path_coords, goal_grid, start_grid = run_dstar(graph, goal_cam)
        print(f"  Goal grid: {goal_grid}  |  Path length: {len(path_coords)} steps")
        if not path_coords:
            print("  → No path found (fully blocked or unreachable)")

        visualize_scenario(
            axes[i, 0], axes[i, 1],
            graph, path_coords, goal_grid, start_grid, title
        )

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "dstar_visualization.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
