import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../d-star-lite")))
try:
    from grid import GridWorld
    from d_star_lite import initDStarLite, scanForObstacles, computeShortestPath, nextInShortestPath, stateNameToCoords
except ImportError:
    print("[WARNING] Could not import d-star-lite library. Please check the path.")


# ---------------------------------------------------------------------------
# Persistent world-frame obstacle memory
# Key: (world_grid_x, world_grid_y) — integer grid coords at _MEM_RES resolution
# Value: True (obstacle seen here)
# ---------------------------------------------------------------------------
_world_memory = {}
_MEM_RES = 0.25  # must match CELL_RES in get_d_star_path


def clear_memory():
    """Call this to reset the world map (e.g. new episode)."""
    global _world_memory
    _world_memory = {}


def _quat_to_rot_matrix(q):
    """
    Convert quaternion [w, x, y, z] to 3x3 rotation matrix (list of lists).
    Convention matches Isaac Sim / Isaac Lab.
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return [
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),   2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),   1 - 2*(x*x + y*y)],
    ]


def _cam_to_world_2d(xc, zc, cam_pos_xy, R):
    """
    Project a camera-frame point (xc, zc) to world (wx, wy), ignoring elevation.
    cam_pos_xy: [world_x, world_y]
    R: 3x3 rotation matrix (camera-to-world)
    """
    wx = R[0][0] * xc + R[0][2] * zc + cam_pos_xy[0]
    wy = R[1][0] * xc + R[1][2] * zc + cam_pos_xy[1]
    return wx, wy


def _world_to_cam_2d(wx, wy, cam_pos_xy, R):
    """
    Project a world point (wx, wy) back to camera frame (xc, zc).
    Uses R^T (inverse of rotation), ignoring elevation.
    """
    dx = wx - cam_pos_xy[0]
    dy = wy - cam_pos_xy[1]
    # R^T: row i of R^T is column i of R
    xc = R[0][0] * dx + R[1][0] * dy
    zc = R[0][2] * dx + R[1][2] * dy
    return xc, zc


def _build_grid_from_memory(cam_pos_xy, R, X_DIM, Y_DIM, CELL_RES):
    """
    Build a fresh local GridWorld (camera frame) populated from world memory.
    Unknown cells are treated as free (optimistic assumption).
    """
    graph = GridWorld(X_DIM, Y_DIM)
    for (mgx, mgy) in _world_memory:
        # World grid coords → world metric coords (cell center)
        wx = mgx * _MEM_RES + _MEM_RES * 0.5
        wy = mgy * _MEM_RES + _MEM_RES * 0.5
        # World → camera frame
        xc, zc = _world_to_cam_2d(wx, wy, cam_pos_xy, R)
        # Camera frame → local grid index
        grid_x = int(xc / CELL_RES + X_DIM / 2)
        grid_y = int(zc / CELL_RES)
        if 0 <= grid_x < X_DIM and 0 <= grid_y < Y_DIM:
            graph.cells[grid_y][grid_x] = -1
    return graph


def _extract_path(graph, s_start, s_goal, X_DIM, CELL_RES):
    """Run path extraction from s_start to s_goal, return list of [x, 0, z] points."""
    path_points = []
    curr = s_start
    for _ in range(100):
        if curr == s_goal:
            break
        try:
            curr = nextInShortestPath(graph, curr)
            coords = stateNameToCoords(curr)
            x_w = (coords[0] - X_DIM / 2) * CELL_RES
            z_w = coords[1] * CELL_RES
            path_points.append([x_w, 0.0, z_w])
        except Exception:
            break
    return path_points


def get_d_star_path(depth_tensor, goal_cam, intrinsics, cam_position=None, cam_orientation=None):
    """
    Generates a path using D* Lite based on depth sensor data.

    depth_tensor:    [H, W] tensor (GPU or CPU)
    goal_cam:        [3] tensor (x, y, z) in camera frame
    intrinsics:      [3, 3] tensor camera intrinsics
    cam_position:    [3] world position tensor (optional, enables memory)
    cam_orientation: [4] quaternion [w, x, y, z] tensor (optional, enables memory)
    """
    global _world_memory

    X_DIM, Y_DIM = 40, 40
    CELL_RES = 0.25  # 0.25 m/cell → 10 m x 10 m local grid

    # ------------------------------------------------------------------
    # Precompute rotation matrix and camera XY position for frame transforms
    # ------------------------------------------------------------------
    R = None
    cam_pos_xy = None
    if cam_position is not None and cam_orientation is not None:
        R = _quat_to_rot_matrix(cam_orientation)
        cam_pos_xy = [float(cam_position[0]), float(cam_position[1])]

    # ------------------------------------------------------------------
    # Build local grid from current depth image
    # ------------------------------------------------------------------
    graph = GridWorld(X_DIM, Y_DIM)
    d_np = depth_tensor.squeeze().cpu().numpy()
    H, W = d_np.shape[:2]
    fx, cx = intrinsics[0, 0].item(), intrinsics[0, 2].item()

    step = 8
    for v in range(0, H, step):
        for u in range(0, W, step):
            z = d_np[v, u]
            if z <= 0.1 or z > 9.0:
                continue
            x = (u - cx) * z / fx
            grid_x = int(x / CELL_RES + X_DIM / 2)
            grid_y = int(z / CELL_RES)
            if 0 <= grid_x < X_DIM and 0 <= grid_y < Y_DIM:
                graph.cells[grid_y][grid_x] = -1
                # Update persistent world memory
                if R is not None:
                    wx, wy = _cam_to_world_2d(x, z, cam_pos_xy, R)
                    mgx = int(wx / _MEM_RES)
                    mgy = int(wy / _MEM_RES)
                    _world_memory[(mgx, mgy)] = True

    # ------------------------------------------------------------------
    # Handle goal behind robot
    # ------------------------------------------------------------------
    if goal_cam[2].item() < 0:
        turn_dir = 1.0 if goal_cam[0].item() >= 0 else -1.0
        turn_path = [[turn_dir * i * CELL_RES, 0.0, CELL_RES] for i in range(1, 6)]
        return torch.tensor(turn_path, dtype=torch.float32, device=depth_tensor.device)

    # ------------------------------------------------------------------
    # Setup start / goal in local grid
    # ------------------------------------------------------------------
    s_start = f"x{int(X_DIM / 2)}y0"
    gx = int(goal_cam[0].item() / CELL_RES + X_DIM / 2)
    gy = int(goal_cam[2].item() / CELL_RES)
    gx = max(0, min(X_DIM - 1, gx))
    gy = max(0, min(Y_DIM - 1, gy))
    s_goal = f"x{gx}y{gy}"

    graph.setStart(s_start)
    graph.setGoal(s_goal)

    # ------------------------------------------------------------------
    # Run D* Lite on current view
    # ------------------------------------------------------------------
    queue = []
    k_m = 0
    graph, queue, k_m = initDStarLite(graph, queue, s_start, s_goal, k_m)
    scanForObstacles(graph, queue, s_start, 100, k_m)
    computeShortestPath(graph, queue, s_start, k_m)

    path_points = _extract_path(graph, s_start, s_goal, X_DIM, CELL_RES)

    # ------------------------------------------------------------------
    # Fallback: replan using world memory when current view is fully blocked
    # ------------------------------------------------------------------
    if not path_points and R is not None and len(_world_memory) > 0:
        print(f"[D* Memory] Current view fully blocked — replanning from {len(_world_memory)} remembered cells")
        mem_graph = _build_grid_from_memory(cam_pos_xy, R, X_DIM, Y_DIM, CELL_RES)
        mem_graph.setStart(s_start)
        mem_graph.setGoal(s_goal)

        mem_queue = []
        mem_k_m = 0
        mem_graph, mem_queue, mem_k_m = initDStarLite(mem_graph, mem_queue, s_start, s_goal, mem_k_m)
        scanForObstacles(mem_graph, mem_queue, s_start, 100, mem_k_m)
        computeShortestPath(mem_graph, mem_queue, s_start, mem_k_m)

        path_points = _extract_path(mem_graph, s_start, s_goal, X_DIM, CELL_RES, depth_tensor.device)
        if path_points:
            print(f"[D* Memory] Found path via memory ({len(path_points)} waypoints)")
        else:
            print("[D* Memory] Memory replan also failed — turning toward goal")

    # ------------------------------------------------------------------
    # Last resort: turn in place toward goal side
    # ------------------------------------------------------------------
    if not path_points:
        turn_dir = 1.0 if goal_cam[0].item() >= 0 else -1.0
        turn_path = [[turn_dir * i * CELL_RES, 0.0, CELL_RES] for i in range(1, 6)]
        return torch.tensor(turn_path, dtype=torch.float32, device=depth_tensor.device)

    return torch.tensor(path_points, device=depth_tensor.device)
