import math
import os
import sys
import torch
import time




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
_MEM_RES = 0.1  # must match CELL_RES in get_d_star_path
_last_grid_print_time = 0.0


def clear_memory():
    """Call this to reset the world map (e.g. new episode)."""
    global _world_memory
    _world_memory = {}

def print_world_memory_map():
    global _world_memory

    print("\n=== AGGREGATE WORLD MEMORY MAP ===")
    if len(_world_memory) == 0:
        print("empty")
        print("==================================\n")
        return

    xs = [p[0] for p in _world_memory.keys()]
    ys = [p[1] for p in _world_memory.keys()]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    for gy in range(max_y, min_y - 1, -1):
        row = []
        for gx in range(min_x, max_x + 1):
            if (gx, gy) in _world_memory:
                row.append("x")
            else:
                row.append("o")
        print("".join(row))

    print("==================================\n")

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
        # Skip obstacles that are now behind the robot (zc <= 0)
        if zc <= 0:
            continue
        # Camera frame → local grid index
        grid_x = int(xc / CELL_RES + X_DIM / 2)
        grid_y = int(zc / CELL_RES)
        if 0 <= grid_x < X_DIM and 0 <= grid_y < Y_DIM:
            INFLATION_RADIUS_M = 0.3
            inflation_cells = max(1, int(round(INFLATION_RADIUS_M / CELL_RES)))
            _inflate_obstacle_cells(graph.cells, grid_x, grid_y, inflation_cells)
    return graph
    
def _inflate_obstacle_cells(cells, cx, cy, radius_cells):
    y_dim = len(cells)
    x_dim = len(cells[0]) if y_dim > 0 else 0

    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            nx = cx + dx
            ny = cy + dy

            if 0 <= nx < x_dim and 0 <= ny < y_dim:
                # circular inflation
                if dx * dx + dy * dy <= radius_cells * radius_cells:
                    cells[ny][nx] = -1


def _extract_path(graph, s_start, s_goal, X_DIM, CELL_RES):
    """Run path extraction from s_start to s_goal, return list of [front, horizontal, height] points."""
    path_points = []
    curr = s_start
    for _ in range(100):
        if curr == s_goal:
            break
        try:
            curr = nextInShortestPath(graph, curr)
            coords = stateNameToCoords(curr)
            
            horizontal_y = -(coords[0] - X_DIM / 2) * CELL_RES  # Grid X maps to vehicle Y (horizontal)
            forward_x = coords[1] * CELL_RES                   # Grid Y maps to vehicle X (front)
            
            # Output in vehicle frame: [X (front), Y (horizontal), Z (height)]
            path_points.append([forward_x, horizontal_y, 0.0])
        except Exception:
            break
    return path_points

def _path_has_clearance(graph, path_points, X_DIM, CELL_RES, clearance_cells=2, check_n=None):
    if check_n is None:
        pts_to_check = path_points
    else:
        pts_to_check = path_points[:check_n]

    for pt in pts_to_check:
        # pt is [front, horizontal, height].
        gx = int(-pt[1] / CELL_RES + X_DIM / 2)  # pt[1] is horizontal -> maps to Grid X
        gy = int(pt[0] / CELL_RES)              # pt[0] is front -> maps to Grid Y

        # path point outside grid -> reject
        if gx < 0 or gx >= X_DIM or gy < 0 or gy >= len(graph.cells):
            return False

        if _is_near_obstacle(graph.cells, gx, gy, clearance_cells):
            return False

    return True

def _is_near_obstacle(cells, gx, gy, radius_cells):
    y_dim = len(cells)
    x_dim = len(cells[0]) if y_dim > 0 else 0

    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            nx = gx + dx
            ny = gy + dy
            if 0 <= nx < x_dim and 0 <= ny < y_dim:
                if cells[ny][nx] < 0:
                    return True
    return False


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

    X_DIM, Y_DIM = 30,30
    CELL_RES = 0.1  # 0.25 m/cell → 10 m x 10 m local grid

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
    fy, cy = intrinsics[1, 1].item(), intrinsics[1, 2].item()

    step = 8
    for v in range(0, H, step):
        for u in range(0, W, step):
            z = d_np[v, u]
            if z <= 0.2 or z > 9.0:
                continue

            # Prevent mapping the floor/ground as an obstacle:
            # Project vertical coordinate 'y' (+y is downwards in standard CV)
            y = (v - cy) * z / fy
            # Adjust 0.5 threshold based on your actual camera mounting height.
            if y > 0.5:  
                continue

            x = (u - cx) * z / fx
            # --- NEW: FILTER OUT THE GOAL CUBE ---
            # Assuming you applied the flipped X-axis fix (-goal_cam[1])
            goal_x = -goal_cam[1].item() 
            goal_z = goal_cam[0].item()  # cam0 is depth
            
            # Calculate distance from the current depth point to the goal center
            dist_to_goal = math.sqrt((x - goal_x)**2 + (z - goal_z)**2)
            
            # If the point is within 0.4 meters (cube size + margin), ignore it
            if dist_to_goal < 0.4:
                continue
            grid_x = int(x / CELL_RES + X_DIM / 2)
            grid_y = int(z / CELL_RES)
            if 0 <= grid_x < X_DIM and 0 <= grid_y < Y_DIM:
                INFLATION_RADIUS_M = 0.3   # example: 0.5 m
                inflation_cells = max(1, int(round(INFLATION_RADIUS_M / CELL_RES)))
                _inflate_obstacle_cells(graph.cells, grid_x, grid_y, inflation_cells)
                # Update persistent world memory
                if R is not None:
                    wx, wy = _cam_to_world_2d(x, z, cam_pos_xy, R)
                    mgx = math.floor(wx / _MEM_RES)
                    mgy = math.floor(wy / _MEM_RES)
                    _world_memory[(mgx, mgy)] = True
    
    # ------------------------------------------------------------------
    # DEBUG: print occupancy grid
    # 0  = free
    # -1 = obstacle / inflated obstacle
    # ------------------------------------------------------------------
    #global _last_grid_print_time
    #now = time.time()
    #if now - _last_grid_print_time >= 2.0:
    #    print("\n=== OCCUPANCY GRID ===")
        #print_world_memory_map()
        #for row in graph.cells:
        #    print("".join("x" if cell < 0 else "_" for cell in row))
    #    print("======================\n")
    #    _last_grid_print_time = now
    
    # ------------------------------------------------------------------
    # Handle goal behind robot
    # ------------------------------------------------------------------
    if goal_cam[0].item() < 0:  # cam0 is depth, < 0 correctly means behind
        turn_dir = 1.0 if goal_cam[1].item() >= 0 else -1.0  # cam1 is horizontal
        # Path format must be: [front(x), horizontal(y), height(z)]
        turn_path = [[CELL_RES, turn_dir * i * CELL_RES, 0.0] for i in range(1, 6)]
        #final_path = [[0.0, 0.0, 0.0]] + turn_path
        return torch.tensor(turn_path, dtype=torch.float32, device=depth_tensor.device)

    # ------------------------------------------------------------------
    # Setup start / goal in local grid
    # ------------------------------------------------------------------
    s_start = f"x{int(X_DIM / 2)}y0"
    raw_gx = int(-goal_cam[1].item() / CELL_RES+ Y_DIM / 2)  # cam1 is horizontal
    raw_gy = int(goal_cam[0].item() / CELL_RES )              # cam0 is depth (front). Starts at 0, no offset needed
    gx = max(0, min(X_DIM - 1, raw_gx))
    gy = max(0, min(Y_DIM - 1, raw_gy))
 
    s_goal = f"x{gx}y{gy}"
    
    # ------------------------------------------------------------------
    # DEBUG: print occupancy grid with ego + local goal + raw goal
    # ------------------------------------------------------------------
    global _last_grid_print_time
    now = time.time()
    if now - _last_grid_print_time >= 2.0:
        start_x = X_DIM // 2
        start_y = 0

        print("\n=== OCCUPANCY GRID ===")
        for y in range(Y_DIM):
            row_chars = []
            for x in range(X_DIM):
                if x == start_x and y == start_y:
                    row_chars.append("E")   # ego vehicle
                elif x == gx and y == gy:
                    row_chars.append("L")   # local clamped goal used by D* Lite
                elif x == raw_gx and y == raw_gy:
                    row_chars.append("R")   # raw projected goal before clamping
                elif graph.cells[y][x] < 0:
                    row_chars.append("x")   # obstacle
                else:
                    row_chars.append("_")   # free
            print("".join(row_chars))

        print(f"[Goal] raw projected goal cell: ({raw_gx}, {raw_gy})")
        print(f"[Goal] local clamped goal cell: ({gx}, {gy})")
        print(
            f"[Goal] goal_cam: "
            f"x={float(goal_cam[0]):.2f}, "
            f"y={float(goal_cam[1]):.2f}, "
            f"z={float(goal_cam[2]):.2f}"
        )

        if raw_gx < 0 or raw_gx >= X_DIM or raw_gy < 0 or raw_gy >= Y_DIM:
            print("[Goal] raw projected goal is outside the local occupancy grid")

        if cam_position is not None:
            print(
                f"[Ego] world position: "
                f"x={float(cam_position[0]):.2f}, "
                f"y={float(cam_position[1]):.2f}, "
                f"z={float(cam_position[2]):.2f}"
            )

        print("======================\n")
        
        print("\n=== GOAL DEBUG ===")
        print(f"goal_cam (meters): x={float(goal_cam[0]):.3f}, y={float(goal_cam[1]):.3f}, z={float(goal_cam[2]):.3f}")
        print(f"CELL_RES = {CELL_RES}, X_DIM = {X_DIM}, Y_DIM = {Y_DIM}")
        print(f"robot local grid center x = {X_DIM // 2}")
        print(f"raw_gx = int({float(goal_cam[0]):.3f} / {CELL_RES} + {X_DIM // 2}) = {raw_gx}")
        print(f"raw_gy = int({float(goal_cam[1]):.3f} / {CELL_RES}) = {raw_gy}")
        print(f"clamped gx, gy = ({gx}, {gy})")
        print(f"s_goal = {s_goal}")
        print("==============\n")
        _last_grid_print_time = now

    graph.setStart(s_start)
    graph.setGoal(s_goal)
    
    start_x = X_DIM // 2
    start_y = 0
    for cy in range(start_y - 1, start_y + 2):
        for cx in range(start_x - 1, start_x + 2):
            if 0 <= cx < X_DIM and 0 <= cy < Y_DIM:
                graph.cells[cy][cx] = 0

    # ------------------------------------------------------------------
    # Run D* Lite on current view
    # ------------------------------------------------------------------
    queue = []
    k_m = 0
    graph, queue, k_m = initDStarLite(graph, queue, s_start, s_goal, k_m)
    scanForObstacles(graph, queue, s_start, 100, k_m)
    computeShortestPath(graph, queue, s_start, k_m)

    path_points = _extract_path(graph, s_start, s_goal, X_DIM, CELL_RES)
    
    if path_points and not _path_has_clearance(graph, path_points, X_DIM, CELL_RES, clearance_cells=0, check_n=None):
        path_points = []

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

        path_points = _extract_path(mem_graph, s_start, s_goal, X_DIM, CELL_RES)
        
        if path_points and not _path_has_clearance(mem_graph, path_points, X_DIM, CELL_RES, clearance_cells=0, check_n=None):
            path_points = []
        if path_points:
            print(f"[D* Memory] Found path via memory ({len(path_points)} waypoints)")
        else:
            print("[D* Memory] Memory replan also failed — turning toward goal")

# ------------------------------------------------------------------
    # Last resort: turn in place toward goal side
    # ------------------------------------------------------------------
    if not path_points:
        turn_dir = 1.0 if goal_cam[1].item() >= 0 else -1.0  # cam1 is horizontal
        # Path format must be: [front(x), horizontal(y), height(z)]
        turn_path = [[CELL_RES, turn_dir * i * CELL_RES, 0.0] for i in range(1, 6)]
        path_points = turn_path
        
    # Prepend robot origin to path. This ensures the output path has at least length >= 2, 
    # which prevents PyTorch's linear interpolation from crashing in `viplanner_demo.py`.
    final_path = [[0.0, 0.0, 0.0]] + path_points
    if len(final_path) < 2:
        final_path.append(final_path[-1])

    return torch.tensor(final_path, dtype=torch.float32, device=depth_tensor.device)
