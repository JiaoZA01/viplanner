# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the rigid objects class.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

# omni-isaac-lab
from omni.isaac.lab.app import AppLauncher



parser = argparse.ArgumentParser(description="This script demonstrates how to use the camera sensor.")
parser.add_argument("--conv_distance", default=0.2, type=float, help="Distance for a goal considered to be reached.")
parser.add_argument(
    "--scene", default="warehouse", choices=["matterport", "carla", "warehouse"], type=str, help="Scene to load."
)
parser.add_argument("--model_dir", default=None, type=str, help="Path to model directory.")

# add applauncher arguments
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import omni.isaac.core.utils.prims as prim_utils
import torch
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.viplanner.config import (
    ViPlannerCarlaCfg,
    ViPlannerMatterportCfg,
    ViPlannerWarehouseCfg,
)
from omni.viplanner.viplanner import VIPlannerAlgo
from pxr import UsdGeom

"""D start Lite"""
import get_d_star_path as d_star_module

"""
Main
"""




#`adding a helper to check if there is a collision
import numpy as np
import get_d_star_path # Import to access the memory map

def is_viplanner_path_colliding(path_tensor, clearance_radius=0.2):
    """
    Checks if the predicted viplanner path intersects with known obstacles.
    """
    path_pts = path_tensor.cpu().numpy()
    
    # Use the alias here
    world_memory = d_star_module._world_memory
    mem_res = d_star_module._MEM_RES
    
    for i in range(len(path_pts) - 1):
        p1 = path_pts[i][:2]
        p2 = path_pts[i+1][:2]
        
        dist = np.linalg.norm(p2 - p1)
        num_samples = max(2, int(dist / (mem_res / 2.0)))
        
        for t in np.linspace(0, 1, num_samples):
            pt = p1 + t * (p2 - p1)
            
            grid_radius = int(np.ceil(clearance_radius / mem_res))
            center_x = int(round(pt[0] / mem_res))
            center_y = int(round(pt[1] / mem_res))
            
            for dx in range(-grid_radius, grid_radius + 1):
                for dy in range(-grid_radius, grid_radius + 1):
                    if (center_x + dx, center_y + dy) in world_memory:
                        return True # Collision detected!
                        
    return False
    
    
    
    

def main():
    """Imports all legged robots supported in IsaacLab and applies zero actions."""

    # create environment cfg
    if args_cli.scene == "matterport":
        env_cfg = ViPlannerMatterportCfg(seed=1234)
        goal_pos = torch.tensor([8.0, -13.5, 1.0])
    elif args_cli.scene == "carla":
        env_cfg = ViPlannerCarlaCfg(seed=1234)

        
        goal_pos = torch.tensor([329, 347, 0.8])
            #the above is added so that if you input valid closer goals, we use it or
            #default it back to 120 335 1
    elif args_cli.scene == "warehouse":
        env_cfg = ViPlannerWarehouseCfg(seed=1234)
        goal_pos = torch.tensor([3, -4.5, 1.0])
    else:
        raise NotImplementedError(f"Scene {args_cli.scene} not yet supported!")

    # create environment
    env = ManagerBasedRLEnv(env_cfg)

    # adjust the intrinsics of the camera
    depth_intrinsic = torch.tensor([[430.31607, 0.0, 428.28408], [0.0, 430.31607, 244.00695], [0.0, 0.0, 1.0]])
    env.scene.sensors["depth_camera"].set_intrinsic_matrices(matrices=depth_intrinsic.repeat(env.num_envs, 1, 1))
    semantic_intrinsic = torch.tensor([[644.15496, 0.0, 639.53125], [0.0, 643.49212, 366.30880], [0.0, 0.0, 1.0]])
    env.scene.sensors["semantic_camera"].set_intrinsic_matrices(matrices=semantic_intrinsic.repeat(env.num_envs, 1, 1))

    # Make sure that groundplane is invisible
    if args_cli.scene == "carla":
        assert (
            prim_utils.get_prim_at_path("/World/GroundPlane").GetAttribute("visibility").Set(UsdGeom.Tokens.invisible)
        )

    # reset the environment
    with torch.inference_mode():
        obs = env.reset()[0]

    # set goal cube
    VisualCuboid(
        prim_path="/World/goal",  # The prim path of the cube in the USD stage
        name="waypoint",  # The unique name used to retrieve the object from the scene later on
        position=goal_pos,  # Using the current stage units which is in meters by default.
        scale=torch.tensor([0.15, 0.15, 0.15]),  # most arguments accept mainly numpy arrays.
        size=1.0,
        color=torch.tensor([1, 0, 0]),  # RGB channels, going from 0-1
    )
    goal_pos = prim_utils.get_prim_at_path("/World/goal").GetAttribute("xformOp:translate")

    # pause the simulator
    # env.sim.pause()

    # load viplanner
    viplanner = VIPlannerAlgo(model_dir=args_cli.model_dir, device=env.device)

    goals = torch.tensor(goal_pos.Get(), device=env.device).repeat(env.num_envs, 1)
    
    
    #modified so that with the even closer local goal, we print debugs:
    robot_start = env.scene["robot"].data.root_pos_w[0].clone()
    goal_start = goals[0].clone()
    start_dist = torch.norm(robot_start[:2] - goal_start[:2], p=2)

    print(f"[DEBUG] Robot start position: {robot_start.tolist()}")
    print(f"[DEBUG] Goal position:        {goal_start.tolist()}")
    print(f"[DEBUG] Start XY distance:    {float(start_dist):.3f}")
    
    
    
    
    # initial paths
    _, paths, fear = viplanner.plan_dual(
        obs["planner_image"]["depth_measurement"], obs["planner_image"]["semantic_measurement"], goals
    )
    
    # fear_print_counter = 0

    # [18744] Fear reaction tracking for stuck detection
    fear_buffer = 0
    buffer_size = 4  # Number of consecutive high-fear frames to trigger reaction
    is_fear_reaction = False
    replan_cnt = 0
    saved_d_lite_path = None

    # Simulate physics
    while simulation_app.is_running():
        with torch.inference_mode():
            # If simulation is paused, then skip.
            if not env.sim.is_playing():
                env.sim.step(render=~args_cli.headless)
                continue

            obs = env.step(action=paths.reshape(paths.shape[0], -1))[0]

        # apply planner
        goals = torch.tensor(goal_pos.Get(), device=env.device).repeat(env.num_envs, 1)
       
       
       
        curr_goal_dist = torch.norm(obs["planner_transform"]["cam_position"] - goals)
        max_goal_dist = viplanner.train_config.data_cfg[0].max_goal_distance

        if torch.any(curr_goal_dist > max_goal_dist):
            print(f"[WARNING] max_goal_distance = {max_goal_dist}")
            print(f"[WARNING] current goal distance = {curr_goal_dist}")
            print("[WARNING] Goal is too far from the camera/robot. Please select a nearer goal.")
            env.sim.pause()
            continue
       
        goal_cam_frame = viplanner.goal_transformer(
            goals, obs["planner_transform"]["cam_position"], obs["planner_transform"]["cam_orientation"]
        )

        # [18744] Get raw sensor data for D* Lite
        raw_depth = obs["planner_image"]["depth_measurement"]               # Shape: [Num_Envs, H, W]
        raw_cam_position = obs["planner_transform"]["cam_position"]         # Shape: [Num_Envs, 3]
        raw_cam_orientation = obs["planner_transform"]["cam_orientation"]   # Shape: [Num_Envs, 4]
        
        
        # [18744] Run D* Lite Planner
        # Using the first environment's data (index 0) for the demo
        # [18744] Run D* Lite Planner
        d_lite_path_cam = d_star_module.get_d_star_path(
            raw_depth[0], 
            goal_cam_frame[0], 
            depth_intrinsic, 
            raw_cam_position[0], 
            raw_cam_orientation[0]
        )

        # [18744] Run ViPlanner
        _, paths, fear = viplanner.plan_dual(
            obs["planner_image"]["depth_measurement"], obs["planner_image"]["semantic_measurement"], goal_cam_frame
        )
	
	
        # [18744] convert waypoints from the camera's frame into the world's coordinate frame
        paths = viplanner.path_transformer(
            paths, obs["planner_transform"]["cam_position"], obs["planner_transform"]["cam_orientation"]
        )
        num_waypoints = paths.shape[1]  # Save expected waypoint count for D* Lite resampling

        # [18744] Transform D* Lite path to world frame
        d_lite_path_world = viplanner.path_transformer(
            d_lite_path_cam.unsqueeze(0), raw_cam_position[0:1], raw_cam_orientation[0:1]
        )
        # Resample D* Lite path to match VIPlanner's fixed waypoint count
        if d_lite_path_world.shape[1] != num_waypoints:
            d_lite_path_world = torch.nn.functional.interpolate(
                d_lite_path_world.permute(0, 2, 1),  # [1, 3, N]
                size=num_waypoints,
                mode="linear",
                align_corners=True,
            ).permute(0, 2, 1)  # [1, num_waypoints, 3]
        
        fear_value = fear[0].item() if fear.numel() > 0 else 0.0
        
        # fear_print_counter += 1
        # if fear_print_counter % 5 == 0:
        #     print(f"[Fear] {fear_value:.4f}")



# --- PATH COLLISION CHECK ---
        viplanner_world_path = paths[0] 
        
        path_is_colliding = is_viplanner_path_colliding(
            viplanner_world_path, 
            clearance_radius=0.2
        )




        if fear_value > 0.5:
            fear_buffer = min(fear_buffer + 1, buffer_size + 1)
            print(f"[WARNING]: High fear detected: {fear_value:.3f} (buffer: {fear_buffer}/{buffer_size})")
        else:
            fear_buffer = max(fear_buffer - 1, 0)
        
        replan_cnt -= 1
        if fear_buffer >= buffer_size or path_is_colliding:
            if path_is_colliding:
                print("[COLLISION DETECTED]: ViPlanner path intersects obstacle! Switching to D* Lite.")
            elif not is_fear_reaction:
                print(f"[STUCK DETECTED]: Fear threshold exceeded! Switching to D* Lite.")
                
            is_fear_reaction = True
            

            if saved_d_lite_path is not None:
                saved_path_is_unsafe = is_viplanner_path_colliding(saved_d_lite_path[0], clearance_radius=0.0) 
                
                if saved_path_is_unsafe:
                    print("[WARNING]: Static D* Lite path is blocked! Forcing a replan...")
                    saved_d_lite_path = None  
                    replan_counter = 0        
                    
            if saved_d_lite_path is None or replan_counter <= 0:
                saved_d_lite_path = d_lite_path_world.clone()
                replan_counter = 60000  
                print(f"[FALLBACK]: Re-planned static D* Lite path. Cooldown reset to {replan_counter}.")
            
            # 3. Apply the saved static path
            paths = saved_d_lite_path
            
        else:
            if is_fear_reaction:
                print(f"[RECOVERY]: Clear path found, resuming neural network planner.")
                is_fear_reaction = False
                saved_d_lite_path = None
                replan_counter = 0


        
        # raw_semantic = obs["planner_image"]["semantic_measurement"]         # Shape: [Num_Envs, H, W]

        # [DEBUG] Print detected semantic IDs
        # unique_ids = torch.unique(raw_semantic)
        #print(f"[Sensors] Detected Semantic IDs: {unique_ids.tolist()}")

        # draw path
        viplanner.debug_draw(paths, fear, goals)
        #print(f"[Demo] Path End (World): {paths[0, 0, :2].cpu().numpy()}")

if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
