import torch
import os
from simulator import Simulator
from objective import Objective
from dynamics import JackalDynamics


import yaml
from tqdm import tqdm
import pdb
import numpy as np
abs_path = os.path.dirname(os.path.abspath(__file__))
CONFIG = yaml.safe_load(open(f"{abs_path}/config.yaml"))

sim_iter_per_mpc_call = int(CONFIG["control_dt"]//CONFIG["dt"])

#"delay_with_cvar_comp" #"delay_with_cvar_comp" # "no_delay" # "delay_with_mean_seq_comp" , "delay_with_cvar_comp" , "biased_mppi"
result_type = '/root/workspace/src/delay_with_cvar_comp_36000_ancillary_control_control_array.npy'
pdb.set_trace()
control_array =np.load(f'{result_type}') #np.load(f'{result_type}_control_array.npy')
# import sys
# def handle_exit_signal():
#     global init_ego_pose_array,goal_pose,obs_states
#     print("\nCtrl+C detected! Calling the exit function...")
#     # Place the logic you want to run here
#     result_type = 'cvar_mppi'
#     #np.save(f'results/{result_type}_current_state',current_state_array)
#     #np.save(f'results/{result_type}_predicted_state_array',predicted_state_array)
#     np.save(f'{result_type}_control_array',control_array)
#     # pdb.set_trace()
#     sys.exit()





import time

def run_point_robot_example():
    simulator = Simulator(
        cfg=CONFIG["simulator"],
        dt=CONFIG["dt"],
        goal=CONFIG["goal"],
        initial_pose=CONFIG["initial_actor_positions"],
        device=CONFIG["device"],
    )
    dynamics = JackalDynamics(
        dt=CONFIG["control_dt"], device=CONFIG["device"]
    )
    objective = Objective(goal=CONFIG["goal"], device=CONFIG["device"])
    
    

    initial_action = torch.zeros(2, device=CONFIG["device"])
    ob, *_ =  simulator._environment.step(initial_action.cpu().numpy())

    # simulator._environment.start_video_recording("jackal_nobias_sim.mp4")
    video_filename = f"{result_type}_jackal_simulation.mp4"
    simulator._environment.start_video_recording(video_filename)
    control_itr = 0
    for _ in tqdm(range(CONFIG["steps"])):
        ob_robot = ob["robot_0"]
        obst = ob["robot_0"]["FullSensor"]["obstacles"]
        observation_tensor = torch.tensor(
            [
                [*ob_robot["joint_state"]["position"],
                *ob_robot["joint_state"]["velocity"],
                *obst[3]['position'],
                *obst[3]['velocity']]
            ],
            device=CONFIG["device"],
        )
        action =control_array[control_itr].squeeze()
        #(ob,*_,) = simulator._environment.step(action.cpu().numpy())
        (ob,*_,) = simulator._environment.step(action)
        state = np.array(observation_tensor.to(dtype=torch.float32, device='cpu') ).squeeze()
        print("vx:",state[3],"vy:",state[4],"omega:",state[5])
        control_itr=control_itr+1
        time.sleep(0.05)
    simulator._environment.stop_video_recording()
if __name__ == "__main__":
    run_point_robot_example()
