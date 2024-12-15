import torch
import os
from simulator import Simulator
from objective import Objective
from dynamics import JackalDynamics
from mppi_torch.mppi import MPPIPlanner
from mppi_jax.cvar_and_biased_mppiV2 import MPPI
import yaml
from tqdm import tqdm
import pdb
import numpy as np
import numpy as np
import copy
from delay_models.simple_delay_models import turncated_normal_delay , constant_delay,MultimodalTruncatedNormalDelay

mus = np.array([0.05, 0.3])  # Means for two modes: low latency and high latency
sigmas = np.array([0.04, 0.6])  # Standard deviations
low = 0.0  # Lower bound for truncation
high = 1.3  # Upper bound for truncation
weights = np.array([0.6, 0.4])  # Weights for each mode
MEAN_DELAY =  weights[0]*mus[0] + weights[1]*mus[1]
SIGMA_DELAY =  0.4
total_delay  = MultimodalTruncatedNormalDelay(mus, sigmas, low, high, weights)
# Create the multimodal distribution
# MEAN_DELAY = 0.2
# total_delay  = turncated_normal_delay(mu=0.2 ,sigma=0.2,low=0.05, high=1.0) #constant_delay(delay=0.0) # turncated_normal_delay(mu=0.2 ,sigma=0.5,low=0.05, high=1.0) # constant_delay(delay=0.8) #MultimodalTruncatedNormalDelay(mus, sigmas, low, high, weights)


# total_delay = turncated_normal_delay(mu=MEAN_DELAY ,sigma=1,low=0.05, high=2) #constant_delay(delay=0.0) #turncated_normal_delay(mu=0.30 ,sigma=0.2,low=0.05, high=0.8)   #constant_delay(delay=0.0) #turncated_normal_delay(mu=0.35 ,sigma=0.35,low=0.0, high=1.0) #constant_delay(delay=0.0) #turncated_normal_delay(mu=0.35 ,sigma=0.35,low=0.0, high=1.0) #constant_delay(delay=0.0) #turncated_normal_delay(mu=0.30  ,sigma=0.2,low=0.0, high=0.8) #constant_delay(delay=0.0) #turncated_normal_delay(mu=0.30  ,sigma=0.2,low=0.0, high=0.8) #constant_delay(delay=0.0) constant_delay(delay=0.0) #
experiment_type  ="delay_with_cvar_comp" #"biased_mppi_with_shifted_control" #"delay_with_cvar_comp" #"biased_mppi" , "delay_with_cvar_comp" #"delay_with_cvar_comp" # "no_delay" # "delay_with_mean_seq_comp" , "delay_with_cvar_comp"
abs_path = os.path.dirname(os.path.abspath(__file__))
CONFIG = yaml.safe_load(open(f"{abs_path}/config.yaml"))

sim_iter_per_mpc_call = int(CONFIG["control_dt"]//CONFIG["dt"])


delay_seq =np.empty(shape=(0,1))
control_array = np.empty((0,2))
import sys
def handle_exit_signal():
    global control_array , experiment_type,delay_seq
    print("\nCtrl+C detected! Calling the exit function...")
    # Place the logic you want to run here
    #np.save(f'results/{result_type}_current_state',current_state_array)
    #np.save(f'results/{result_type}_predicted_state_array',predicted_state_array)
    np.save(f'{experiment_type}_worst_delay_compensation',control_array)
    if experiment_type=="biased_mppi_with_shifted_control" : #or experiment_type=="biased_mppi_with_shifted_control":
        np.save(f'/root/workspace/src/pybullet_examples/jackal/{experiment_type}_DELAY_SEQ_biased',delay_seq)
    # pdb.set_trace()
    sys.exit()







def run_point_robot_example():
    global control_array, total_delay , experiment_type,delay_seq,MEAN_DELAY
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
    


    if experiment_type=="delay_with_cvar_comp":
        # planner = MPPI(
        #         initialState= CONFIG["initial_actor_positions"],
        #         goal= CONFIG["goal"] + [-0.05],
        #         obstacle_states= [[0,0,0.0]]

        # )
        planner = MPPI()
    else:
        planner = MPPIPlanner(
        cfg=CONFIG["mppi"],
        nx=12,
        dynamics=dynamics.step,
        running_cost=objective.compute_running_cost,
        )    
    initial_action = torch.zeros(2, device=CONFIG["device"])
    ob, *_ =  simulator._environment.step(initial_action.cpu().numpy())

    # simulator._environment.start_video_recording("jackal_nobias_sim.mp4")

    sensor_dt = 0.05
    sim_dt = CONFIG["dt"]
    controller_dt  = 0.05

    sim_time = 0

    

    action_Seq = np.zeros((2,100))
    t=0.0
    
    
       
    current_control = np.array([3,0])
    
    current_time = 0.0
    ob_robot = ob["robot_0"]
    print("desired-speed",current_control[0] , "-current-speed=>-",ob_robot["joint_state"]["forward_velocity"] , "time",current_time)
    for i in range(10):
        (ob,*_,) = simulator._environment.step(current_control)
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
        current_time+=sim_dt
        #pdb.set_trace()
        print(ob_robot["joint_state"]["forward_velocity"])
        if(abs(ob_robot["joint_state"]["forward_velocity"] -current_control[0])<1e-3 ):
            print("desired-speed",current_control[0] , "-current-speed=>-",ob_robot["joint_state"]["forward_velocity"] , "time",current_time)
            pdb.set_trace()
    
    
    
    
    
    
    

if __name__ == "__main__":
    run_point_robot_example()