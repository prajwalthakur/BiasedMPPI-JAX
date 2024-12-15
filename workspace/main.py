import torch
import os
from simulator import Simulator
from objective import Objective
from dynamics import JackalDynamics
from mppi_jax.cvar_and_biased_mppi import MPPI
import yaml
from tqdm import tqdm
import pdb
import numpy as np
import numpy as np
import copy


abs_path = os.path.dirname(os.path.abspath(__file__))
CONFIG = yaml.safe_load(open(f"{abs_path}/config.yaml"))

sim_iter_per_mpc_call = int(CONFIG["control_dt"]//CONFIG["dt"])


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
    

    planner = MPPI()  
    initial_action = torch.zeros(2, device=CONFIG["device"])
    ob, *_ =  simulator._environment.step(initial_action.cpu().numpy())

    sensor_dt = 0.05
    sim_dt = 0.05
    controller_dt  = 0.05

    sim_time = 0
    action_Seq = np.zeros((2,100))
    t=0.0
    for _ in tqdm(range(CONFIG["steps"])):
        sim_time = 0.0
        default_itr=0
        control_itr = 0
        t=t+1

        while(sim_time<sensor_dt):
            sim_time = sim_time+ sim_dt
            current_control = action_Seq[:,control_itr]
            (ob,*_,) = simulator._environment.step(current_control)
            control_itr  = int(default_itr + np.floor(sim_time /controller_dt -1e-6))

        control_itr+=1  
        obs_after_sensor_sampling = copy.deepcopy(ob)
        time_after_sensor_sampling = copy.deepcopy(sim_time)
        control_itr_after_sampling = copy.deepcopy(control_itr)
        ob_robot = obs_after_sensor_sampling["robot_0"]
        obst = obs_after_sensor_sampling["robot_0"]["FullSensor"]["obstacles"]
        observation_tensor = torch.tensor(
            [
                [*ob_robot["joint_state"]["position"],
                *ob_robot["joint_state"]["velocity"],
                *obst[3]['position'],
                *obst[3]['velocity']]
            ],
            device=CONFIG["device"],
        )
        state_after_sensor_sample = np.array(observation_tensor.to(dtype=torch.float32, device='cpu') ).squeeze()
        controller_sampled_state_array = state_after_sensor_sample
        action_Seq = planner.command(state_after_sensor_sample,control_itr = control_itr)
        action_Seq  = action_Seq.T
      

if __name__ == "__main__":
    run_point_robot_example()