
import numpy as np
from dataclasses import dataclass
import yaml
import copy

with open('utils/sim_parameters.yml','r') as file :
    sim_parameters = yaml.safe_load(file)
 
 
 
@dataclass
class obstacle_state:
    x:np.float32
    y:np.float32
    r:np.float32
    vt:np.float32=0
   
@dataclass
class robotState:
    x:np.float32
    y:np.float32
    yaw:np.float32


@dataclass
class robotlogs:
    states : list[float]
    stamps : list[float]

@dataclass
class robotControl:
    speed  : np.float32 
    omega : np.float32

    
from abc import ABC, abstractmethod
class network_system(ABC):
    sim_dt  :    np.float32  
    sim_timer =  np.float32(0.00)     
    time_control_ack = np.float32(0.00)
    sensor_dt : np.float32
    sensor_itr : np.float32
    
    
    uplink_delay_itr :np.float32
    uplink_delay : np.float32
        
    downlink_delay_itr  : np.float32
    downlink_delay  : np.float32
    
    sim_parameters=sim_parameters

    def __init__(self, init_state:robotState, init_controller : robotControl,track_obj:object,car_phys_params :dict ):
        self.car_parameters = car_phys_params
        self.track_obj = track_obj
        self.sim_dt = self.sim_parameters['sim_dt']
        
        self.sim_timer+= self.sim_dt   #important
        
        
        
        self.uplink_delay =    self.sim_parameters['UPLINK_DELAY']
        self.uplink_delay_itr = 0
        
        self.downlink_delay = self.sim_parameters['DOWNLINK_DELAY']
        self.downlink_delay_itr = 0 
       
       
        self.sensor_dt = self.sim_parameters['sensor_dt']
        self.sensor_itr = 0.0

        
        
        self.t = np.linspace(0, 10 , 500*10, endpoint=False)


        #initialization 
        self.true_state = copy.deepcopy(init_state) #signal.square(2 * np.pi * 1 *self.sim_timer)
        
        
        self.control_message_ack = True  # is message has been acknowledged by the client
        self.sensor_message_ack  =  True  # is message has been acknowledged by the server
        
        self.sensor_state = self.true_state  # sensor will sample the state at every 10hz
        self.next_message_to_server =   self.sensor_state # send it through the network
        
        self.controller_state = self.next_message_to_server  #  what state does the controller should work on  
        self.controller_control = np.zeros((3,12)) #robotControl(acceleration=0.0,steering_angle=0.0,progress_speed=0.0,time_sampled=0.0)   # compute the control
        
        self.system_control = self.controller_control     # client has recieved this control
        self.actuator_control  = np.zeros((3,1))  #copy.deepcopy(init_controller) # copy.deepcopy(self.system_control) # actual control being executed by the actuator
        
        
        self.true_logs = robotlogs([self.true_state] , [0])
        self.controller_logs = robotlogs([self.true_state] , [0])
        self.system_logs = robotlogs([self.true_state] , [0])
        
        
        self.true_logs.states.append(self.true_state )
        self.true_logs.stamps.append(self.sim_timer)
        
        self.controller_logs.states.append(self.controller_control)
        self.controller_logs.stamps.append(self.sim_timer)   

        self.system_logs.states.append(self.system_control)
        self.system_logs.stamps.append(self.sim_timer)          


        self.new_control_flag = True
        
    @classmethod
    @abstractmethod
    def get_control(self,state):
        control = state
        return control   
    
    @classmethod
    @abstractmethod     
    def step_sim(self,control, new_control:bool):
        
        state = np.random.choice([-1,1] , size=1)
        return state
    
        
        