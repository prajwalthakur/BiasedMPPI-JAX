#!/usr/bin/env python3
# desigining cost function as i did before 
import torch
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
import ghalton
from scipy.interpolate import interp1d
from scipy import signal
import scipy.interpolate as si
import scipy.special as scsp
import pdb
import copy
from jax import config
from jax import lax



# # jax-config
config.update("jax_enable_x64", True)
class EgoParams:
    def __init__(self):
        self.ctrTs  = 0.05
        self.T  = 4
        self.horizonLength = int(self.T/self.ctrTs)
        self.traj_vis_number = 1
        self.control_dim = 2
        self.state_dim = 3
        # control-limit
        self.speed_min_limit = -0.3
        self.speed_max_limit = 1.9
        self.angular_speed_min = -0.9
        self.angular_speed_max = 0.9
        self.control_cov = np.diag([0.2,0.04])    # speeed , angular-speed 
        self.control_mean =  np.zeros(2)
        self.mppi_num_rollouts = 1200
        self.speed_default = 0.5
    
        self.terminal_goal_cost = 5
        
        self.lambda_weight = 0.1 # 0.75  # dyanmic update now 
        self.rollout_var_discount =1 # less weight on horizon
        self.sample_zero_seq = False
        self.add_ancillary_action = False 
        
        
        self.sampling_type = "gaussian_halton"                     
        self.input_sample_every = 4
        self.param_exploration = 0.2
        # mppi weight
        self.goal_tolerance  = 0.4

        self.collision_cost =100
        self.weight_goal = np.diag([5,5,0])
        
        
        # extra features
        self.update_lambda = True
        self.beta_lm = 0.9
        self.beta_um = 1.2
        self.eta_u_bound = 10
        self.eta_l_bound = 5
        
        
        self.knot_scale = 2
        self.seed_value = 0
        self.n_knots = self.horizonLength//self.knot_scale
        self.ndims = self.n_knots*self.control_dim
        self.degree = 1                 # From sample_lib storm is 2
        self.tensor_args={'device':'cpu', 'dtype':torch.float64}
        
        
        self.sgf_window = 9
        self.sgf_order = 2

def bspline(c_arr, t_arr=None, n=100, degree=3):
    #sample_device = c_arr.device
    sample_dtype = c_arr.dtype
    cv = c_arr

    if(t_arr is None):
        t_arr = np.linspace(0, cv.shape[0], cv.shape[0])
    else:
        t_arr = t_arr
    spl = si.splrep(t_arr, cv, k=degree, s=0.5)
    xx = np.linspace(0, cv.shape[0], n)
    samples = si.splev(xx, spl, ext=3)
    samples = np.array(samples, dtype=sample_dtype)
    return samples

class MPPI(EgoParams):
    def __init__(self):
        EgoParams.__init__(self)
        
        self.Z_seq = np.zeros((1,self.horizonLength,self.control_dim))
        self.constant_forward_control_seq = np.zeros((1,self.horizonLength,self.control_dim))
        self.constant_forward_control_seq[:,:,0] = self.speed_default
        self.constant_backward_control_seq = np.zeros((1,self.horizonLength,self.control_dim))
        self.constant_backward_control_seq[:,:,0] = -self.speed_default
        self.U_seqs = np.zeros((self.horizonLength,self.control_dim))
        self.mppi_fcn_init()
        
    def fwd_sim_external(self, state:jnp.ndarray,uk:jnp.ndarray,dt):
        x0 = state[0][0]
        y0 = state[1][0]
        yaw0=state[2][0]
        v0 = uk[0][0]
        omega = uk[1][0]
        xdot = jnp.asarray([[v0*jnp.cos(yaw0) ],
                [v0*jnp.sin(yaw0)] ,
                [omega]
                ])
        x_next = state + xdot*dt
        return x_next

    def fwd_sim(self, state:jnp.ndarray,uk:jnp.ndarray):
        dt = self.ctrTs
        x0 = state[0][0]
        y0 = state[1][0]
        yaw0=state[2][0]
        v0 = uk[0][0]
        omega0 = uk[1][0]
        v0 =    jnp.clip(v0,  self.speed_min_limit,  self.speed_max_limit) # limit acceleraiton input
        omega0 = jnp.clip( omega0 , self.angular_speed_min, self.angular_speed_max) # limit steering input
        xdot = jnp.asarray([[v0*jnp.cos(yaw0) ],
                [v0*jnp.sin(yaw0)] ,
                [omega0],
                ])
        x_next = state + xdot*dt
        return x_next

   
 
    def fwd_sim_delay(self,xk, uk, sim_timestep):
        body_fun = lambda i, x_u: self.fwd_sim_perturb(x_u,i)
        x_next,_,_ = lax.fori_loop(0, sim_timestep[0], body_fun, (xk, uk,self.ctrTs))
        return x_next
    def mppi_fcn_init(self):
        self.cart_fwd_sim_jit = jax.jit(jax.vmap(self.fwd_sim , in_axes=0,out_axes=0))
        self.stage_cost_jit = jax.jit(jax.vmap(self.stage_cost,in_axes=(0,0,None,None),out_axes=(0,0)))
        self.terminal_cost_jit = jax.jit(jax.vmap(self.terminal_cost , in_axes=(0,0,None,None),out_axes=0))
        


    
    def command(self,state,control_itr=1):
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        state = np.array(state.to(dtype=self.tensor_args['dtype'], device=self.tensor_args['device']) ).squeeze()
        
        predicted_ego_State = np.array([state[0],state[1],state[2]])[...,np.newaxis]
        predicted_obs_state = np.array([state[6], state[7] , 0.5 ])[np.newaxis,...]
        #predicted_obs_state = np.vstack((predicted_obs_state,np.array([state[6], state[7] , 0.5 ])))
        goal_array = np.array([2.7,0.0,0.0])[...,np.newaxis]
        self.u_prev = self.U_seqs[0,:][np.newaxis,...]
        self.U_seqs = jnp.roll(self.U_seqs,-control_itr,axis=0) 
        uopt_seq = self.control_cb(predicted_ego_State,predicted_obs_state,goal_array)
        return uopt_seq
    def control_cb(self,predicted_ego_State,predicted_obs_state,goal_array):
        delta_u = self.control_pertubations(mean=self.control_mean, cov= self.control_cov, sampling_type=self.sampling_type)

        nominal_mppi_cost , perturb_control , delta_u = self.cal_nominal_mppi_cost(predicted_ego_State,predicted_obs_state,goal_array,delta_u)

        cvar_cost=0


        total_cost = nominal_mppi_cost + cvar_cost
            
        wght ,eta  = self.compute_weights(total_cost)
        
        #pdb.set_trace()
        temp =  jnp.sum(jnp.multiply(wght[:,jnp.newaxis],delta_u),axis=0)
        temp =  self._moving_average_filter(xx=temp,window_size=5)
        self.U_seqs +=temp
        self.U_seqs = jnp.round(self.U_seqs,decimals=4)
        action = np.array(self.U_seqs)
        #pdb.set_trace()
        if self.update_lambda:
            if eta > self.eta_u_bound:
                self.lambda_weight = self.lambda_weight*self.beta_lm
        elif eta < self.eta_l_bound:
            self.lambda_weight = self.lambda_weight*self.beta_um 


        u_filtered = self.control_clip( action)   
        return u_filtered
    def _moving_average_filter(self,xx:np.ndarray,window_size:int)->np.ndarray:
        """apply moving average filter for smoothing input sequence
        Ref. https://zenn.dev/bluepost/articles/1b7b580ab54e95
        """   
        b = np.ones(window_size)/window_size # 1D array of size 10 value = 1/10
        dim = xx.shape[1]  # 3
        xx_mean = np.zeros(xx.shape) #20x3
        for d in range(dim):
            xx_mean[:,d] = np.convolve(xx[:,d] , b , mode = "same")
            n_cov = int(np.ceil(window_size/2))
            for i in range(1,n_cov):
                xx_mean[i,d] *= window_size/(i+n_cov)
                xx_mean[-i,d] *= window_size/(i+n_cov - (window_size % 2))    
        return xx_mean
            
    def compute_weights(self,S:jnp.ndarray) -> jnp.ndarray:
        "compute  weights for each rollout"
        #prepare buffer
        
        rho = jnp.min(S)
        #cal eta
        eta = jnp.sum(jnp.exp( (-1.0/self.lambda_weight) * (S-rho) ))
        #calculate weight
        wt = (1.0 / eta) * jnp.exp( (-1.0/self.lambda_weight) * (S-rho) )
        return wt ,eta  
    def cal_nominal_mppi_cost(self, predicted_ego_State,predicted_obs_state,goal_array,delta_u):
        state_tensor = jnp.tile(predicted_ego_State , (self.mppi_num_rollouts,1,1))
        U_prev_seqs_tensor = jnp.tile(self.U_seqs,(self.mppi_num_rollouts,1,1))
        
        idx_exp : int = int((1-self.param_exploration)*self.mppi_num_rollouts) 
        temp_jax_tensor = np.zeros((self.mppi_num_rollouts,self.horizonLength,self.control_dim))
        temp_jax_tensor[:idx_exp] = U_prev_seqs_tensor[:idx_exp] + delta_u[:idx_exp]
        temp_jax_tensor[idx_exp:,:,:] = delta_u[idx_exp:]       
        temp_jax_tensor= U_prev_seqs_tensor + delta_u
        U_current_tensor = self.control_clip_vec(temp_jax_tensor)   

        if self.add_ancillary_action:
            U_current_tensor[-2,:,:] = self.Z_seq 
            #U_current_tensor[-3,:,:] = self.constant_forward_control_seq
            # U_current_tensor[-4,:,:] =  self.constant_backward_control_seq 
        U_current_tensor = jnp.round(jnp.asarray(U_current_tensor), decimals=4)
        seq_cost =  jnp.zeros((self.mppi_num_rollouts,1))
        prev_is_reached = jnp.ones((self.mppi_num_rollouts,))
        for itr in range(0,self.horizonLength):
            #pdb.set_trace()
            state_tensor= self.cart_fwd_sim_jit(state_tensor,U_current_tensor[:,itr,:,jnp.newaxis])
            stage_cost,prev_is_reached = self.stage_cost_jit(state_tensor,prev_is_reached,goal_array,predicted_obs_state)
            #pdb.set_trace()
            seq_cost+= (self.rollout_var_discount**itr)*stage_cost

        stage_cost = self.terminal_cost_jit(state_tensor,prev_is_reached,goal_array , predicted_obs_state)    
        seq_cost+=stage_cost 
        seq_cost += seq_cost.mean()
        delta_u = U_current_tensor-U_prev_seqs_tensor
        return seq_cost,U_current_tensor,delta_u
     
    def stage_cost(self,s_t,prev_is_reached,goal_array,obstacle_array):
        x_obs = obstacle_array[0,0]
        y_obs = obstacle_array[0,1]
        x_ego = s_t[0,0]
        y_ego = s_t[1,0]
        x_goal = goal_array[0,0]
        y_goal = goal_array[1,0]
        dist_to_obs = jnp.linalg.norm(jnp.asarray([x_obs-x_ego , y_obs-y_ego]))
        cost_obs = jnp.where(dist_to_obs<1,0.5,0)*self.collision_cost
        
        dist_to_goal =    jnp.linalg.norm(jnp.asarray([x_goal-x_ego , y_goal-y_ego]))
        is_reached = prev_is_reached*jnp.where(dist_to_goal<=self.goal_tolerance,0.0,1).squeeze()
        cost_to_goal = self.ctrTs*is_reached*dist_to_goal*1.0 #
        #pdb.set_trace()
        final_cost = cost_obs + cost_to_goal#cost_to_goal #dist_to_goal #cost_to_goal
        final_cost = final_cost[jnp.newaxis,...]
        
        return final_cost,is_reached
    
    def terminal_cost(self,s_t, prev_is_reached ,goal ,predicted_obs_state):
        # goal tolerance 
        dist_to_goal = ((s_t[0] - goal[0] )**2 + (s_t[1] - goal[1])**2 )**0.5
        terminal_cost = dist_to_goal*self.terminal_goal_cost/self.speed_default
        is_reached = prev_is_reached*jnp.where(dist_to_goal<=self.goal_tolerance,0.0,1)

        final_terminal_cost = terminal_cost*is_reached*dist_to_goal
        return final_terminal_cost    

 
 
        
        
        
                    
    def control_pertubations(self,mean:float=jnp.zeros(5),cov:float= jnp.eye(5) , sampling_type:str=None)->jnp.ndarray:
        """Sample an array of control perturbations delta_u. Samples for two distinct rollouts are always independent

        :param stdev: standard deviation of samples if Gaussian, defaults to 1.0
        :type stdev: float, optional
        :param sampling_type: defaults to None, can be one of
            - "random_walk" - The next horizon step's perturbation is correlated with the previous one
            - "uniform" - Draw uniformly distributed samples between -1.0 and 1.0
            - "repeated" - Sample only one perturbation per rollout, apply it repeatedly over the course of the rollout
            - "interpolated" - Sample a new independent perturbation every 10th MPC horizon step. Interpolate in between the samples
            - "iid" - Sample independent and identically distributed samples of a Gaussian distribution
        :type sampling_type: str, optional
        :return: Independent perturbation samples of shape (num_rollouts x horizon_steps)
        :rtype: jnp.ndarray
        """        
        if sampling_type=="interpolated":
            mpc_horizon = self.horizonLength
            step =  self.input_sample_every
            range_stop = int(jnp.ceil((mpc_horizon/step)*step)) + 1
            t = jnp.arange(start=0 , stop = range_stop ,step= step)
            t_interp = np.arange(start=0 ,stop = range_stop,step =1 )
            t_interp = np.delete(t_interp,t)
            delta_u = np.zeros(shape=(self.mppi_num_rollouts,range_stop,self.control_dim),dtype=jnp.float64)
            delta_u[:,t] = self.rng_mppi.multivariate_normal(mean,cov,
                size=(self.mppi_num_rollouts,t.size)
            )
            f=  interp1d(t,delta_u[:,t],axis = 1)
            delta_u[:,t_interp] = f(t_interp)
            delta_u = delta_u[:,:mpc_horizon]
        elif sampling_type == "iid":
            delta_u = self.rng_mppi.multivariate_normal(mean,cov,
                size=(self.mppi_num_rollouts,self.horizonLength))     

        elif sampling_type == "gaussian_halton":
            sample_shape = self.mppi_num_rollouts
            # self.knot_points = generate_gaussian_halton_samples(
            #     sample_shape,               # Number of samples
            #     self.ndims,                 # n_knots * nu (knots per number of actions)
            #     use_ghalton=False,
            #     seed_val=self.seed_val)
            sequencer = ghalton.GeneralizedHalton(self.ndims, self.seed_value)
    
            knot_points = np.array(sequencer.get(self.mppi_num_rollouts),dtype=float)
            gaussian_halton_samples = np.sqrt(2.0)*scsp.erfinv(2 * knot_points - 1)
            # Sample splines from knot points:
            # iteratre over action dimension:
            knot_samples = gaussian_halton_samples.reshape(sample_shape, self.control_dim, self.n_knots) # n knots is T/knot_scale (30/4 = 7)
            delta_u = np.zeros((sample_shape, self.horizonLength, self.control_dim))
          
            for i in range(sample_shape):
                for j in range(self.control_dim):
                    delta_u[i,:,j] = bspline(knot_samples[i,j,:], n=self.horizonLength, degree=self.degree)
        #print("delta-u shape,",  delta_u.shape)
        #pdb.set_trace()
        temp =  np.sqrt((cov))
        delta_u = np.matmul(delta_u ,temp )
        if self.sample_zero_seq:
            delta_u[-1,:,:] = self.Z_seq
        return delta_u
    
    
    def control_clip_vec(self, v: jnp.ndarray) :
        """clamp input"""
        # limit control inputs
        temp_array = np.asarray(np.array(v))
        temp_array[:,:,0] = jnp.clip(v[:,:,0],  self.speed_min_limit,  self.speed_max_limit) # limit acceleraiton input
        temp_array[:,:,1] = jnp.clip(v[:,:,1], self.angular_speed_min, self.angular_speed_max) # limit steering input
        #temp_array = jnp.asarray(temp_array)
        return temp_array
    
    def control_clip(self, v: jnp.ndarray) :
        """clamp input"""
        # limit control inputs
        v[:,0] = jnp.clip(v[:,0],  self.speed_min_limit,  self.speed_max_limit) # limit acceleraiton input
        v[:,1] = jnp.clip(v[:,1], self.angular_speed_min, self.angular_speed_max) # limit steering input
        return v

   