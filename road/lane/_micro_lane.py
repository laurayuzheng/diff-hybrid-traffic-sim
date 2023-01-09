'''
@ author: SonSang (Sanghyun Son)
@ email: shh1295@gmail.com
'''
import torch as th

from road.lane._base_lane import BaseLane
from road.vehicle.micro_vehicle import MicroVehicle
from model.micro._idm import IDM
from dmath.operation import sigmoid

from typing import List, Union, Dict
import copy, gc

DEFAULT_HEAD_POSITION_DELTA = 1000
DEFAULT_HEAD_SPEED_DELTA = 0

POSITION_DELTA_EPS = 1e-5

class MicroLane(BaseLane):

    '''
    Lane that simulates traffic flow using microscopic IDM model.

    Use automatic differentiation for differentiation.
    '''
    
    # Lateral policy parameters
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]

    def __init__(self, id: int, lane_length: float, speed_limit: float):
    
        super().__init__(id, lane_length, speed_limit)

        # list of vehicles;
        # here we assume i-th vehicle is right behind (i + 1)-th vehicle;
        
        self.curr_vehicle: List[MicroVehicle] = []

        # position and speed that would be used to update vehicle states;

        self.acc_info = []
        self.next_vehicle_position: List[Union[float, th.Tensor]] = []
        self.next_vehicle_speed: List[Union[float, th.Tensor]] = []

        # value to use for the head vehicle, which does not have leading vehicle;

        self.head_position_delta = DEFAULT_HEAD_POSITION_DELTA
        self.head_speed_delta = DEFAULT_HEAD_SPEED_DELTA

    def is_macro(self):
        return False

    def is_micro(self):
        return True

    def add_head_vehicle(self, vehicle: MicroVehicle):

        self.curr_vehicle.append(vehicle)

    def add_tail_vehicle(self, vehicle: MicroVehicle):

        self.curr_vehicle.insert(0, vehicle)

    def add_vehicle(self, vehicle: MicroVehicle) -> int:

        # find proper index to insert;

        assert vehicle.position >= 0 and vehicle.position <= self.length, ""

        if self.num_vehicle() == 0:

            self.add_head_vehicle(vehicle)

            return len(self.curr_vehicle) - 1

        for i, v in enumerate(self.curr_vehicle):

            if v.position > vehicle.position:

                # insert at tail;

                assert i == 0 and v.position - vehicle.position >= (v.length + vehicle.length) * 0.5, ""

                self.add_tail_vehicle(vehicle)
                
                return 0
                # break

            else:

                pv = v

                # check if the vehicle collides with following vehicle;

                assert vehicle.position - pv.position > (vehicle.length + pv.length) * 0.5, ""

                if i == self.num_vehicle() - 1:

                    self.add_head_vehicle(vehicle)

                    return len(self.curr_vehicle) - 1 

                else:

                    nv = v

                    if nv.position > vehicle.position:

                        # check if the vehicle collides with leading vehicle;

                        assert nv.position - vehicle.position > (vehicle.length + nv.length) * 0.5, ""

                        self.curr_vehicle.insert(i + 1, vehicle)

                        return i+1

                    else:

                        continue

                # break

    def remove_vehicle(self, vehicle: MicroVehicle) -> int:
        try:
            self.curr_vehicle.remove(vehicle)
            return 0
        except ValueError: 
            return 1

    def num_vehicle(self):

        return len(self.curr_vehicle)
            
    def get_head_vehicle(self):

        assert self.num_vehicle(), ""

        return self.curr_vehicle[-1]

    def get_tail_vehicle(self):

        assert self.num_vehicle(), ""

        return self.curr_vehicle[0]

    def get_candidate_new_lanes(self):

        # Dictionary of candidate adjacent lanes by vehicle idx and new lane objects
        candidate_adj_lanes : Dict[int, LaneChangeCandidate] = {}

        for vi, veh in enumerate(self.curr_vehicle):
            
            candidate_lanes = []
            vi_candidates = []
            original_lanes = []

            for lane_id, lane in self.adjacent_lane.items():

                if lane.is_macro():
                    continue
                
                # Re-scale vehicle position relative to adjacent lane
                adj_veh_position = veh.position / self.length * lane.length
                
                # Make a deep copy of vehicle to add to candidate lane
                veh_cpy = MicroVehicle(veh.id, 
                                        adj_veh_position, 
                                        veh.speed, 
                                        veh.accel_max, 
                                        veh.accel_pref, 
                                        veh.target_speed, 
                                        veh.min_space, 
                                        veh.time_pref, 
                                        veh.length, 
                                        veh.a)

                candidate_lane = copy.deepcopy(lane)
                ind_in_cand_lane = candidate_lane.add_vehicle(veh_cpy) 
                candidate_lanes.append(candidate_lane)
                vi_candidates.append(ind_in_cand_lane)
                original_lanes.append(lane)
            
            candidate_obj = LaneChangeCandidate(vi, vi_candidates, candidate_lanes, original_lanes)
            candidate_adj_lanes[vi] = candidate_obj

        return candidate_adj_lanes


    def mobil(self, delta_time: float):
        '''
        Determine whether a lane change should occur based on the MOBIL lane change model. 
        References adjacent lanes to check for lane change criteria. 
        '''

        # valid target lanes (key: Vehicle ID) to change to for each vehicle
        valid_target_lanes : Dict[int, List[MicroLane]] = {} 
        candidate_lanes : Dict[int, LaneChangeCandidate] = self.get_candidate_new_lanes()

        for vi, candidates in candidate_lanes.items():
            mv = self.curr_vehicle[vi]
            valid_target_lanes[vi] = [self] # staying in lane is valid option

            for i in range(candidates.num_candidates):

                original_lane = candidates.original_lanes[i]
                candidate_lane = candidates.candidate_lanes[i]
                candidate_idx = candidates.vi_candidate[i]

                # If last vehicle, make sure next lane has enough space 
                if candidate_idx == 0 and \
                    candidate_lane.curr_vehicle[0].position < \
                         (candidate_lane.curr_vehicle[0].length / 2):
                    continue
                
                # If first vehicle, make sure next lane has enough space in front
                if candidate_idx == len(candidate_lane.curr_vehicle) - 1 and \
                    (candidate_lane.curr_vehicle[0].position + (candidate_lane.curr_vehicle[0].length / 2)) > \
                         candidate_lane.length:
                    continue

                # candidate_acc = self.get_lane_accel(candidate_lane, candidate_idx, mv, delta_time)

                # Is the maneuver unsafe for the new following vehicle? 
                new_following_a = self.get_lane_accel(original_lane, candidate_idx-1, mv, delta_time) if candidate_idx > 0 else 0 
                new_following_pred_a = self.get_lane_accel(candidate_lane, candidate_idx-1, mv, delta_time) if candidate_idx > 0 else 0 

                if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                    continue

                # Do I have a planned route for a specific lane which is safe for me to access?
                self_pred_a = self.get_lane_accel(candidate_lane, candidate_idx, mv, delta_time)
                if self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                    continue

                self_a = self.get_lane_accel(self, vi, mv, delta_time)
                old_following_a = self.get_lane_accel(self, vi-1, mv, delta_time) if vi > 0 else 0.
                pred_self = copy.deepcopy(self)
                pred_self.remove_vehicle(mv)
                old_following_pred_a = self.get_lane_accel(pred_self, vi-1, mv, delta_time) if vi > 0 else 0.
                jerk = self_pred_a - self_a + self.POLITENESS * (new_following_pred_a - new_following_a
                                                                + old_following_pred_a - old_following_a)
                if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                    continue

                valid_target_lanes[vi].append(candidate_lane)

        return valid_target_lanes

    def get_lane_accel(self, lane : BaseLane, vi : int, mv : MicroVehicle, delta_time : float):
        # compute next position and speed using Eulerian method;

        position_delta, speed_delta = lane.compute_state_delta(vi)

        try:

            self.handle_collision(position_delta)

        except Exception as e:

            print(e)
            print("Set deltas to 0, but please check traffic flow for unrealistic behavior...")

            position_delta, speed_delta = 0, 0

        assert position_delta >= 0, "Vehicle collision detected"

        # prevent division by zero;

        position_delta = max(position_delta, POSITION_DELTA_EPS)

        acc_info = IDM.compute_acceleration(mv.accel_max,
                                                mv.accel_pref,
                                                mv.speed,
                                                mv.target_speed,
                                                position_delta,
                                                speed_delta,
                                                mv.min_space,
                                                mv.time_pref,
                                                delta_time)
        acc = acc_info[0]

        return acc 

    def forward(self, delta_time: float):

        '''
        Take a single forward simulation step by computing vehicle state values of next time step.

        Note that the new state values are stored in [next_vehicle_position] and [next_vehicle_speed]; 
        call [update_state] to apply them to [curr_vehicle].
        '''      

        self.next_vehicle_position.clear()
        self.next_vehicle_speed.clear()

        self.acc_info.clear()

        for vi, mv in enumerate(self.curr_vehicle):

            # compute next position and speed using Eulerian method;

            position_delta, speed_delta = self.compute_state_delta(vi)

            try:

                self.handle_collision(position_delta)

            except Exception as e:

                print(e)
                print("Set deltas to 0, but please check traffic flow for unrealistic behavior...")

                position_delta, speed_delta = 0, 0

            assert position_delta >= 0, "Vehicle collision detected"

            # prevent division by zero;

            position_delta = max(position_delta, POSITION_DELTA_EPS)

            acc_info = IDM.compute_acceleration(mv.accel_max,
                                                    mv.accel_pref,
                                                    mv.speed,
                                                    mv.target_speed,
                                                    position_delta,
                                                    speed_delta,
                                                    mv.min_space,
                                                    mv.time_pref,
                                                    delta_time)

            self.acc_info.append(acc_info)

            acc = acc_info[0]

            next_position = mv.position + delta_time * mv.speed
            next_speed = mv.speed + delta_time * acc

            self.next_vehicle_position.append(next_position)
            self.next_vehicle_speed.append(next_speed)

        print(self.mobil(delta_time))
        # gc.collect()

    def handle_collision(self, position_delta: float):

        if position_delta < 0:

            raise ValueError("Collision detected, position delta = {:.2f}".format(position_delta))
            

    def compute_state_delta(self, id):
        
        '''
        Compute position and speed delta to leading vehicle.
        '''

        if id == len(self.curr_vehicle) - 1:

            position_delta = self.head_position_delta
            speed_delta = self.head_speed_delta

        else:

            mv = self.curr_vehicle[id]
            lv = self.curr_vehicle[id + 1]          # leading vehicle;
            
            position_delta = abs(lv.position - mv.position) - ((lv.length + mv.length) * 0.5)
            speed_delta = mv.speed - lv.speed

        return position_delta, speed_delta

    def update_state(self):
        
        '''
        Update current vehicle state with next state.
        '''

        for i, mv in enumerate(self.curr_vehicle):

            mv.position = self.next_vehicle_position[i]
            mv.speed = self.next_vehicle_speed[i]        

    def set_state_vector(self, position: th.Tensor, speed: th.Tensor):

        '''
        Set vehicle state from given vector, of which length equals to number of vehicles.
        '''

        assert len(position) == self.num_vehicle(), "Vehicle number mismatch"
        assert len(speed) == self.num_vehicle(), "Vehicle number mismatch"

        for i in range(self.num_vehicle()):

            self.curr_vehicle[i].position = position[i]
            self.curr_vehicle[i].speed = speed[i]

    def get_state_vector(self):

        '''
        Get state vector in the order of position and speed.
        '''

        position = th.zeros((self.num_vehicle(),))
        speed = th.zeros((self.num_vehicle(),))

        for i in range(self.num_vehicle()):
            position[i] = self.curr_vehicle[i].position
            speed[i] = self.curr_vehicle[i].speed

        return position, speed


    def set_next_state_vector(self, position: th.Tensor, speed: th.Tensor):

        '''
        Set next vehicle state from given vector, of which length equals to number of vehicles.
        '''

        assert len(position) == self.num_vehicle(), "Vehicle number mismatch"
        assert len(speed) == self.num_vehicle(), "Vehicle number mismatch"

        self.next_vehicle_position.clear()
        self.next_vehicle_speed.clear()

        for i in range(self.num_vehicle()):

            self.next_vehicle_position.append(position[i])
            self.next_vehicle_speed.append(speed[i])

    def get_next_state_vector(self):

        '''
        Get next state vector in the order of position and speed.
        '''

        position = th.zeros((self.num_vehicle(),))
        speed = th.zeros((self.num_vehicle(),))

        for i in range(self.num_vehicle()):
            position[i] = self.next_vehicle_position[i]
            speed[i] = self.next_vehicle_speed[i]

        return position, speed

    def entering_free_space(self):

        '''
        Get free space at the beginning of the lane;
        '''

        if self.num_vehicle():

            return self.curr_vehicle[0].position - 0.5 * (self.curr_vehicle[0].length)

        else:

            return self.length

    def on_this_lane(self, position: float, differentiable: bool):

        '''
        Return True (1.0) if [position] is in between 0 and this lane's length.
        '''

        if not isinstance(position, th.Tensor):

            position = th.tensor(position)

        if differentiable:

            return sigmoid(position, constant=16.0) * sigmoid(self.length - position, constant=16.0)

        else:

            return float(position >= 0 and position <= self.length)

    def clear(self):

        '''
        Clear every vehicle and next state info.
        '''

        self.curr_vehicle.clear()

        self.next_vehicle_position = []
        self.next_vehicle_speed = []

class LaneChangeCandidate: 

    def __init__(self, vi: int, vi_candidate: int, candidate_lanes: List[MicroLane], original_lanes : List[MicroLane]):
        self.vi_original = vi
        self.vi_candidate = vi_candidate
        self.candidate_lanes = candidate_lanes
        self.original_lanes = original_lanes 
        self.num_candidates = len(candidate_lanes)
