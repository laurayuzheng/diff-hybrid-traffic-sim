'''
@ author: SonSang (Sanghyun Son)
@ email: shh1295@gmail.com
'''
import torch as th
import numpy as np

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

    def add_vehicle(self, vehicle: MicroVehicle):

        # find proper index to insert;

        assert vehicle.position >= 0 and vehicle.position <= self.length, ""

        if self.num_vehicle() == 0:

            self.add_head_vehicle(vehicle)

            return

        for i, v in enumerate(self.curr_vehicle):

            if v.position > vehicle.position:

                # insert at tail;

                assert i == 0 and v.position - vehicle.position >= (v.length + vehicle.length) * 0.5, ""

                self.add_tail_vehicle(vehicle)
                
                break

            else:

                pv = v

                # check if the vehicle collides with following vehicle;

                assert vehicle.position - pv.position > (vehicle.length + pv.length) * 0.5, ""

                if i == self.num_vehicle() - 1:

                    self.add_head_vehicle(vehicle)

                else:

                    nv = v

                    if nv.position > vehicle.position:

                        # check if the vehicle collides with leading vehicle;

                        assert nv.position - vehicle.position > (vehicle.length + nv.length) * 0.5, ""

                        self.curr_vehicle.insert(i + 1, vehicle)

                    else:

                        continue

                break

    def remove_vehicle(self, vehicle: MicroVehicle) -> int:
        try:
            self.curr_vehicle.remove(vehicle)
            return 0
        except ValueError: 
            return 1

    def remove_vehicle(self, vi: int) -> int: 
        ''' Polymorphism of above. Parameter is index in curr_vehicle.
        '''
        self.curr_vehicle.pop(vi)

    def num_vehicle(self):

        return len(self.curr_vehicle)
            
    def get_head_vehicle(self):

        assert self.num_vehicle(), ""

        return self.curr_vehicle[-1]

    def get_tail_vehicle(self):

        assert self.num_vehicle(), ""

        return self.curr_vehicle[0]

    def get_front_back_vehicles(self, vehicle: MicroVehicle, same_lane=False) -> Union[MicroVehicle, MicroVehicle]:
        ''' Get MicroVehicle in front of and behind a query MicroVehicle. 
        Query MicroVehicle is not necessarily in the road.

        same_lane is True if query vehicle already exists on the lane. 

        Returns: 
        - front: MicroVehicle 
        - back: MicroVehicle 
        '''

        # find proper index to insert;

        assert vehicle.position >= 0 and vehicle.position <= self.length, ""

        if self.num_vehicle() == 0:
            
            assert same_lane == False, "lane has no vehicles yet same_lane is True"

            return None, None

        for i, v in enumerate(self.curr_vehicle):

            if v.position > vehicle.position and i == 0:

                # insert at tail;
                # assert i == 0, ""

                assert v.position - vehicle.position >= (v.length + vehicle.length) * 0.5, ""

                return self.curr_vehicle[0], None

            else:

                pv = v

                # check if the vehicle collides with following vehicle;

                if same_lane == False:
                    assert vehicle.position - pv.position > (vehicle.length + pv.length) * 0.5, ""

                if i == self.num_vehicle() - 1:
                    
                    if same_lane: 
                        return None, None
                    else:
                        return None, self.curr_vehicle[-1]

                else:

                    nv = v

                    if nv.position > vehicle.position:

                        # check if the vehicle collides with leading vehicle;

                        assert nv.position - vehicle.position > (vehicle.length + nv.length) * 0.5, ""

                        if same_lane:
                            return self.curr_vehicle[i+1], self.curr_vehicle[i-1]
                        else:
                            return self.curr_vehicle[i+1], self.curr_vehicle[i]

                    else:

                        continue

                # break

    def get_mini_state(self, vm : MicroVehicle) -> Union[List[float], List[float]]:
        ''' Return a mini state vector of 3 vehicles, with vm centered in the middle, 
            surrounded by front and rear vehicles from candidate_lane.

            Returns: 
            - qp --> state vector of size 3 for position 
            - qs --> state vector of size 3 for speed
        '''
        
        # This will throw exception if 3 vehicles will result in collision; catch accordingly
        frontv, backv = self.get_front_back_vehicles(vm)

        qp = [backv.position, vm.position, frontv.position]
        qs = [backv.speed, vm.speed, frontv.speed]

        return qp, qs

    def get_accels_ministate(self, mv : MicroVehicle, delta_time : float, same_lane : bool = False):
        ''' Compute IDM for middle and back vehicles of a 3-vehicle position and velocity state.
        '''

        try:
            frontv, backv = self.get_front_back_vehicles(mv, same_lane=same_lane)
        except: 
            return -1, -1

        if frontv is None: 
            middle_pos_delta = self.head_position_delta
            middle_speed_delta = self.head_speed_delta
        else: 
            middle_pos_delta = max(frontv.position - mv.position, POSITION_DELTA_EPS)
            middle_speed_delta = frontv.speed - mv.speed

        if backv is None:
            back_pos_delta = self.head_position_delta
            back_speed_delta = self.head_speed_delta
        else: 
            back_pos_delta = max(mv.position - backv.position, POSITION_DELTA_EPS)
            back_speed_delta = mv.speed - backv.speed 

        acc_info_middle = IDM.compute_acceleration(mv.accel_max,
                                                mv.accel_pref,
                                                mv.speed,
                                                mv.target_speed,
                                                middle_pos_delta,
                                                middle_speed_delta,
                                                mv.min_space,
                                                mv.time_pref,
                                                delta_time)
        
        if backv is None: 
            acc_info_last = [-1]
        else:
            acc_info_last = IDM.compute_acceleration(backv.accel_max,
                                                    backv.accel_pref,
                                                    backv.speed,
                                                    backv.target_speed,
                                                    back_pos_delta,
                                                    back_speed_delta,
                                                    backv.min_space,
                                                    backv.time_pref,
                                                    delta_time)

        return acc_info_middle[0], acc_info_last[0]

    def mobil(self, delta_time: float):
        valid_target_lanes : Dict[int, List[MicroLane]] = {} 
        
        for li, lane in self.adjacent_lane.items():
            for vi, mv in enumerate(self.curr_vehicle):
                valid_target_lanes[vi] = [] # [self]

                try:
                    cpy_mv_normalized = copy.deepcopy(mv)
                    cpy_mv_normalized.position = mv.position / self.length * lane.length
                    _, backv = lane.get_front_back_vehicles(cpy_mv_normalized, same_lane=False)
                    frontoldv, backoldv = self.get_front_back_vehicles(mv, same_lane=True)
                except: 
                    continue 
                
                new_following_a, _ = lane.get_accels_ministate(backv, delta_time, same_lane=True)
                self_pred_a, new_following_pred_a = lane.get_accels_ministate(cpy_mv_normalized, delta_time, same_lane=False)
                self_a, old_following_a = self.get_accels_ministate(mv, delta_time, same_lane=True)
                
                if backoldv:
                    old_following_pred_a = IDM.compute_acceleration(backoldv.accel_max,
                                                    backoldv.accel_pref,
                                                    backoldv.speed,
                                                    backoldv.target_speed,
                                                    max(frontoldv.position - backoldv.position, POSITION_DELTA_EPS),
                                                    frontoldv.speed - backoldv.speed,
                                                    backoldv.min_space,
                                                    backoldv.time_pref,
                                                    delta_time)
                else: 
                    old_following_pred_a = 0


                # Is the maneuver unsafe for the new following vehicle? 

                if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                    continue

                # Do I have a planned route for a specific lane which is safe for me to access?
                if self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                    continue

                jerk = self_pred_a - self_a + self.POLITENESS * (new_following_pred_a - new_following_a
                                                                + old_following_pred_a - old_following_a)
                if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                    continue

                valid_target_lanes[vi].append(lane)
        
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

        print(len(self.curr_vehicle), len(self.next_vehicle_position))

    def lane_change_policy(self, delta_time) -> List[MicroVehicle] : 
        
        valid_lanes = self.mobil(delta_time)
        num_removed = 0
        vehicles_changed = []

        for vi, lanes in valid_lanes.items():
            
            if lanes:
                vi = vi - num_removed 
                target_lane = lanes[0]

                veh = self.curr_vehicle[vi]
                old_pos = veh.position 
                veh.position = veh.position / self.length * target_lane.length

                try:
                    target_lane.add_vehicle(veh)
                except: 
                    veh.position = old_pos
                    continue 

                self.remove_vehicle(vi)
                num_removed += 1
                vehicles_changed.append(veh)

        return vehicles_changed


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
