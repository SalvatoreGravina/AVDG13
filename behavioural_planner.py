#!/usr/bin/env python3
import numpy as np
import math

# State machine states
FOLLOW_LANE = 0
DECELERATE_TO_STOP = 1
STAY_STOPPED = 2
# Stop speed threshold
STOP_THRESHOLD = 0.02
# accuracy thresholds for trafficlight
TRAFFICLIGHT_STOP_THRESHOLD = 0.40
TRAFFICLIGHT_GO_THRESHOLD = 0.45
# distance for which a waypoint is passed
PASSED_WAYPOINT_THRESHOLD = 0.40

class BehaviouralPlanner:
    def __init__(self, lookahead, lead_vehicle_lookahead, waypoints_intersections):
        self._lookahead                      = lookahead
        self._follow_lead_vehicle_lookahead  = lead_vehicle_lookahead
        self._state                          = FOLLOW_LANE
        self._follow_lead_vehicle            = False
        self._obstacle_on_lane               = False
        self._goal_state                     = [0.0, 0.0, 0.0]
        self._goal_state_prec                = [0.0, 0.0, 0.0]
        self._goal_index                     = 0
        self._lookahead_collision_index      = 0
        self._waypoints_intersections        = waypoints_intersections
        self._detection_state                = False
        self._trafficlight_position_acquired = False
        self._trafficlight_waypoint          = [0.0, 0.0, 0.0]
        self._trafficlight_state             = []
        self._trafficlight_state1            = []
        self._closest_index                  = 0

    def set_lookahead(self, lookahead):
        self._lookahead = lookahead

    # Handles state transitions and computes the goal state.
    def transition_state(self, waypoints, ego_state, closed_loop_speed, trafficlight_state, trafficlight_state1):
        """Handles state transitions and computes the goal state.  
        
        args:
            waypoints: current waypoints to track (global frame). 
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]: 
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            closed_loop_speed: 
                current (closed-loop) speed for vehicle (m/s)
            trafficlight_state: current detections and accuracy from camera0
                labels can be "go" or "stop", accuracy is float beetween 0 and 1
                format: [[label,accuracy]]
                example: [["go",0.9042]]
            trafficlight_state1: current detections and accuracy from camera1
                labels can be "go" or "stop", accuracy is float beetween 0 and 1
                format: [[label,accuracy]]
                example: [["go",0.9042]]
        variables to set:
            self._goal_index: Goal index for the vehicle to reach
                i.e. waypoints[self._goal_index] gives the goal waypoint
            self._goal_state: Goal state for the vehicle to reach (global frame)
                format: [x_goal, y_goal, v_goal]
            self._state: The current state of the vehicle.
                available states: 
                    FOLLOW_LANE         : Follow the global waypoints (lane).
                    DECELERATE_TO_STOP  : Decelerate to stop.
                    STAY_STOPPED        : Stay stopped.
            self._trafficlight_state: list of all trafficlight detections from
                camera0
            self._trafficlight_state: list of all trafficlight detections from
                camera1
            self._detetion_state: flag that activate the detector
            self._trafficlight_position_acquired: flag that show it the position
                of the trafficlight is known
        useful_constants:
            STOP_THRESHOLD  : Stop speed threshold (m). The vehicle should fully
                              stop when its speed falls within this threshold.
            TRAFFICLIGHT_STOP_THRESHOLD : accuracy metrics for trafficlight 
                detection to be taken into account
            TRAFFICLIGHT_GO_THRESHOLD : accuracy metrics for trafficlight 
                detection to be taken into account
        """


        self._trafficlight_state.append(trafficlight_state)
        self._trafficlight_state1.append(trafficlight_state1)

        # In this state, continue tracking the lane by finding the
        # goal index in the waypoint list that is within the lookahead
        # distance. Then, check to see if the waypoint path is part of
        # an intersection. If it does, then activate the detector and
        # if a trafficlight is present, predict it's state and if
        # conditions occurred, transition to DECELERATE_TO_STOP
        if self._state == FOLLOW_LANE:

            # First, find the NEXT closest index to the ego vehicle.
            closest_len, self._closest_index = get_closest_index(waypoints, ego_state, self._closest_index)

            # Check if closest waypoint is in the vicinity of the car, if yes, go to next waypoint.
            if closest_len < PASSED_WAYPOINT_THRESHOLD:
                self._closest_index += 1
            print(closest_len)

            # Next, find the goal index that lies within the lookahead distance
            # along the waypoints.
            goal_index = self.get_goal_index(waypoints, ego_state, closest_len, self._closest_index)
            while waypoints[goal_index][2] <= 0.1: goal_index += 1

            # Print new goal index
            if self._goal_index != goal_index:
                print('nuovo waypoint: ', waypoints[goal_index])

            self._goal_index = goal_index
            self._goal_state = waypoints[goal_index]

            # Check if new waypoint is an intersection point and activate detection if so
            if np.any(np.all(np.isin(self._waypoints_intersections, self._goal_state, True), axis=1)) and self._trafficlight_position_acquired == False:
                self._detection_state = True
            elif not (np.any(np.all(np.isin(self._waypoints_intersections, self._goal_state, True), axis=1))):
                self._detection_state = False
                self._trafficlight_state = []
                self._trafficlight_state1 = []
                self._trafficlight_position_acquired = False


            state, accuracy = self.get_trafficlight_state(self._trafficlight_state, self._trafficlight_state1)
            if state == 'stop' and accuracy > TRAFFICLIGHT_STOP_THRESHOLD and self._trafficlight_position_acquired == True :
                print("Identificato semaforo rosso")
                self._goal_state_prec = np.copy(self._goal_state)
                self._goal_state[0],self._goal_state[1], self._goal_state[2] = self._trafficlight_waypoint[0], self._trafficlight_waypoint[1], 0
                print(self._goal_state)
                self._state = DECELERATE_TO_STOP
                print('passaggio a DECELERATE_TO_STOP')


        # In this state, check if we have reached a complete stop. Use the
        # closed loop speed to do so, to ensure we are actually at a complete
        # stop, and compare to STOP_THRESHOLD.  If so, transition to STAY_STOPPED.
        # Otherwise while decelerating, if the trafficlight state changes to go,
        # transition to FOLLOW_LANE
        elif self._state == DECELERATE_TO_STOP:

            if abs(closed_loop_speed) <= STOP_THRESHOLD:
                self._state = STAY_STOPPED
                print('passaggio a STAY_STOPPED')


            state, accuracy = self.get_trafficlight_state(self._trafficlight_state, self._trafficlight_state1)
            if state == 'go' and accuracy > TRAFFICLIGHT_GO_THRESHOLD:
                self._state = FOLLOW_LANE
                self._first_measure = False
                print('passaggio a FOLLOW_LANE')

        # In this state, check to see if the trafficlight state
        # changes to go, If so, we can now leave the intersection
        # and transition to FOLLOW_LANE
        elif self._state == STAY_STOPPED:

            state, accuracy = self.get_trafficlight_state(self._trafficlight_state, self._trafficlight_state1)
            if state == 'go' and accuracy > TRAFFICLIGHT_GO_THRESHOLD:
                self._state = FOLLOW_LANE
                self._first_measure = False
                self._detection_state = False
                self._trafficlight_state = []
                print('passaggio a FOLLOW_LANE')                           
                    
        else:
            raise ValueError('Invalid state value.')  
            
    # Compute state and detection accuracy of the trafficlight
    # Check if last three detected states are equals and return
    # the best accuracy beetween them
    def get_trafficlight_state(self, trafficlight_state, trafficlight_state1):
        """Given the last three detections, check their consistency and
            compute state and accuracy

            args:
                trafficlight_state: current detections and accuracy from camera0
                    labels can be "go" or "stop", accuracy is float beetween 0 and 1
                    format: [[label,accuracy]]
                    example: [["go",0.9042]]
                trafficlight_state1: current detections and accuracy from camera0
                    labels can be "go" or "stop", accuracy is float beetween 0 and 1
                    format: [[label,accuracy]]
                    example: [["go",0.9042]]
            returns:
                observation: state of the detection to be used to transition, can be None
                accuracy: score of the detection, can be None
        """
        if len(trafficlight_state) < 3 and len(trafficlight_state1) < 3:
            return None, None
        observation     = []
        accuracy = 0
        for trafficlight_frame, trafficlight_frame1 in zip(trafficlight_state[-3:], trafficlight_state1[-3:]):
            for detection, detection1 in zip(trafficlight_frame, trafficlight_frame1):

                if detection[1] > detection1[1]:
                    observation.append(detection[0])
                else:
                    observation.append(detection1[0])

                accuracy = max(accuracy, detection[1], detection1[1])

        if len(observation) < 3: 
            return None, None
            

        if observation[0] == observation[1] and observation[0] == observation[2]:
            return observation[0], accuracy
        else:
            return None, None

    # Gets the goal index in the list of waypoints, based on the lookahead and
    # the current ego state. In particular, find the earliest waypoint that has accumulated
    # arc length (including closest_len) that is greater than or equal to self._lookahead.
    def get_goal_index(self, waypoints, ego_state, closest_len, closest_index):
        """Gets the goal index for the vehicle. 
        
        Set to be the earliest waypoint that has accumulated arc length
        accumulated arc length (including closest_len) that is greater than or
        equal to self._lookahead.

        args:
            waypoints: current waypoints to track. (global frame)
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]:
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
        returns:
            wp_index: Goal index for the vehicle to reach
                i.e. waypoints[wp_index] gives the goal waypoint
        """
        # Find the farthest point along the path that is within the
        # lookahead distance of the ego vehicle.
        # Take the distance from the ego vehicle to the closest waypoint into
        # consideration.
        arc_length = closest_len
        wp_index = closest_index

        # In this case, reaching the closest waypoint is already far enough for
        # the planner.  No need to check additional waypoints.
        if arc_length > self._lookahead:
            return wp_index

        # We are already at the end of the path.
        if wp_index == len(waypoints) - 1:
            return wp_index

        # Otherwise, find our next waypoint.
        while wp_index < len(waypoints) - 1:
            arc_length += np.sqrt((waypoints[wp_index][0] - waypoints[wp_index+1][0])**2 + (waypoints[wp_index][1] - waypoints[wp_index+1][1])**2)
            if arc_length > self._lookahead: break
            wp_index += 1

        return wp_index % len(waypoints)

    # Checks to see if we need to modify our velocity profile to accomodate the
    # lead vehicle.
    def check_for_lead_vehicle(self, ego_state, lead_car_position):
        """Checks for lead vehicle within the proximity of the ego car, such
        that the ego car should begin to follow the lead vehicle.

        args:
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            lead_car_position: The [x, y] position of the lead vehicle.
                Lengths are in meters, and it is in the global frame.
            index: index of the current checked car
        sets:
            self._follow_lead_vehicle: Boolean flag on whether the ego vehicle
                should follow (true) the lead car or not (false).
        """
        # Check lead car position delta vector relative to heading, as well as
        # distance, to determine if car should be followed.

        # Check to see if lead vehicle is within range, and is ahead of us.
        if not self._follow_lead_vehicle:

            # Compute the angle between the normalized vector between the lead vehicle
            # and ego vehicle position with the ego vehicle's heading vector.
            lead_car_delta_vector = [lead_car_position[0] - ego_state[0],
                                     lead_car_position[1] - ego_state[1]]
        
            lead_car_distance = np.linalg.norm(lead_car_delta_vector)
            # In this case, the car is too far away.   
            if lead_car_distance > self._follow_lead_vehicle_lookahead:
                return

            lead_car_delta_vector = np.divide(lead_car_delta_vector, lead_car_distance)

            ego_heading_vector = [math.cos(ego_state[2]),
                                  math.sin(ego_state[2])]
            # Check to see if the relative angle between the lead vehicle and the ego
            # vehicle lies within +/- 45 degrees of the ego vehicle's heading.
            if np.dot(lead_car_delta_vector, ego_heading_vector) < (1 / math.sqrt(2)):
                return

            # Check if the vehicle are in the same orientations
            if abs(lead_car_position[2] - ego_state[2]) > math.pi / 8:
                return

            self._follow_lead_vehicle = True

        else:
            lead_car_delta_vector = [lead_car_position[0] - ego_state[0], 
                                     lead_car_position[1] - ego_state[1]]
            lead_car_distance = np.linalg.norm(lead_car_delta_vector)

            # Add a 15m buffer to prevent oscillations for the distance check.
            if lead_car_distance < self._follow_lead_vehicle_lookahead:
                return

            # Check to see if the lead vehicle is still within the ego vehicle's
            # frame of view.
            lead_car_delta_vector = np.divide(lead_car_delta_vector, lead_car_distance)
            ego_heading_vector = [math.cos(ego_state[2]), math.sin(ego_state[2])]
            if np.dot(lead_car_delta_vector, ego_heading_vector) > (1 / math.sqrt(2)):
                return

            self._follow_lead_vehicle = False

# Compute the waypoint index that is closest and next to the ego vehicle, and return
# it as well as the distance from the ego vehicle to that waypoint.
def get_closest_index(waypoints, ego_state, closest_index):
    """Gets closest index a given list of waypoints to the vehicle position.

    args:
        waypoints: current waypoints to track. (global frame)
            length and speed in m and m/s.
            (includes speed to track at each x,y location.)
            format: [[x0, y0, v0],
                     [x1, y1, v1],
                     ...
                     [xn, yn, vn]]
            example:
                waypoints[2][1]: 
                returns the 3rd waypoint's y position

                waypoints[5]:
                returns [x5, y5, v5] (6th waypoint)
        ego_state: ego state vector for the vehicle. (global frame)
            format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                ego_x and ego_y     : position (m)
                ego_yaw             : top-down orientation [-pi to pi]
                ego_open_loop_speed : open loop speed (m/s)
        closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
    returns:
        [closest_len, closest_index]:
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
    """
    closest_len = float('Inf')

    for i in range(closest_index, len(waypoints)):
        temp = (waypoints[i][0] - ego_state[0])**2 + (waypoints[i][1] - ego_state[1])**2
        if temp < closest_len:
            closest_len = temp
            closest_index = i
    closest_len = np.sqrt(closest_len)

    return closest_len, closest_index

# Checks if p2 lies on segment p1-p3, if p1, p2, p3 are collinear.        
def pointOnSegment(p1, p2, p3):
    if (p2[0] <= max(p1[0], p3[0]) and (p2[0] >= min(p1[0], p3[0])) and \
       (p2[1] <= max(p1[1], p3[1])) and (p2[1] >= min(p1[1], p3[1]))):
        return True
    else:
        return False
