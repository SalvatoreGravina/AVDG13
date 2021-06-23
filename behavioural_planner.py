#!/usr/bin/env python3
import logging
import numpy as np
import math

# State machine states
FOLLOW_LANE = 0
DECELERATE_TO_STOP = 1
STAY_STOPPED = 2
# Stop speed threshold
STOP_THRESHOLD = 0.02
# Number of cycles before moving from stop sign.
STOP_COUNTS = 10
# distance threshold for depth camera
DEPTH_THRESHOLD = 10

class BehaviouralPlanner:
    def __init__(self, lookahead, lead_vehicle_lookahead, waypoints_intersections):
        self._lookahead                      = lookahead
        self._follow_lead_vehicle_lookahead  = lead_vehicle_lookahead
        self._state                          = FOLLOW_LANE
        self._follow_lead_vehicle            = False
        self._obstacle_on_lane               = False
        self._ego_state_prec                 = [0.0, 0.0, 0.0, 0.0]
        self._trafficlight_distance_prec     = 0
        self._first_measure                  = False
        self._goal_state                     = [0.0, 0.0, 0.0]
        self._goal_state_prec                = [0.0, 0.0, 0.0]
        self._goal_index                     = 0
        self._lookahead_collision_index      = 0
        self._waypoints_intersections        = waypoints_intersections
        self._detection_state                = False
        self._trafficlight_position          = [0.0,0.0]
        self._trafficlight_position_acquired = False
        self._trafficlight_waypoint          = [0.0, 0.0, 0.0]

    def set_lookahead(self, lookahead):
        self._lookahead = lookahead

    # Handles state transitions and computes the goal state.
    def transition_state(self, waypoints, ego_state, closed_loop_speed, trafficlight_state, trafficlight_distance):
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
            closed_loop_speed: current (closed-loop) speed for vehicle (m/s)
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
            self._stop_count: Counter used to count the number of cycles which
                the vehicle was in the STAY_STOPPED state so far.
        useful_constants:
            STOP_THRESHOLD  : Stop speed threshold (m). The vehicle should fully
                              stop when its speed falls within this threshold.
            STOP_COUNTS     : Number of cycles (simulation iterations) 
                              before moving from stop sign.
        """
        # In this state, continue tracking the lane by finding the
        # goal index in the waypoint list that is within the lookahead
        # distance. Then, check to see if the waypoint path intersects
        # with any stop lines. If it does, then ensure that the goal
        # state enforces the car to be stopped before the stop line.
        # You should use the get_closest_index(), get_goal_index(), and
        # check_for_stop_signs() helper functions.
        # Make sure that get_closest_index() and get_goal_index() functions are
        # complete, and examine the check_for_stop_signs() function to
        # understand it.



        if self._state == FOLLOW_LANE:
            #print("FOLLOW_LANE")
            # First, find the closest index to the ego vehicle.
            closest_len, closest_index = get_closest_index(waypoints, ego_state)

            # Next, find the goal index that lies within the lookahead distance
            # along the waypoints.
            goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
            while waypoints[goal_index][2] <= 0.1: goal_index += 1

            # PRINT DEL GOAL INDEX SOLO QUANDO CAMBIA (PROBLEMA CON L'INCROCIO DRITTO)
            if self._goal_index != goal_index:
                print('nuovo waypoint: ',waypoints[goal_index])

            self._goal_index = goal_index
            self._goal_state = waypoints[goal_index]
            #if self._goal_state in self._waypoints_intersections:
            #if np.any(np.all(np.isin(self._waypoints_intersections, self._goal_state, True), axis=1)): 
            if np.any(np.all(np.isin(self._waypoints_intersections, self._goal_state, True), axis=1)) and self._trafficlight_position_acquired == True:
                pass
            elif np.any(np.all(np.isin(self._waypoints_intersections, self._goal_state, True), axis=1)) and self._trafficlight_position_acquired == False:
                self._detection_state = True
            else:
                self._detection_state = False
                self._trafficlight_position_acquired = False

            state, accuracy = self.get_trafficlight_state(trafficlight_state)
            if state == 'stop' and accuracy > 0.40 and self._trafficlight_position_acquired == True :
                print("Identificato semaforo rosso")
                self._goal_state_prec = np.copy(self._goal_state)
                self._goal_state[0],self._goal_state[1], self._goal_state[2] = self._trafficlight_waypoint[0], self._trafficlight_waypoint[1], 0
                self._state = DECELERATE_TO_STOP
                logging.info('passaggio a DECELERATE_TO_STOP')


        # In this state, check if we have reached a complete stop. Use the
        # closed loop speed to do so, to ensure we are actually at a complete
        # stop, and compare to STOP_THRESHOLD.  If so, transition to the next
        # state.
        elif self._state == DECELERATE_TO_STOP:
            if abs(closed_loop_speed) <= STOP_THRESHOLD:
                self._state = STAY_STOPPED
                logging.info('passaggio a STAY_STOPPED')
            state, accuracy = self.get_trafficlight_state(trafficlight_state)
            if state == 'go' and accuracy > 0.45:
                self._state = FOLLOW_LANE
                self._first_measure = False
                logging.info('passaggio a FOLLOW_LANE')

        # In this state, check to see if we have stayed stopped for at
        # least STOP_COUNTS number of cycles. If so, we can now leave
        # the stop sign and transition to the next state.
        elif self._state == STAY_STOPPED:

            # We have stayed stopped for the required number of cycles.
            # Allow the ego vehicle to leave the stop sign. Once it has
            # passed the stop sign, return to lane following
            # You should use the get_closest_index(), get_goal_index(), and 
            # check_for_stop_signs() helper functions.
            state, accuracy = self.get_trafficlight_state(trafficlight_state)
            if state == 'go' and accuracy > 0.45:
                self._state = FOLLOW_LANE
                self._first_measure = False
                self._detection_state = False
                logging.info('passaggio a FOLLOW_LANE')                           
                    

            # If the stop sign is no longer along our path, we can now
            # transition back to our lane following state.
                    
            #if not stop_sign_found: self._state = FOLLOW_LANE


        else:
            raise ValueError('Invalid state value.')  

        #Valutare spostamento nel main

        if self._trafficlight_position_acquired == False:
            for detection in trafficlight_state:
                if detection[1]>0.30:
                    if self._first_measure == False:
                        if trafficlight_distance < DEPTH_THRESHOLD: # treshold for depth camera
                            self._ego_state_prec = ego_state
                            self._trafficlight_distance_prec = trafficlight_distance
                            self._first_measure = True
                    elif trafficlight_distance < DEPTH_THRESHOLD:
                        try: 
                            self._trafficlight_waypoint = self.get_trafficlight_waypoint(ego_state,trafficlight_distance, self._ego_state_prec, self._trafficlight_distance_prec, self._goal_state)
                        except: 
                            self._first_measure = False
                            break
                        self._trafficlight_position_acquired = True
                        print('tl position acquired')
 
    # Aggiungere descrizione chiatta
    def get_trafficlight_waypoint(self, ego_state, trafficlight_distance, ego_state_prec, trafficlight_distance_prec, goal_state):
        a_b = math.sqrt((ego_state[0] - ego_state_prec[0]) ** 2 + (ego_state[1] - ego_state_prec[1]) ** 2)
        b_c = trafficlight_distance
        a_c = trafficlight_distance_prec

        xp, yp = self.triangulate(a_b, b_c, a_c)
        x, y = self.coordinate_to_world(xp, yp, ego_state_prec[0], ego_state_prec[1], ego_state[2])

        self._trafficlight_position[0] = x
        self._trafficlight_position[1] = y
        trafficlight_waypoint = self.projection(self._trafficlight_position, ego_state, goal_state)
        return trafficlight_waypoint

    def get_trafficlight_state(self, trafficlight_state):
        if len(trafficlight_state) < 3:
            return None, None
        observation = []
        for trafficlight_frame in trafficlight_state[-3:]:
            for detection in trafficlight_frame:
                observation.append(detection[0])

        if len(observation) < 3: 
            return None, None
        if observation[0] == observation[1] and observation[0] == observation[2]:
            return observation[0], trafficlight_state[-1][0][1]
        else:
            return None, None


    # Triangulate a point
    def triangulate(self, AB, BC, AC):

        y = (AB ** 2 + AC ** 2 - BC ** 2) / (2 * AB)
        x = math.sqrt(AC ** 2 - y ** 2)
        
        return x, y

    # transform coordinate to world frame
    def coordinate_to_world(self, xp, yp, Ox, Oy, R):

        x = Ox + xp * math.cos(R) - yp * math.sin(R)
        y = Oy + xp * math.sin(R) + yp * math.cos(R)
        
        return x, y

    # Project point on a straight line
    def projection(self, project_point, a_point, b_point):

        p1 = np.array([a_point[0], a_point[1]])
        p2 = np.array([b_point[0], b_point[1]])
        p3 = np.array([project_point[0], project_point[1]])
        l2 = np.sum((p1-p2)**2)

        t = max(0, min(1, np.sum((p3 - p1) * (p2 - p1)) / l2))

        return p1 + t * (p2 - p1)

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
            # vehicle lies within +/- 15 degrees of the ego vehicle's heading.
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

# Compute the waypoint index that is closest to the ego vehicle, and return
# it as well as the distance from the ego vehicle to that waypoint.
def get_closest_index(waypoints, ego_state):
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

    returns:
        [closest_len, closest_index]:
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
    """
    closest_len = float('Inf')
    closest_index = 0

    for i in range(len(waypoints)):
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
