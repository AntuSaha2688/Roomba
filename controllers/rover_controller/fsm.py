"""
FSM (Finite State Machine) for Rover-S Cleaner Bot.
States: CLEANING_FORWARD, OBSTACLE_AVOIDANCE, RE_ROUTING, RECOVERY, MISSION_COMPLETE

CLEANING_FORWARD now steers toward a target cell when the supervisor
provides one. Reactive states (avoidance / recovery) still take over
when sensors detect obstacles.
"""

import math
import random
from enum import Enum


class State(Enum):
    CLEANING_FORWARD = 1
    OBSTACLE_AVOIDANCE = 2
    RE_ROUTING = 3
    RECOVERY = 4
    MISSION_COMPLETE = 5


# IR sensor index groups (e-puck layout)
FRONT_SENSORS = [0, 7]
RIGHT_SENSORS = [1, 2]
LEFT_SENSORS = [5, 6]
BACK_SENSORS = [3, 4]

# Tunable parameters
OBSTACLE_THRESHOLD = 80.0
CORNER_THRESHOLD = 250.0
CRUISE_SPEED = 3.0
TURN_SPEED = 1.8
BACKUP_SPEED = -2.0
AVOIDANCE_MIN_STEPS = 15
AVOIDANCE_MAX_STEPS = 40
RECOVERY_STEPS = 70
REROUTING_STEPS = 55

# Heading control - how close to target heading before driving forward
HEADING_TOLERANCE = 0.25   # ~14 degrees


class RoverFSM:
    """
    Holds current state and produces motor speeds each step.
    Call .update(ir_values, nav_info) once per simulation step.

    nav_info is either None or a dict with:
      done (bool), and if not done: x, y, yaw, target_x, target_y
    """

    def __init__(self):
        self.state = State.CLEANING_FORWARD
        self.previous_state = None
        self.timer = 0
        self.turn_direction = 1

    def update(self, ir_values, nav_info=None):
        # --- SENSE ---
        front = max(ir_values[i] for i in FRONT_SENSORS)
        right_side = max(ir_values[i] for i in RIGHT_SENSORS)
        left_side = max(ir_values[i] for i in LEFT_SENSORS)

        self.timer += 1

        # If supervisor reports the mission is done, stop everything
        if nav_info and nav_info.get("done"):
            if self.state != State.MISSION_COMPLETE:
                self._set_state(State.MISSION_COMPLETE)

        # --- THINK (transitions) ---
        if self.state == State.CLEANING_FORWARD:
            if front > OBSTACLE_THRESHOLD:
                in_corner = (right_side > CORNER_THRESHOLD or
                             left_side > CORNER_THRESHOLD)
                self.turn_direction = -1 if right_side > left_side else 1
                if in_corner:
                    self._set_state(State.RE_ROUTING)
                else:
                    self._set_state(State.OBSTACLE_AVOIDANCE)

        elif self.state == State.OBSTACLE_AVOIDANCE:
            if self.timer > AVOIDANCE_MIN_STEPS and front < OBSTACLE_THRESHOLD * 0.5:
                self._set_state(State.CLEANING_FORWARD)
            elif self.timer > AVOIDANCE_MAX_STEPS:
                self.turn_direction = random.choice([-1, 1])
                self._set_state(State.RECOVERY)

        elif self.state == State.RE_ROUTING:
            if self.timer > REROUTING_STEPS:
                still_trapped = (front > OBSTACLE_THRESHOLD or
                                 right_side > CORNER_THRESHOLD or
                                 left_side > CORNER_THRESHOLD)
                if still_trapped:
                    self.turn_direction = random.choice([-1, 1])
                    self._set_state(State.RECOVERY)
                else:
                    self._set_state(State.CLEANING_FORWARD)

        elif self.state == State.RECOVERY:
            if (self.timer > RECOVERY_STEPS // 2 and
                    front < OBSTACLE_THRESHOLD * 0.3):
                self._set_state(State.CLEANING_FORWARD)
            elif self.timer > RECOVERY_STEPS:
                self._set_state(State.CLEANING_FORWARD)

        # --- ACT ---
        if self.state == State.CLEANING_FORWARD:
            return self._action_cleaning_forward(nav_info)

        if self.state == State.OBSTACLE_AVOIDANCE:
            return self._turn_in_place()

        if self.state == State.RE_ROUTING:
            return self._turn_in_place()

        if self.state == State.RECOVERY:
            if self.timer < RECOVERY_STEPS // 2:
                return (BACKUP_SPEED, BACKUP_SPEED)
            else:
                return self._turn_in_place()

        if self.state == State.MISSION_COMPLETE:
            return (0.0, 0.0)

        return (0.0, 0.0)

    def _action_cleaning_forward(self, nav_info):
        """Steer toward the supervisor's target. Fallback to wander if no info."""
        if nav_info and not nav_info.get("done"):
            # Vector from rover to target
            dx = nav_info["target_x"] - nav_info["x"]
            dy = nav_info["target_y"] - nav_info["y"]

            target_angle = math.atan2(dy, dx)
            angle_error = target_angle - nav_info["yaw"]

            # Normalize to [-pi, pi]
            while angle_error > math.pi:
                angle_error -= 2 * math.pi
            while angle_error < -math.pi:
                angle_error += 2 * math.pi

            if abs(angle_error) > HEADING_TOLERANCE:
                # Turn in place toward the target
                if angle_error > 0:
                    return (-TURN_SPEED * 0.6, TURN_SPEED * 0.6)
                else:
                    return (TURN_SPEED * 0.6, -TURN_SPEED * 0.6)
            else:
                # Roughly aligned - drive forward with small steering correction
                correction = angle_error * 1.5
                return (CRUISE_SPEED - correction, CRUISE_SPEED + correction)
        else:
            # No supervisor data yet - fallback to small wander
            wander = random.uniform(-0.15, 0.15)
            return (CRUISE_SPEED + wander, CRUISE_SPEED - wander)

    def _turn_in_place(self):
        if self.turn_direction > 0:
            return (TURN_SPEED, -TURN_SPEED)
        else:
            return (-TURN_SPEED, TURN_SPEED)

    def _set_state(self, new_state):
        if new_state == self.state:
            return
        print(f"[FSM] {self.state.name} -> {new_state.name}")
        self.previous_state = self.state
        self.state = new_state
        self.timer = 0