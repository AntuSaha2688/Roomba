"""
Rover-S Cleaner Bot - main controller (single self-supervising node).

The e-puck runs as a Supervisor, so this one controller does the whole
Sense -> Think -> Act loop without any inter-robot radio link:

  SENSE   read the 8 IR proximity sensors and read the robot's own pose
          (position + heading) directly via the Supervisor API.
  THINK   ask CoverageGrid for the nearest uncleaned cell (the navigation
          goal) and hand that plus the sensor readings to the FSM, which
          decides what the robot should do.
  ACT     drive the two wheel motors at the speeds the FSM returns.

Responsibilities are kept separate on purpose:
  - coverage.py  -> WHERE to go (room memory / coverage map)
  - fsm.py       -> HOW to behave (state machine, obstacle handling, safety)
  - this file    -> wiring sensors and actuators to that logic.

Author: Antu Saha
Project: 3003ICT Programming for Robotics
"""

import math

from controller import Supervisor
from fsm import RoverFSM
from coverage import CoverageGrid

TIME_STEP = 64  # ms per control step

# --- Mission tuning -------------------------------------------------------- #
# The mission is complete when NO reachable cell is left uncleaned - not at an
# arbitrary percentage. Cells physically under obstacles are pre-marked blocked
# (see detect_obstacles below) so they never count as reachable targets.

# Each attempt at a goal cell is allowed this many steps WITHOUT making
# distance progress before we give up on it for now. Kept short (~19 s sim
# time) so the robot doesn't oscillate forever in front of a tricky cell -
# it bails out and tries elsewhere, then retries from a fresh approach.
TARGET_TIMEOUT_STEPS = 300

# How many failed attempts before we PERMANENTLY mark a cell BLOCKED. Lower
# values give up faster (more cells end up red); higher values make the robot
# more persistent at the cost of total run time.
MAX_RETRY_ATTEMPTS = 3


def compute_yaw(orientation):
    """
    Extract the heading (rotation about the vertical Z axis) from a Webots
    3x3 orientation matrix returned as a flat 9-element list.

    Row-major layout: [R00 R01 R02  R10 R11 R12  R20 R21 R22].
    yaw = atan2(R10, R00) -> indices 3 and 0.
    """
    return math.atan2(orientation[3], orientation[0])


def detect_arena_size(supervisor):
    """
    Read the RectangleArena's floorSize from the world so the coverage grid can
    auto-size to the room. Returns (size_x, size_y) in metres, or None if no
    arena is found (caller then falls back to a default).
    """
    children = supervisor.getRoot().getField("children")
    for k in range(children.getCount()):
        node = children.getMFNode(k)
        if node.getTypeName() == "RectangleArena":
            field = node.getField("floorSize")
            if field is not None:
                sx, sy = field.getSFVec2f()
                return sx, sy
    return None


def detect_obstacles(supervisor):
    """
    Scan the world for WoodenBox obstacles and return their floor footprints as
    a list of (cx, cy, half_x, half_y) in world metres. The supervisor reads
    each box's real translation and size, so the coverage map stays correct
    even if boxes are moved or resized in the world file.
    """
    boxes = []
    children = supervisor.getRoot().getField("children")
    for k in range(children.getCount()):
        node = children.getMFNode(k)
        if node.getTypeName() != "WoodenBox":
            continue
        tx, ty, _ = node.getField("translation").getSFVec3f()
        size_field = node.getField("size")
        if size_field is not None:
            sx, sy, _ = size_field.getSFVec3f()
        else:
            sx, sy = 0.6, 0.6  # WoodenBox default size
        boxes.append((tx, ty, sx / 2.0, sy / 2.0))
    return boxes


def main():
    robot = Supervisor()

    # The robot node is its own "self" - used to read true pose each step.
    self_node = robot.getSelf()
    if self_node is None:
        print("[Rover-S] ERROR: getSelf() returned None. "
              "Is 'supervisor TRUE' set on the e-puck?")
        return

    # --- Actuators: wheel motors in velocity-control mode --- #
    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    # --- Sensors: 8 IR proximity sensors (ps0..ps7) --- #
    ir_sensors = []
    for i in range(8):
        sensor = robot.getDevice(f"ps{i}")
        sensor.enable(TIME_STEP)
        ir_sensors.append(sensor)

    # --- Processing modules --- #
    fsm = RoverFSM()

    # Auto-size the coverage grid to the actual room dimensions.
    arena = detect_arena_size(robot)
    if arena is None:
        arena = (5.0, 5.0)
        print("[Rover-S] No RectangleArena found; defaulting to 5.0x5.0 m room.")
    size_x, size_y = arena
    grid = CoverageGrid(robot, size_x=size_x, size_y=size_y,
                        cell_size=0.25, cleaning_radius=0.08)
    print(f"[Rover-S] Arena {size_x:.1f}x{size_y:.1f} m -> grid "
          f"{grid.n_cols}x{grid.n_rows} (cell ~{grid.cell_x:.2f} m).")

    # Pre-mark cells under obstacles so they are never chased as targets.
    # A small margin (~2 cm) keeps cells whose centre is fractionally outside
    # the box from being false-blocked; the cleaning-sweep radius (20 cm) is
    # what actually keeps the robot safe at runtime.
    obstacles = detect_obstacles(robot)
    blocked = grid.mark_obstacle_footprints(obstacles, margin=0.02)
    print(f"[Rover-S] Detected {len(obstacles)} obstacle(s); "
          f"pre-blocked {blocked} cells.")
    print(f"[Rover-S] Started. Initial state: {fsm.state.name}")

    # --- Goal-tracking state ---------------------------------------------- #
    # current_goal is the (i, j) cell the robot is committed to driving toward.
    # We keep the same goal across many steps (instead of recomputing every
    # frame) so the robot drives in long straight runs - that's what gives the
    # Roomba-like motion. The goal only changes when:
    #   1. The cell has been cleaned (usually by the cleaning sweep as the
    #      robot gets close), or
    #   2. The robot has stopped making progress toward it for a long time
    #      (then we declare it unreachable and pick a new one).
    current_goal = None
    best_dist = float("inf")  # closest we've gotten to current_goal
    target_age = 0            # steps without progress toward current_goal
    blocked_revealed = False  # red no-go tiles drawn once, on completion

    # Cells we are currently giving up on, with the number of failed attempts
    # so far. They are temporarily excluded from target selection so the robot
    # tries elsewhere first, then comes back to them with a fresh approach.
    # Only after MAX_RETRY_ATTEMPTS without progress does a cell get
    # permanently marked BLOCKED.
    skip_attempts = {}        # (i, j) -> int

    # --- Sense -> Think -> Act loop --- #
    while robot.step(TIME_STEP) != -1:
        # === SENSE ===
        ir_values = [s.getValue() for s in ir_sensors]

        pos = self_node.getPosition()
        x, y = pos[0], pos[1]
        yaw = compute_yaw(self_node.getOrientation())

        # === THINK ===
        # Update the coverage map with where we are now. Several cells may be
        # cleaned in one step thanks to the cleaning-sweep radius.
        newly_cleaned, coverage = grid.mark_visited(x, y)
        if newly_cleaned:
            cells = "cell" if newly_cleaned == 1 else "cells"
            print(f"[Rover-S] Cleaned {newly_cleaned} new {cells}. "
                  f"Coverage: {coverage * 100:.1f}%")

        # --- Goal management ---
        # Drop the current goal if it has already been cleaned (almost always
        # this happens because the cleaning sweep covered it as we approached).
        if (current_goal is not None and
                grid.cells[current_goal[0]][current_goal[1]] != 0):  # 0 == UNVISITED
            current_goal = None

        # Acquire a new goal if we have none. Exclude cells we have recently
        # given up on so the robot moves to a different area.
        if current_goal is None:
            current_goal = grid.find_nearest_unvisited(
                x, y, skip=set(skip_attempts.keys()))
            if current_goal is None and skip_attempts:
                # Everything else is done - retry the skipped cells now,
                # hopefully from a better angle than last time.
                print(f"[Rover-S] Retrying {len(skip_attempts)} skipped cell(s).")
                skip_attempts.clear()
                current_goal = grid.find_nearest_unvisited(x, y)
            if current_goal is not None:
                gx, gy = grid.grid_to_world(*current_goal)
                best_dist = math.hypot(gx - x, gy - y)
                target_age = 0

        # Decide what to tell the FSM this step.
        if current_goal is None:
            # Truly nothing left - mission complete.
            nav_info = {"done": True}
            if not blocked_revealed:
                grid.render_blocked()
                blocked_revealed = True
                print(f"[Rover-S] Coverage complete: {coverage * 100:.1f}%. "
                      "Obstacle zones marked.")
        else:
            gx, gy = grid.grid_to_world(*current_goal)
            cur_dist = math.hypot(gx - x, gy - y)

            # Progress-based timeout: only count steps when the robot is NOT
            # getting meaningfully closer to the goal. This way, time spent
            # in OBSTACLE_AVOIDANCE / RECOVERY (which is real, useful work)
            # does not accidentally burn the budget for an otherwise
            # reachable cell.
            if cur_dist < best_dist - 0.02:  # >= 2 cm of progress
                best_dist = cur_dist
                target_age = 0
            else:
                target_age += 1

            if target_age > TARGET_TIMEOUT_STEPS:
                # Bail out of this goal. Try other cells first; come back
                # later. Only mark BLOCKED after MAX_RETRY_ATTEMPTS failures.
                attempts = skip_attempts.get(current_goal, 0) + 1
                if attempts >= MAX_RETRY_ATTEMPTS:
                    print(f"[Rover-S] Cell {current_goal} unreachable after "
                          f"{attempts} attempts - marking blocked.")
                    grid.mark_blocked(*current_goal)
                    skip_attempts.pop(current_goal, None)
                else:
                    print(f"[Rover-S] No progress on {current_goal} "
                          f"(attempt {attempts}/{MAX_RETRY_ATTEMPTS}) - "
                          "trying elsewhere first.")
                    skip_attempts[current_goal] = attempts
                current_goal = None
                # Use a do-nothing nav for this step; next iteration picks the
                # new goal cleanly without sending the FSM stale coordinates.
                nav_info = {"done": False,
                            "x": x, "y": y, "yaw": yaw,
                            "target_x": x, "target_y": y}
            else:
                nav_info = {"done": False,
                            "x": x, "y": y, "yaw": yaw,
                            "target_x": gx, "target_y": gy}

        # The FSM turns sensor readings + goal into wheel speeds.
        left_speed, right_speed = fsm.update(ir_values, nav_info)

        # === ACT ===
        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)


if __name__ == "__main__":
    main()
