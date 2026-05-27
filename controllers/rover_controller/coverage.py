"""
Coverage grid for the Rover-S Cleaner Bot.

This module turns the room into a 2D occupancy grid and tracks which cells the
robot has already cleaned. It is the "memory" of the system: the FSM handles
*how* to move, while CoverageGrid decides *where* to go next (the nearest cell
that has not been cleaned yet) and reports how much of the room is done.

The grid auto-sizes to the room: pass the arena's width/height (in metres) and a
target cell size, and it works out how many columns/rows it needs. This means
changing the room dimensions in the world file needs no code changes - the
controller reads the arena size and hands it to this class.

It also draws a tile on each resolved cell (green = cleaned, red = obstacle
no-go zone) so coverage is visible in the Webots 3D view. Tile drawing requires
a Supervisor handle; with None the grid still works (no drawing), which keeps
the class easy to unit-test outside Webots.

Author: Antu Saha
Project: 3003ICT Programming for Robotics
"""

# Cell life-cycle states
UNVISITED = 0   # not cleaned yet - a valid navigation target
VISITED = 1     # cleaned (robot drove through it)
BLOCKED = 2     # known to be unreachable (under an obstacle / repeatedly failed)

# Tile colours (baseColor, emissiveColor) per cell kind.
_TILE_STYLE = {
    VISITED: ("0.2 0.85 0.35", "0.1 0.4 0.15"),    # green = cleaned floor
    BLOCKED: ("0.55 0.12 0.12", "0.2 0.04 0.04"),   # red   = obstacle / no-go zone
}


def _tile_node_string(x, y, size, base_color, emissive_color):
    """Webots node text for a thin coloured tile centred at (x, y), just above the floor."""
    return (
        "Solid { "
        f"translation {x} {y} 0.002 "
        "children [ Shape { appearance PBRAppearance { "
        f"baseColor {base_color} emissiveColor {emissive_color} "
        "roughness 1 metalness 0 } "
        f"geometry Box {{ size {size} {size} 0.002 }} }} ] }}"
    )


class CoverageGrid:
    """
    A grid laid over the (origin-centred) room floor, auto-sized to the room.

    supervisor : Supervisor or None - used only to spawn visualization tiles.
    size_x, size_y : room dimensions in metres (the arena's floorSize).
    cell_size : desired cell edge in metres; the real cell size is rounded so a
                whole number of cells fits each axis.
    """

    def __init__(self, supervisor=None, size_x=5.0, size_y=5.0, cell_size=0.25,
                 cleaning_radius=0.15):
        """
        cleaning_radius : metres. The robot is treated as a circular cleaning
        sweep of this radius - a cell is marked cleaned whenever the robot's
        body passes within this distance of the cell's centre, not only when
        the exact centre enters the cell. This stops cells next to obstacles
        getting stuck unvisited because IR avoidance deflects the robot just
        before it reaches the cell's centre.
        """
        self.supervisor = supervisor
        self.half_x = size_x / 2.0
        self.half_y = size_y / 2.0
        self.cleaning_radius = cleaning_radius

        # Choose a whole number of cells per axis closest to the target size.
        self.n_cols = max(1, round(size_x / cell_size))
        self.n_rows = max(1, round(size_y / cell_size))
        self.cell_x = size_x / self.n_cols
        self.cell_y = size_y / self.n_rows
        self.tile_size = min(self.cell_x, self.cell_y) * 0.9  # small visible border

        self.cells = [[UNVISITED] * self.n_rows for _ in range(self.n_cols)]
        self.total_cells = self.n_cols * self.n_rows
        self.visited_count = 0  # cleaned + blocked = "resolved"

        # Cache the root "children" field so we can spawn tiles at runtime.
        self._root_children = None
        if supervisor is not None:
            self._root_children = supervisor.getRoot().getField("children")

    # ---- coordinate conversion ----
    def world_to_grid(self, x, y):
        """Convert a world (x, y) position to clamped grid indices (i, j)."""
        i = int((x + self.half_x) / self.cell_x)
        j = int((y + self.half_y) / self.cell_y)
        i = max(0, min(self.n_cols - 1, i))
        j = max(0, min(self.n_rows - 1, j))
        return i, j

    def grid_to_world(self, i, j):
        """Convert grid indices (i, j) to the world (x, y) at the cell's centre."""
        x = (i + 0.5) * self.cell_x - self.half_x
        y = (j + 0.5) * self.cell_y - self.half_y
        return x, y

    # ---- marking cells ----
    def mark_visited(self, x, y):
        """
        Mark cleaned cells around the robot. Two things happen each call:

        1. The cell containing (x, y) is ALWAYS marked - the robot occupied it,
           so it counts as cleaned regardless of cleaning_radius. This is the
           guarantee that the robot's own footprint never gets "missed" just
           because the cell centre happens to be more than cleaning_radius
           away (a cell half-diagonal can exceed a small radius).
        2. Any other UNVISITED cell whose centre lies within cleaning_radius of
           (x, y) is also marked. Models the robot as a circular cleaning
           sweep, so cells right next to obstacles get cleaned as the robot
           passes by without having to enter the cell's exact centre.

        Returns (newly_cleaned, coverage_fraction); newly_cleaned is the count
        of cells just cleaned this call.
        """
        newly = 0

        # (1) Always mark the cell the robot is currently in.
        i0, j0 = self.world_to_grid(x, y)
        if self.cells[i0][j0] == UNVISITED:
            self.cells[i0][j0] = VISITED
            self.visited_count += 1
            self._spawn_tile(i0, j0, VISITED)
            newly += 1

        # (2) Sweep any other cells whose centres are within cleaning_radius.
        r = self.cleaning_radius
        if r > 0.0:
            i_min, j_min = self.world_to_grid(x - r, y - r)
            i_max, j_max = self.world_to_grid(x + r, y + r)
            r2 = r * r
            for i in range(i_min, i_max + 1):
                for j in range(j_min, j_max + 1):
                    if (i, j) == (i0, j0):
                        continue
                    if self.cells[i][j] != UNVISITED:
                        continue
                    cx, cy = self.grid_to_world(i, j)
                    if (cx - x) ** 2 + (cy - y) ** 2 <= r2:
                        self.cells[i][j] = VISITED
                        self.visited_count += 1
                        self._spawn_tile(i, j, VISITED)
                        newly += 1
        return newly, self.coverage()

    def mark_blocked(self, i, j):
        """
        Flag a cell as unreachable so it is never chased as a target again.
        No tile is drawn here - blocked cells are only revealed (in red) once
        the mission completes, via render_blocked().
        """
        if self.cells[i][j] == UNVISITED:
            self.cells[i][j] = BLOCKED
            self.visited_count += 1  # count toward resolved so coverage completes

    def mark_obstacle_footprints(self, boxes, margin=0.06):
        """
        Pre-mark every cell under a known obstacle as BLOCKED so the robot never
        picks an impossible target inside a box. This is what lets "mission
        complete" mean *all reachable floor is clean* rather than a fixed %.
        No tiles are drawn here (see render_blocked).

        boxes : list of (cx, cy, half_x, half_y) footprints in world metres.
        margin : extra clearance (robot body radius) around each box.
        Returns the number of cells blocked.
        """
        blocked = 0
        for i in range(self.n_cols):
            for j in range(self.n_rows):
                if self.cells[i][j] != UNVISITED:
                    continue
                cx, cy = self.grid_to_world(i, j)
                for bx, by, hx, hy in boxes:
                    if (bx - hx - margin <= cx <= bx + hx + margin and
                            by - hy - margin <= cy <= by + hy + margin):
                        self.cells[i][j] = BLOCKED
                        self.visited_count += 1
                        blocked += 1
                        break
        return blocked

    def render_blocked(self):
        """
        Draw a red tile on every BLOCKED cell. Call this once when the mission
        completes so obstacle no-go zones are revealed only at the end, after
        the green coverage has filled in.
        """
        for i in range(self.n_cols):
            for j in range(self.n_rows):
                if self.cells[i][j] == BLOCKED:
                    self._spawn_tile(i, j, BLOCKED)

    # ---- querying ----
    def find_nearest_unvisited(self, x, y, skip=None):
        """
        Return grid indices (i, j) of the closest UNVISITED cell to (x, y), or
        None if every cell is VISITED, BLOCKED, or in the optional `skip` set.

        `skip` is a set of (i, j) cells the caller is temporarily ignoring -
        e.g. cells that the robot couldn't make progress on and wants to retry
        later from a different angle. Uses squared Euclidean distance in grid
        space; cheap and good enough for target selection.
        """
        ci, cj = self.world_to_grid(x, y)
        best = None
        best_dist = float("inf")
        for i in range(self.n_cols):
            for j in range(self.n_rows):
                if self.cells[i][j] != UNVISITED:
                    continue
                if skip is not None and (i, j) in skip:
                    continue
                d = (i - ci) ** 2 + (j - cj) ** 2
                if d < best_dist:
                    best_dist = d
                    best = (i, j)
        return best

    def coverage(self):
        """Fraction of cells that are resolved (cleaned or blocked), 0.0-1.0."""
        return self.visited_count / self.total_cells

    def is_complete(self, threshold=1.0):
        """True once the resolved fraction reaches `threshold`."""
        return self.coverage() >= threshold

    # ---- visualization ----
    def _spawn_tile(self, i, j, kind=VISITED):
        """
        Draw a tile at cell (i, j): green for cleaned cells, red for obstacle
        no-go cells. No-op when running without a Supervisor.
        """
        if self._root_children is None:
            return
        base_color, emissive_color = _TILE_STYLE[kind]
        x, y = self.grid_to_world(i, j)
        self._root_children.importMFNodeFromString(
            -1, _tile_node_string(x, y, self.tile_size, base_color, emissive_color))
