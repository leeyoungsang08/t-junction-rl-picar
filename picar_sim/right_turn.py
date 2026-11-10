"""Unprotected right turn, granted by the central PC (PiCars cannot yield on their own).

Rule (signal.unprotected_right = true), used by the SUMO environment and the real central server alike:
a car on a red approach may turn right only when it is stopped at the stop line (ARRIVE: "stop, then turn"),
the junction is green and healthy (not yellow / clear / fault), no car from another approach is inside the box
or holds a reservation, and no car on the green approach is approaching. While it crosses, cars from other
approaches are held. So it only uses idle moments of the green phase (no cars coming on the green approach)
instead of waiting for min-green, yellow and all-red.

Why this strict: lanes are 0.21 m and the 0.24 x 0.17 m PiCar turns on the spot in the box, so its corners
sweep 0.147 m around the turn point, about 4 cm into the neighbouring lane; two right turns from different
approaches meet 0.21 m apart (2D pre-check collision at J2: W->N and N->E). A right turn therefore conflicts
with every movement from another approach; movements from the same approach follow each other.
With the flag off the behaviour is exactly the old per-approach GO/STOP (active then never holds another arm).
"""
from picar_sim.supervisor import APPROACHES

RIGHT = {("J", "E", "N"), ("J", "N", "W"), ("J2", "W", "N"), ("J2", "N", "E")}
RESERVATION_S = 3.0        # a right-turn grant not used within this time is dropped


def arm_of(edge):
    return edge.split("_")[0] if edge else None


def is_right(jn, arm_in, arm_out):
    return (jn, arm_in, arm_out) in RIGHT


def conflicts(jn, m1, m2):
    """Different approaches always conflict in the box (spot turns sweep into the next lane, see above)."""
    return m1[0] != m2[0]


def may_enter(jn, move, status, enabled, active, green_queue, state):
    """move = (arm_in, arm_out) of the car at the head of its lane; status = SupervisorStatus;
    active = movements inside the box or holding a right-turn reservation;
    green_queue = movements of APPROACH/ARRIVE cars on the green approach; state = the car's ledger state."""
    if status.stage != "green" or status.fault is not None or None in move:
        return False
    green = APPROACHES[status.phase]
    if move[0] == green:
        return not any(conflicts(jn, move, m) for m in active)
    if not (enabled and is_right(jn, *move) and state == "ARRIVE"):
        return False
    return not any(conflicts(jn, move, m) for m in list(active) + list(green_queue))
