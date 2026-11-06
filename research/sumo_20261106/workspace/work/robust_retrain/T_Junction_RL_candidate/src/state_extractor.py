from src.state_adapter import SensorState
from src.utils import DIRECTIONS


def extract(connection, controller, vehicle_data=None):
    if vehicle_data is not None:
        groups = [[v for v in vehicle_data.values() if v["road"] == f"{d}_in"] for d in DIRECTIONS]
        queues = [sum(v["speed"] < 0.1 for v in g) for g in groups]
        waits = [[v["wait"] for v in g] for g in groups]
        internal = [v for v in vehicle_data.values() if v["road"].startswith(":")]
        state = SensorState([len(g) for g in groups], [sum(w) / len(w) if w else 0 for w in waits],
                            len(internal), controller.phase, controller.green_elapsed, controller.transitioning)
        return state, {"queues": queues, "max_direction_wait": [max(w, default=0) for w in waits],
                       "internal_stopped": sum(v["speed"] < 0.1 for v in internal)}
    lanes = [f"{d}_in_0" for d in DIRECTIONS]
    ids = [connection.lane.getLastStepVehicleIDs(lane) for lane in lanes]
    waits = [[connection.vehicle.getWaitingTime(vid) for vid in group] for group in ids]
    queues = [connection.lane.getLastStepHaltingNumber(lane) for lane in lanes]
    internal_ids = [vid for vid in connection.vehicle.getIDList() if connection.vehicle.getRoadID(vid).startswith(":")]
    state = SensorState([len(v) for v in ids], [sum(v) / len(v) if v else 0 for v in waits],
                        len(internal_ids), controller.phase, controller.green_elapsed, controller.transitioning)
    return state, {"queues": queues, "max_direction_wait": [max(v, default=0) for v in waits],
                   "internal_stopped": sum(connection.vehicle.getSpeed(vid) < 0.1 for vid in internal_ids)}
