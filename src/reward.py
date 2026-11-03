def reward_components(queues, stopped_seconds, internal_stopped, max_waits, switch_events, config, *, pending=0, internal_count=0):
    terms = {
        "queue_cost": sum(queues) / config["queue_scale"],
        "waiting_cost": (stopped_seconds + pending) / config["waiting_scale"],
        "internal_cost": (internal_stopped + max(0, internal_count - config.get("internal_free_occupancy", float("inf")))) / config["internal_scale"],
        "starvation_cost": sum(max(0, t - config["starvation_seconds"]) for t in max_waits) / config["starvation_scale"],
        "switch_cost": float(switch_events),
    }
    if config["mode"] == "simple":
        total = -config["queue_weight"] * terms["queue_cost"]
    elif config["mode"] == "composite":
        total = -sum(terms[k + "_cost"] * config[k + "_weight"] for k in ("waiting", "internal", "starvation", "switch"))
        total -= terms["queue_cost"] * config.get("queue_weight_composite", config["queue_weight"])
    else:
        raise ValueError("reward mode must be simple or composite")
    return float(total), terms
