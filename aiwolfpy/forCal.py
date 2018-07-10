def dead_or_alive(arrayRoll, arrayAlive):
	roll_set = set(arrayRoll)
	alive_set = set(arrayAlive)
	matched_list = list(roll_set & alive_set)
	return matched_list 