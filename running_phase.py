class RunningGaitCycleCounter:
    def __init__(self):
        self.cycle_count = 0
        self.previous_right_phase = None
        self.previous_left_phase = None

    def get_running_phase(self, right_ankle, right_knee, right_hip, left_ankle, left_knee, left_hip):
        # Define the threshold for Flight Phase (you might need to adjust this value)
        flight_threshold = 5  # Example threshold value, adjust as needed

        # Check the phase for the right side
        if right_ankle[1] > right_knee[1] and right_knee[1] > right_hip[1]:
            right_side_phase = "Stance"
        elif right_ankle[1] > flight_threshold and left_ankle[1] > flight_threshold:
            right_side_phase = "Flight"
        else:
            right_side_phase = "Swing"

        # Check the phase for the left side
        if left_ankle[1] > left_knee[1] and left_knee[1] > left_hip[1]:
            left_side_phase = "Stance"
        elif right_ankle[1] > flight_threshold and left_ankle[1] > flight_threshold:
            left_side_phase = "Flight"
        else:
            left_side_phase = "Swing"

        return {"right_phase": right_side_phase, "left_phase": left_side_phase}

    def process_phases(self, right_ankle, right_knee, right_hip, left_ankle, left_knee, left_hip):
        phases = self.get_running_phase(right_ankle, right_knee, right_hip, left_ankle, left_knee, left_hip)

        if self.previous_right_phase == "Swing" and phases["right_phase"] == "Stance":
            self.cycle_count += 1

        self.previous_right_phase = phases["right_phase"]
        self.previous_left_phase = phases["left_phase"]

    def get_cycle_count(self):
        return self.cycle_count
