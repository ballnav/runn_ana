import math


class RunningGaitCycleCounter:
    def __init__(self):
        self.cycle_count = 0
        self.previous_right_phase = None
        self.previous_left_phase = None
        self.subphase_counts = {
            "Initial Contact": 0,
            "Midstance": 0,
            "Terminal Stance": 0,
            "Pre-Swing": 0,
            "Initial Swing": 0,
            "Mid Swing": 0,
            "Terminal Swing": 0
        }
        self.total_phases = 0

    def calculate_angle(self, joint_a, joint_b, joint_c):
        vec_ab = [joint_b[0] - joint_a[0], joint_b[1] - joint_a[1]]
        vec_bc = [joint_c[0] - joint_b[0], joint_c[1] - joint_b[1]]

        dot_product = vec_ab[0] * vec_bc[0] + vec_ab[1] * vec_bc[1]
        mag_ab = math.sqrt(vec_ab[0] ** 2 + vec_ab[1] ** 2)
        mag_bc = math.sqrt(vec_bc[0] ** 2 + vec_bc[1] ** 2)

        angle = math.acos(dot_product / (mag_ab * mag_bc))
        return math.degrees(angle)

    def get_running_phase(self, right_ankle, right_knee, right_hip, left_ankle, left_knee, left_hip):
        right_side_phase = "Stance" if right_ankle[1] > right_knee[1] and right_knee[1] > right_hip[1] else "Swing"
        left_side_phase = "Stance" if left_ankle[1] > left_knee[1] and left_knee[1] > left_hip[1] else "Swing"
        return {"right_phase": right_side_phase, "left_phase": left_side_phase}

    def get_subphase(self, ankle, knee, hip):
        knee_angle = self.calculate_angle(hip, knee, ankle)
        ankle_knee_distance = math.sqrt((ankle[0] - knee[0]) ** 2 + (ankle[1] - knee[1]) ** 2)

        if ankle[1] > knee[1] and knee[1] > hip[1]:
            if knee_angle > 160 and ankle_knee_distance < 0.15:  # Adjust as needed
                return "Initial Contact"
            elif 140 < knee_angle <= 160:
                return "Midstance"
            elif 120 < knee_angle <= 140:
                return "Terminal Stance"
            else:
                return "Pre-Swing"
        else:
            if knee_angle < 60:
                return "Initial Swing"
            elif 60 <= knee_angle < 100:
                return "Mid Swing"
            else:
                return "Terminal Swing"

    def process_phases(self, right_ankle, right_knee, right_hip, left_ankle, left_knee, left_hip):
        phases = self.get_running_phase(right_ankle, right_knee, right_hip, left_ankle, left_knee, left_hip)
        right_subphase = self.get_subphase(right_ankle, right_knee, right_hip)
        left_subphase = self.get_subphase(left_ankle, left_knee, left_hip)

        self.subphase_counts[right_subphase] += 1
        self.subphase_counts[left_subphase] += 1
        self.total_phases += 2

        # Check transition for both legs to improve accuracy
        if (self.previous_right_phase == "Swing" and phases["right_phase"] == "Stance") or \
                (self.previous_left_phase == "Swing" and phases["left_phase"] == "Stance"):
            self.cycle_count += 1

        self.previous_right_phase = phases["right_phase"]
        self.previous_left_phase = phases["left_phase"]

    def get_cycle_count(self):
        return self.cycle_count

    def get_subphase_percentages(self):
        return {phase: (count / self.total_phases) * 100 for phase, count in self.subphase_counts.items()}
