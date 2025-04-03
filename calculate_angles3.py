import math
def calculate_angle(a, b, c):
    vector_ab = (b[0] - a[0], b[1] - a[1])
    vector_bc = (c[0] - b[0], c[1] - b[1])
    dot_product = vector_ab[0] * vector_bc[0] + vector_ab[1] * vector_bc[1]
    magnitude_ab = math.sqrt(vector_ab[0] ** 2 + vector_ab[1] ** 2)
    magnitude_bc = math.sqrt(vector_bc[0] ** 2 + vector_bc[1] ** 2)
    cos_angle = dot_product / (magnitude_ab * magnitude_bc)
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    return angle_deg
def calculate_angle_2(a, b, c):
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = math.degrees(radians)
    return (angle)
def calculate_trunk_lean(a, b):
    vertical_line = (a[0], a[1] - 1)
    return calculate_angle_2(vertical_line, a, b)

def evaluate_angle(value, optimal_range):
    if optimal_range[0] < (value) < optimal_range[1]:
        return "Accurate "
    else:
        return "Inaccurate "

def get_text_color(value, optimal_range):
    if optimal_range[0] <= (value) <= optimal_range[1]:
        return (0, 255, 0)
    else:
        return (0, 0, 255)

def get_point(value, optimal_range):
    if optimal_range[0] <= (value) <= optimal_range[1]:
        return 1
    else:
        return 0

def evaluate_each_body(value, optimal_range):
    if optimal_range[0] < (value):
        return "Good"
    elif optimal_range[0] - 21 < (value) :
        return "Satisfactory"
    else:
        return "Should Improve"

def count_conditions(values, optimal_range):
    result_counts = {"Good": 0, "Satisfactory": 0, "Should Improve": 0}

    for value in values:
        condition = evaluate_each_body(value, optimal_range)
        result_counts[condition] += 1

    return result_counts


def calculate_distance(point1, point2):
    """คำนวณระยะห่างระหว่างจุดสองจุด"""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_stride_length(landmarks, mp_pose):
    """คำนวณความยาวก้าวเดิน (Stride Length) โดยใช้ตำแหน่งของเท้าซ้ายและขวา"""
    left_foot = (landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y)
    right_foot = (landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y)

    stride_length = calculate_distance(left_foot, right_foot)
    return stride_length