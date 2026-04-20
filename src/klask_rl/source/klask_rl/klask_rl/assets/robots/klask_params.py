KLASK_PARAMS = {
    "decimation": 20,  # system is running at 50Hz (night shift with 100Hz)
    "physics_dt": 0.001,
    "actuator_delay": (0.0, 0.0),
    "actuator_x_damping": 10.0,
    "actuator_y_damping": 100.0,
    "actuator_velocity_limit": 3.0,
    "actuator_x_effort_limit": 30.0,
    "actuator_y_effort_limit": 300.0,
    "joint_x_pos_limit": (-0.16, 0.16),  # slider_to_peg limits
    "joint_y1_pos_limit": (-0.21, -0.02),  # ground_to_slider_1 (player half)
    "joint_y2_pos_limit": (0.02, 0.21),  # ground_to_slider_2 (opponent half)
    "timeout": 5.0,
    "player_goal": (0.0, -0.17, 0.01905),
    "opponent_goal": (0.0, 0.17, 0.01905),
    "ball_mass_initial": 0.0017,
    "ball_reset_position_x": (-0.15, 0.15),
    "ball_reset_position_y": (-0.20, -0.04),
    "ball_restitution": 0.8,  # 0.8,  # s2r: 0.3
    "ball_static_friction": 0.18,  # 0.03,  # s2r: 0.3
    "ball_dynamic_friction": 0.12,  # 0.03,  # s2r: 0.6
    "board_static_friction": 0.09,  # 0.2
    "board_dynamic_friction": 0.06,  # 0.2
    "board_restitution": 0.9,  # 0.9
    "peg_static_friction": 0.09,  # 0.4
    "peg_dynamic_friction": 0.06,  # 0.4
    "peg_restitution": 0.7,  # 0.7
    "max_ball_vel": 5.0,  # 100.0 # s2r: 5.0
    "action_history": 0,  # s2r: 10
    "peg_radius": 0.0075,
    "collision_avoidance_decel_distance": 0.09,
    "collision_avoidance_min_clearance_factor": 1.8,
}
