import torch
import torch.nn as nn
from gymnasium import Wrapper
from klask_rl.assets.robots import klask_params


class ActuatorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(ActuatorNetwork, self).__init__()
        # Following the excerpt: 3 hidden layers of 32 units each
        # Use softsign activation
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        # softsign activation: y = x/(1+|x|)
        self.activation = nn.Softsign()

    def forward(self, x_commands, x_states=None):
        if x_states is not None:
            x = torch.cat([x_commands, x_states], dim=1)
        else:
            x = torch.cat([x_commands], dim=1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc_out(x)
        return x


class ActuatorModelWrapper(Wrapper):

    model_file = "source/klask_rl/klask_rl/tasks/manager_based/klask_rl/actuator_model/model_odrive_new_estimator_history_10_interval_0.02_delay_0.0_horizon3_with_states.pt"
    num_history_steps = 10  # Number of past commands to include in input
    hidden_dim = 64
    delay = 0  # Time delay between most recent command included in the input and the current time
    include_states = True  # Whether to include history of peg states in input
    EDGE_x = torch.tensor([-0.16, 0.16])
    EDGE_player_y = torch.tensor([-0.02, -0.22])
    EDGE_opponent_y = torch.tensor([0.02, 0.22])
    DISTANCE_PER_REVOLUTION = 0.04
    DEACCELERATION_DISTANCE = 0.09
    PEG_RADIUS = 0.0075

    def __init__(self, env, device="cuda", model_file=None, pos_idx=slice(0, 2), vel_idx=slice(2, 4)):
        super().__init__(env)

        self.pos_idx = pos_idx
        self.vel_idx = vel_idx

        num_envs = env.unwrapped.num_envs
        self.dT = klask_params.KLASK_PARAMS["decimation"] * klask_params.KLASK_PARAMS["physics_dt"]
        input_dim = self.num_history_steps * 2 + self.include_states * (self.num_history_steps - 1) * 2
        output_dim = 2
        self.model = ActuatorNetwork(input_dim, output_dim, hidden_dim=self.hidden_dim).to(device)
        if model_file is None:
            model_file = self.model_file
        self.model.load_state_dict(torch.load(model_file, map_location=device))
        self.command_buffer_1 = torch.zeros(num_envs, 2 * self.num_history_steps, dtype=torch.float32).to(device)
        self.command_buffer_2 = torch.zeros(num_envs, 2 * self.num_history_steps, dtype=torch.float32).to(device)
        if self.include_states:
            self.state_buffer_1 = torch.zeros(num_envs, 2 * (self.num_history_steps - 1), dtype=torch.float32).to(
                device
            )
            self.state_buffer_2 = torch.zeros(num_envs, 2 * (self.num_history_steps - 1), dtype=torch.float32).to(
                device
            )
            self.position_player = torch.zeros(num_envs, 2, dtype=torch.float32).to(device)
            self.position_opponent = torch.zeros(num_envs, 2, dtype=torch.float32).to(device)

        self.model.eval()

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)

        # Peg 1 history update:
        state_1 = obs["policy"][:, self.vel_idx]

        if self.include_states:
            self.state_buffer_1[:, :] = state_1.repeat(1, self.num_history_steps - 1)
        self.command_buffer_1[:, :] = 0.0

        # Peg 2 history update:
        state_2 = obs["opponent"][:, self.vel_idx]

        if self.include_states:
            self.state_buffer_2[:, :] = state_2.repeat(1, self.num_history_steps - 1)
        self.command_buffer_2[:, :] = 0.0

        return obs, info

    def step(self, actions, *args, **kwargs):
        # Peg 1 command history update:

        # actions = self.apply_boundary_limits(actions)

        command_1 = actions[:, :2]
        prev_buffer = self.command_buffer_1.clone()
        self.command_buffer_1[:, 2:] = prev_buffer[:, :-2]
        self.command_buffer_1[:, :2] = command_1

        # Peg 2 command history update:
        command_2 = actions[:, 2:]
        self.command_buffer_2[:, 2:] = self.command_buffer_2.clone()[:, :-2]
        self.command_buffer_2[:, :2] = command_2

        # Map commands to velocities using the actuator model:
        if self.include_states:
            states_input_1, states_input_2 = self.state_buffer_1, self.state_buffer_2
        else:
            states_input_1, states_input_2 = None, None

        with torch.no_grad():
            actions_1 = self.model(self.command_buffer_1, states_input_1)
            actions_2 = self.model(self.command_buffer_2, states_input_2)

        actions[:, :2] = actions_1
        actions[:, 2:] = actions_2

        # print()
        # print(self.state_buffer_1.min(), self.state_buffer_1.max())
        # print(self.command_buffer_1.min(), self.command_buffer_1.max())
        obs, rew, terminated, truncated, info = self.env.step(actions, *args, **kwargs)

        if self.include_states:
            # Peg 1 state history update:
            self.position_player = obs["policy"][:, self.pos_idx]
            state_1 = obs["policy"][:, self.vel_idx]

            self.state_buffer_1[:, 2:] = self.state_buffer_1.clone()[:, :-2]
            self.state_buffer_1[:, :2] = state_1

            # Peg 2 state history update:
            self.position_opponent = obs["opponent"][:, self.pos_idx]
            state_2 = obs["opponent"][:, self.vel_idx]

            self.state_buffer_2[:, 2:] = self.state_buffer_2.clone()[:, :-2]
            self.state_buffer_2[:, :2] = state_2

        # Reset buffers for terminated envs:
        done = terminated | truncated
        self.command_buffer_1[done, :] = 0.0
        self.command_buffer_2[done, :] = 0.0
        if self.include_states:
            self.state_buffer_1[done, :] = state_1[done].repeat(1, self.num_history_steps - 1)
            self.state_buffer_2[done, :] = state_2[done].repeat(1, self.num_history_steps - 1)

        return obs, rew, terminated, truncated, info


#    def apply_boundary_limits(self, actions):
#
#            # Lower boundary check (EDGE[2])
#        player_at_middle =  self.position_player[:,1] >= self.EDGE_player_y[0] - self.DEACCELERATION_DISTANCE-self.PEG_RADIUS
#        player_at_end =self.position_player[:,1]<= self.EDGE_player_y[1]+self.DEACCELERATION_DISTANCE+self.PEG_RADIUS
#
#        opponent_at_middle = self.position_opponent[:,1]<= self.EDGE_opponent_y[0] +self.DEACCELERATION_DISTANCE+self.PEG_RADIUS
#        opponent_at_end = self.position_opponent[:,1] >= self.EDGE_opponent_y[1]-self.DEACCELERATION_DISTANCE-self.PEG_RADIUS
#
#        player_at_front = self.position_player[:,0]<= self.EDGE_x[0]+self.DEACCELERATION_DISTANCE+self.PEG_RADIUS
#        player_at_back = self.position_player[:,0]>=self.EDGE_x[0]-self.DEACCELERATION_DISTANCE-self.PEG_RADIUS
#
#        opponent_at_front =self.position_player[:,1]<= self.EDGE_x[0]+self.DEACCELERATION_DISTANCE+self.PEG_RADIUS
#        opponent_at_back = self.position_player[:,1]<= self.EDGE_x[0]-self.DEACCELERATION_DISTANCE-self.PEG_RADIUS
#        return
#        actions[:, 0][player_at_front] = torch.maximum
#        actions[:, 0][player_at_back] =
#        actions[:, 1][player_at_middle] =
#        actions[:, 1][player_at_end] =
#        actions[:, 2][opponent_at_front] = torch.maximum
#        actions[:, 2][opponent_at_back] =
#        actions[:, 3][opponent_at_middle] =
#        actions[:, 3][opponent_at_end] =
#
#        v_y[lower_zone & fully_lower] = torch.maximum(v_y[lower_zone & fully_lower], torch.tensor(0.0, device=device))
#        v_y[lower_zone & ~fully_lower] = torch.maximum(
#            v_y[lower_zone & ~fully_lower],
#            -interpolate_vel(y_pos[lower_zone & ~fully_lower] - EDGE_YMIN - PEG_RADIUS)
#        )
#
#        # Upper boundary check (EDGE[3])
#        upper_zone = y_pos + DEACCEL_DIST + PEG_RADIUS >= EDGE_YMAX
#        fully_upper = y_pos + PEG_RADIUS >= EDGE_YMAX
#
#        v_y[upper_zone & fully_upper] = torch.minimum(v_y[upper_zone & fully_upper], torch.tensor(0.0, device=device))
#        v_y[upper_zone & ~fully_upper] = torch.minimum(
#            v_y[upper_zone & ~fully_upper],
#            interpolate_vel(EDGE_YMAX - y_pos[upper_zone & ~fully_upper] + PEG_RADIUS)
#        )
#
#        return v_y
