class StatsObject:
    def __init__(self, episode, mean_reward, mean_length, mean_value, p_l, e_l, g_n, v_n, total_levels):
        self.p_l = p_l
        self.e_l = e_l
        self.g_n = g_n
        self.v_n = v_n
        self.mean_value = mean_value
        self.mean_length = mean_length
        self.mean_reward = mean_reward
        self.episode = episode
        self.total_levels = total_levels
