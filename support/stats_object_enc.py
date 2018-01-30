class StatsObjectEnc:
    def __init__(self, episode, mean_enc_loss, mean_val_loss, mean_suc_loss, test_enc_loss, test_val_loss, test_suc_loss):
        self.test_suc_loss = test_suc_loss
        self.test_val_loss = test_val_loss
        self.test_enc_loss = test_enc_loss
        self.mean_suc_loss = mean_suc_loss
        self.mean_val_loss = mean_val_loss
        self.mean_enc_loss = mean_enc_loss
        self.episode = episode
