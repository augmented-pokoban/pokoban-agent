import matplotlib.pyplot as plt
import numpy as np

from autoencoder.EncoderData import load_object

# data = load_object('mse_data.pkl.zip')
data = load_object('agent_error_data.pkl.zip')

data = np.mean(data, axis=1)
print(data.shape)

for step in range(5):
    x = [step+1]
    plt.scatter(x, data[step], facecolors='none', edgecolors='b')

print('showing...')
plt.show()

