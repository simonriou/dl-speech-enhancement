import numpy as np
import matplotlib.pyplot as plt

ibm = np.load('./dataset/labels/84-121123-0000_ibm.npy')
n_frames = ibm.shape[1]
n_bins = ibm.shape[0]

time_axis = np.arange(n_frames) * (256 / 16000)
freq_axis = np.linspace(0, 8000, n_bins)

plt.figure(figsize=(10, 4))
plt.imshow(ibm, origin='lower', aspect='auto', 
            extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
            cmap='gray_r')  # white=1, black=0
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Ideal Binary Mask (IBM)")
plt.colorbar(label="Mask value")
plt.tight_layout()
plt.show()