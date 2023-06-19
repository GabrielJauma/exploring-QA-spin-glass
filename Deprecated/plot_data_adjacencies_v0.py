import numpy as np
import matplotlib.pyplot as plt
import Modules.figures as figs
import glob
import os.path
# plt.rcParams.update({
#     "text.usetex": True})
# %% Open and load the last file in Architecture/Data
file_type = 'Data/*.npz'
files = glob.glob(file_type)
last_simulation = max(files, key=os.path.getctime)
# last_simulation = 'Data/Model [\'hexagonal\', \'hexagonal_np_1\', \'chimera\']- binary, sizes = [ 72 128 200], T in [0.1, 1.5], N_configs = 192000, N_term = 20000, eq_steps = 1000.npz'
print(last_simulation)

data_file = np.load(last_simulation)
sizes = data_file['sizes']
adjacencies = data_file['adjacencies']
T = data_file['T']
B = data_file['B']

# %% Print data
fig, ax = plt.subplots(ncols=1, nrows=B.shape[0], dpi=400, figsize=[6, 4 * B.shape[0]])
if B.shape[0] == 1:
    ax = [ax]
for i in range(B.shape[0]):
    Ts = np.tile(T[0::2], (len(T[0::2]), 1))
    ax[i].plot(T[0::2], B[i, :, :].T, linewidth=0.5)
    # f_B = figs.multiline(T[0::2], B[i, :, :], sizes,fig, ax[i], cmap='rainbow', linewidth=1)
    # ax[i].set_title(r'$'+adjacencies[i]+'$')
    # ax[i].set_ylim([0.8, 1])
    # ax[i].set_xlim([0, 0.6])
    ax[i].set_xlabel('T')
    # if i > 0:
    #     ax[i].set_yticks([])
# cb = fig.colorbar(f_B, ticks=sizes)
ax[0].set_ylabel(r'$B$')
# fig.suptitle(r'$B \propto [\langle q_{\alpha\beta}^4 \rangle] \, / \, [\langle q_{\alpha\beta}^2 \rangle]^2$')
# fig.suptitle(last_simulation[-53:-4])
plt.tight_layout()
plt.show()
