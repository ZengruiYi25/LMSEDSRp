import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, explained_variance_score, r2_score

num_seed = 252618
np.random.seed(num_seed)
torch.manual_seed(num_seed)


colors = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'] + ['gold', 'deeppink'])

Materials_list = ['3C90', '3C94', '3E6', '3F4', '77', '78', 'N27', 'N30', 'N49', 'N87']

D_list = np.array([[0, 0]] +
                  [[round(i, 1), 0] for i in np.linspace(0.1, 0.9, 9)] +
                  [[0.1, 0.1], [0.1, 0.3], [0.1, 0.5], [0.1, 0.7],
                   [0.2, 0.2], [0.2, 0.4], [0.2, 0.6],
                   [0.3, 0.1], [0.3, 0.3], [0.3, 0.5],
                   [0.4, 0.2], [0.4, 0.4],
                   [0.5, 0.1], [0.5, 0.3],
                   [0.6, 0.2],
                   [0.7, 0.1]])


def read_data(batch_size, if_all=True):
    # Load data
    data_file = '.\\Datasets\\'
    Data_M = np.load(data_file + 'Data_M.npy', allow_pickle=True)  # Material type
    Data_W = np.load(data_file + 'Data_W.npy', allow_pickle=True)  # Waveform type
    Data_D = np.load(data_file + 'Data_D.npy', allow_pickle=True)  # Duty cycle
    Data_B = np.load(data_file + 'Data_B.npy', allow_pickle=True)  # Magnetic flux density
    Data_H = np.load(data_file + 'Data_H.npy', allow_pickle=True)  # Magnetic field strength
    Data_F = np.load(data_file + 'Data_F.npy', allow_pickle=True)  # Frequency
    Data_C = np.load(data_file + 'Data_C.npy', allow_pickle=True)  # Temperature
    Data_P = np.load(data_file + 'Data_P.npy', allow_pickle=True)  # Core loss

    # If not using all data, filter for N87 material
    if not if_all:
        idx = Data_M == 9
        Data_M = Data_M[idx]
        Data_W = Data_W[idx]
        Data_D = Data_D[idx]
        Data_B = Data_B[idx]
        Data_H = Data_H[idx]
        Data_F = Data_F[idx]
        Data_C = Data_C[idx]
        Data_P = Data_P[idx]

    N = Data_P.shape[0]

    # Waveform alignment
    for i in tqdm(range(N)):
        if not Data_W[i] == 0:
            sig = Data_B[i]
            sig = (sig - sig.min()) / (sig.max() - sig.min())
            # Linear fitting
            k, b = np.polyfit(np.arange(64), sig[:64], 1)
            # Calculate offset
            dev_shift = int(b / k)
            # Assign
            Data_B[i] = np.roll(Data_B[i], shift=dev_shift, axis=-1)

    # Duty cycle labels
    label_D = []
    for i in tqdm(range(len(Data_D))):
        a = Data_D[i]
        for j in range(len(D_list)):
            b = D_list[j]
            if np.abs(a - b).sum() < 1e-5:  
                label_D.append(j)
                break
        else:
            print('Error Duty cycle label')
    label_D = np.array(label_D)

    # Calculate other values
    Bm = Data_B.max(-1, keepdims=True)
    Hdc = Data_H.mean(-1, keepdims=True)
    Delta_B = Data_B.max(-1, keepdims=True) - Data_B.min(-1, keepdims=True)
    dt = 2e-6 / 1024  # each sample point's time interval
    dB_dt = np.diff(Data_B) / dt
    f_dB_dt = np.sum(dB_dt ** 2, axis=-1, keepdims=True) * dt
    f_sin = 2/(Delta_B**2 * np.pi ** 2) * f_dB_dt  # corrected frequency （MSE）

    # Logarithmic transformation
    Data_F = np.log(Data_F)
    f_sin = np.log(f_sin)
    Bm = np.log(Bm)
    Data_C = np.log(Data_C)

    # Concatenate data and labels
    data = np.concatenate((
        Data_M.reshape(-1, 1),
        Data_F.reshape(-1, 1),
        f_sin,
        Bm,
        Data_C.reshape(-1, 1),
        Hdc,
        Data_D,
        Data_B), axis=-1)
    label = np.concatenate((Data_P.reshape(-1, 1), label_D.reshape(-1, 1)), axis=-1)
    print(data.shape, label.shape, label_D.shape, Counter(label_D))

    # Split data into training, validation, and test sets
    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2, random_state=num_seed)
    train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.25, random_state=num_seed)

    print("Train.shape: ", train_data.shape)
    print("Val.shape: ", val_data.shape)
    print("Test.shape: ", test_data.shape)

    # Convert to PyTorch tensors and create DataLoader
    train_data = torch.tensor(train_data, dtype=torch.float32)
    val_data = torch.tensor(val_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)

    train_ds = TensorDataset(train_data, torch.log(torch.tensor(train_label[:, 0], dtype=torch.float32)))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_dl, train_data, val_data, test_data, train_label[:, 0], val_label[:, 0], test_label[:, 0]


def get_res(y, predict):
    # Calculate errors
    MRE = np.mean(np.abs(y - predict) / y)

    MRE_max = np.max(np.abs(y - predict) / y)

    MAE = mean_absolute_error(y, predict)

    MSE = mean_squared_error(y, predict)

    RMSE = np.sqrt(MSE)

    MSLE = mean_squared_log_error(y, predict)

    EVS = explained_variance_score(y, predict)

    R2 = r2_score(y, predict)

    return MRE, MRE_max, MAE, MSE, RMSE, MSLE, EVS, R2


def plot_res(save_file, loss_list, mre_epoch_list, mre_train_list, mre_val_list, mre_test_list, coef_list):
    coef_list = np.array(coef_list).T
    # Loss curve
    plt.figure(figsize=(8, 5), dpi=300)
    plt.grid()  
    plt.plot(loss_list, linewidth=3, markersize='3')
    plt.tick_params(labelsize=10)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.savefig(save_file + "/Figs/Loss.png", bbox_inches='tight')
    plt.close('all')  

    # MRE curve
    plt.figure(figsize=(8, 5), dpi=300)
    plt.grid()  
    plt.plot(mre_epoch_list, mre_train_list, linewidth=3, markersize='3')
    plt.plot(mre_epoch_list, mre_val_list, linewidth=3, markersize='3')
    plt.plot(mre_epoch_list, mre_test_list, linewidth=3, markersize='3')
    plt.tick_params(labelsize=10)
    plt.xlabel('Epoch', fontsize=15)
    plt.legend(['MRE_train', 'MRE_val', 'MRE_test'], prop={'size': 15},
                markerscale=3., bbox_to_anchor=(0.5, -0.26), loc=8, ncol=7, handlelength=0.6, frameon=False, columnspacing=1)
    plt.savefig(save_file + "/Figs/MRE.png", bbox_inches='tight')
    plt.close('all')  

    # Coefficient curve
    plt.figure(figsize=(8, 5), dpi=300)
    plt.grid()  
    for i in range(len(coef_list)):
        plt.plot(coef_list[i], linewidth=3, markersize='3')
    plt.tick_params(labelsize=10)
    plt.xlabel('Step', fontsize=15)
    plt.legend([r'$k$', r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\zeta$', r'$\eta$'][:len(coef_list)], prop={'size': 15}, markerscale=3., bbox_to_anchor=(0.5, -0.26), loc=8, ncol=7, handlelength=0.6, frameon=False, columnspacing=1)
    plt.savefig(save_file + "/Figs/Coefs.png", bbox_inches='tight')
    plt.close('all')  

    # Save arguments and results
    np.save(save_file + '/Argus/loss_list.npy', loss_list)
    np.save(save_file + '/Argus/mre_epoch_list.npy', mre_epoch_list)
    np.save(save_file + '/Argus/mre_train_list.npy', mre_train_list)
    np.save(save_file + '/Argus/mre_val_list.npy', mre_val_list)
    np.save(save_file + '/Argus/mre_test_list.npy', mre_test_list)
    np.save(save_file + '/Argus/coef_list.npy', coef_list)