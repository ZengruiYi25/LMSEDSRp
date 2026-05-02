import shutil
import os.path

import numpy as np
import pandas as pd
from tqdm import tqdm
from thop import profile

import torch
import torch.nn as nn

from utils import get_res, read_data, plot_res

num_seed = 252618
np.random.seed(num_seed)
torch.manual_seed(num_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  === Important parameters ===
MODEL = ['DSR', 'LSEDSR', 'LMSEDSR', 'LMSEDSRp'][3]  # Choose the model
if_all = True  # use all data (True: multi-material) or only N87 data (False: single-material)
h = 256  # the dimension of latent space

# === Training parameters ===
batch_size = 1024 if if_all else 128
learning_rate = 1e-3
num_epochs = 12000

# =============================
Net = None
loss_list = []
coefs_list = []
    
mre_train_list = []
mre_val_list = []
mre_test_list = []
mre_epoch_list = []

coef_names = ['k', 'alpha', 'beta', 'gamma', 'zeta', 'eta']

mre_val_min = 9999999
mre_val_min_epoch = 0  
mre_val_min_train = 0
mre_val_min_test = 0

exec("from Models.{} import Net".format(MODEL))
root_file = './O_O/_{}_{}'.format(MODEL, 'All' if if_all else 'N87')
save_file = root_file + "/{}".format(h)

if not os.path.exists(root_file):
    os.mkdir(root_file)
if os.path.exists(save_file):
    shutil.rmtree(save_file)
os.mkdir(save_file)
os.mkdir(save_file + "/Figs")
os.mkdir(save_file + "/Argus")


if __name__ == '__main__':
    # Load data
    train_dl, train_data, val_data, test_data, train_label, val_label, test_label = read_data(batch_size=batch_size, if_all=if_all)

    # Initialize the network
    net = Net(latent=h).to(device)

    # Test the number of parameters and flops
    net.eval()
    flops, params = profile(net, inputs=(train_data[0].to(device).unsqueeze(0), ), verbose=False)
    txt0 = "Parameters = %d(%.2fM)(%.2fK)\n FLOPs = %d(%.2fM)\n\n" % (params, params / 1e6, params / 1e3, flops, flops / 1e6)
    print(" ================== Net ==================  ")
    print("  The number of parameters is %d(%.2fM)(%.2fK)\n"
          "  The number of flops is %d(%.2fM)" % (params, params / 1e6, params / 1e3, flops, flops / 1e6))

    # Define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.NAdam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.98)

    # Training loop
    for epoch in range(num_epochs+1):
        loss_list_temp = []
        net.train()
        torch.cuda.empty_cache()  # clear cache
        data_loader_tqdm = tqdm(train_dl, desc='epoch: {}'.format(epoch))
        for step, (batch_x, batch_y) in enumerate(data_loader_tqdm):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            coefs, P = net(batch_x)

            loss = loss_fn(P, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list_temp.append(loss.item())
            coefs = coefs.mean(0).detach().cpu().numpy()
            coefs_list.append(coefs)

            # show information
            information = dict(zip(
                ['lr'] + ['loss'] + coef_names[:len(coefs)] + ['mre_val_min_test'],
                [optimizer.state_dict()['param_groups'][0]['lr'], loss.cpu().item()] + coefs.tolist() + [mre_val_min_test]
            ))
            data_loader_tqdm.set_postfix(information)

        loss_list.append(np.mean(loss_list_temp))
        if optimizer.state_dict()['param_groups'][0]['lr'] > 1e-5:
            scheduler.step()

        # Evaluate the model every 50 epochs
        if epoch % 50 == 0:
            net.eval()
            res_matx = np.zeros([3, 8])  # 
            mre_epoch_list.append(epoch)
            
            # Training set
            with torch.no_grad():
                coefs, predict = net(train_data.to(device))
                predict = torch.exp(predict)
            MRE, MRE_max, MAE, MSE, RMSE, MSLE, EVS, R2 = get_res(train_label, predict.cpu().numpy())
            txt_train = 'Train: MRE = {:.2f}%({:.2f}%), MAE = {}, MSE = {}, RMSE = {}, MSLE = {:.4f}, EVS = {:.4f}, R2 = {:.4f}\n' \
                .format(MRE * 100, MRE_max * 100, MAE, MSE, RMSE, MSLE, EVS, R2)
            print(txt_train, end='')
            mre_train = MRE
            mre_train_list.append(mre_train)
            res_matx[0, :] = [MRE, MRE_max, MAE, MSE, RMSE, MSLE, EVS, R2]

            # Evaluation set
            with torch.no_grad():
                coefs, predict = net(val_data.to(device))
                predict = torch.exp(predict)
            MRE, MRE_max, MAE, MSE, RMSE, MSLE, EVS, R2 = get_res(val_label, predict.cpu().numpy())
            txt_val = 'Val: MRE = {:.2f}%({:.2f}%), MAE = {}, MSE = {}, RMSE = {}, MSLE = {:.4f}, EVS = {:.4f}, R2 = {:.4f}\n' \
                .format(MRE * 100, MRE_max * 100, MAE, MSE, RMSE, MSLE, EVS, R2)
            print(txt_val, end='')
            mre_val = MRE
            mre_val_list.append(mre_val)
            res_matx[1, :] = [MRE, MRE_max, MAE, MSE, RMSE, MSLE, EVS, R2]

            # Test set
            with torch.no_grad():
                coefs, predict = net(test_data.to(device))
                predict = torch.exp(predict)
            MRE, MRE_max, MAE, MSE, RMSE, MSLE, EVS, R2 = get_res(test_label, predict.cpu().numpy())
            txt_test = 'Test: MRE = {:.2f}%({:.2f}%), MAE = {}, MSE = {}, RMSE = {}, MSLE = {:.4f}, EVS = {:.4f}, R2 = {:.4f}\n' \
                .format(MRE * 100, MRE_max * 100, MAE, MSE, RMSE, MSLE, EVS, R2)
            print(txt_test, end='')
            mre_test = MRE
            mre_test_list.append(mre_test)
            res_matx[2, :] = [MRE, MRE_max, MAE, MSE, RMSE, MSLE, EVS, R2]

            # Save the best model
            if mre_val_min > mre_val:
                mre_val_min = mre_test
                mre_val_min_epoch = epoch
                mre_val_min_train = mre_train
                mre_val_min_test = mre_test

                with open(save_file + '/result.txt', 'w', encoding='utf-8') as f:
                    f.write(txt0)
                    f.write('Epoch = {}\n'.format(epoch))
                    f.write(txt_train)
                    f.write(txt_val)
                    f.write(txt_test)
                    print('Saved ...')
                res_pd = pd.DataFrame(res_matx)
                writer = pd.ExcelWriter(save_file + '/result.xlsx')
                res_pd.to_excel(writer, 'page_1', float_format='%.5f')
                writer.close()
                torch.save(net.state_dict(), save_file + '/Argus/Net')

            # Plot results including loss curve, MRE curve and coefficient curve
            plot_res(save_file, loss_list, mre_epoch_list, mre_train_list, mre_val_list, mre_test_list, coefs_list)
            
print('Finished!')
