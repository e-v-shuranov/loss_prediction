import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from os import walk
from torch.utils.data import Dataset
import os
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
cpu_count = os.cpu_count()
num_workers = cpu_count if device == "cpu" else 0
num_workers, cpu_count

class metr_dataset(Dataset):
    def __init__(self, x, y=None, device="cpu"):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.y is not None:
            return \
                torch.tensor(self.x[idx]),\
                torch.tensor(self.y[idx])

        return torch.tensor(self.x[idx])
def estimate_loss():
    # s = StringIO('"alpha, #42", 10.0\n"beta, #64", 2.0\n')
    # dtype = np.dtype([("label", "U12"), ("value", float)])
    # np.loadtxt(s, dtype=dtype, delimiter=",", quotechar='"')
    # iter 0: loss 10.8834, time 10234.83ms, mfu -100.00%
    dir_path = "C:\Evgenii\projects\metr_test"
    filenames = next(walk(dir_path), (None, None, []))[2]  # [] if no file
    all_xx_data = np.empty(0)
    all_yy_data = np.empty(0)
    i = 0
    for name in filenames:

        # path = "C:\Evgenii\projects\metr_test\examples_0504_2000.txt"
        path = dir_path + "\\" + name
        # dtype = np.dtype([("l_iter", int, "l_loss", float, "l_time", "time", "l_mfv", "mfv")])

        dtype = {'names': ("l_iter", "iter", "l_loss", "loss", "l_time", "time", "l_mfv", "mfv"),
                 'formats': ('S10', 'S10', 'S10','S10', 'S10', 'S10', 'S10', 'S10')}
        data = np.loadtxt(path, dtype=dtype, delimiter=" ", quotechar='"')
        x_data = data[:]["iter"]
        x_data = [int(x[:-1]) for x in x_data]
        n_embd = int(path[39:43])
        n_layer = 4
        block_size = 1024
        batch_size = 64

        x1 = [n_embd for x in x_data]


        x2 = [n_layer * n_embd * n_embd * 12 * 6 * block_size * batch_size * max_iters for max_iters in x_data]

        xx_data = np.column_stack((x_data, x1, x2))
        y_data = data[:]["loss"]
        y_data = [float(x[:-1]) for x in y_data]

        xx_data = xx_data[1:-1]
        y_data = y_data[1:-1]
        print(xx_data,y_data)
        if i == 0:
            all_xx_data = xx_data
            all_yy_data = y_data
        else:
            all_xx_data = np.concatenate((all_xx_data, xx_data), axis=0)
            all_yy_data = np.concatenate((all_yy_data, y_data), axis=0)

        i +=1


    x_train, x_test, y_train, y_test = train_test_split(all_xx_data, all_yy_data, test_size=0.2, random_state=42)
    # x_train = all_xx_data
    # y_train = all_yy_data

    print("!!!Shapes:",x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # Sample data
    # x_train = np.linspace(0, 10, 100)
    # y_train = np.sin(x_train)

    # Convert to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    # y_train = y_train + (0.3 ** 0.5) * torch.randn(y_train.shape[0], y_train.shape[1])

    # Define the neural network
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(3, 50)
            self.bn1 = nn.BatchNorm1d(50)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(50, 500)
            self.bn2 = nn.BatchNorm1d(500)
            self.fc3 = nn.Linear(500, 50)
            self.bn3 = nn.BatchNorm1d(50)
            self.fc4 = nn.Linear(50, 1)
            # self.bn1 = nn.BatchNorm2d(num_features=500)

        def forward(self, x):
            x = self.bn1(self.fc1(x))
            x = self.relu(x)
            x = self.bn2(self.fc2(x))
            x = self.relu(x)
            x = self.bn3(self.fc3(x))
            x = self.relu(x)
            x = self.fc4(x)

            return x

    train_dset = metr_dataset(x_train, y_train, device=device)
    val_dset = metr_dataset(x_test, y_test, device=device)
    train_loader = DataLoader(train_dset, batch_size=50, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dset, batch_size=50, shuffle=False, num_workers=num_workers)

    # Initialize the model, define the loss function and the optimizer
    model = SimpleNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005,  weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # swa_model = torch.optim.swa_utils.AveragedModel(model)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    # swa_start = 160
    # swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=0.0005)

    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        for i_batch, sample_batched in enumerate(train_loader):
            x, y = sample_batched
            model.train()
            optimizer.zero_grad()
            outputs = model(x[:,0,:])
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # if epoch > swa_start:
        #     swa_model.update_parameters(model)
        #     swa_scheduler.step()
        # else:
        #     scheduler.step()

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    # Extrapolate new data
    # x_new = torch.tensor([[12]], dtype=torch.float32)  # Example value outside training range
    # model.eval()
    # y_pred = model(x_new)
    # print(f'Predicted value at x=12: {y_pred.item()}')

    # Plot the results
    # x_test = np.linspace(0, 12, 120)
    x_test_torch = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        y_test_pred = model(x_test_torch[:,0,:]).numpy()

    plt.plot(x_train[:,:,2][:,0].numpy(), y_train[:,0].numpy(), 'bo', label='Training data')
    plt.plot(x_test[:,2], y_test_pred[:,0], 'ro', label='Model prediction')

    # plt.plot(x_train[:,:,2][:,0].numpy(), y_train[:,0].numpy(), 'bo', label='Training data')
    # plt.plot(x_test[:,2], y_test_pred[:,0,0], 'r-', label='Model prediction')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    estimate_loss()
