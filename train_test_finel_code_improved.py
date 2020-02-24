import csv
import os
import numpy as np
import math, random
import statistics as stat
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler, StandardScaler

device = torch.device("cuda")

n_samples = 1
output_dim = 1
input_dim = 1
hidden_size = 12
num_layers = 2

figures_path = '/home/shirili/Downloads/ShirirliDorin/project_A/data_125Hz/2642420-8079/test_on_others'

r_path = '/home/shirili/Downloads/ShirirliDorin/project_A/data_125Hz/2642420-8079/nn_16Sec_2642420-8079.pt' # TODO: the path to save/load the net


def read_csv(path):
    i = 0
    row_count = 0
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count = sum(1 for line in csv_reader)
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = np.zeros((row_count, n_samples))
        for row in csv_reader:
            data[i, 0] = (float(row[0]))  # TODO: row 0 or row 1? depends on the csv file
            i = i + 1
    return row_count, data


def calc_loss_precentage(expec_BP, pred_BP, time_length, patient_num, start_time, graph_name):
    interval_30_sec = 3700
    expec_95 = []
    expec_5 = []
    pred_95 = []
    pred_5 = []
    for i in range(int(time_length / interval_30_sec)):
        expec_95.append(np.percentile(expec_BP[i * interval_30_sec:(i + 1) * interval_30_sec], 95))
        expec_5.append(np.percentile(expec_BP[i * interval_30_sec:(i + 1) * interval_30_sec], 5))
        pred_95.append(np.percentile(pred_BP[i * interval_30_sec:(i + 1) * interval_30_sec], 98))
        pred_5.append(np.percentile(pred_BP[i * interval_30_sec:(i + 1) * interval_30_sec], 2))
    plt.clf()
    X = np.linspace(start_time, ((time_length / interval_30_sec) / 2) + start_time,
                    (time_length / interval_30_sec))  # TODO: change start time + add it to end time
    plt.plot(X, expec_95, label='expec_95')
    plt.plot(X, expec_5, label='expec_5')
    plt.plot(X, pred_95, label='pred_98')
    plt.plot(X, pred_5, label='pred_2')
    plt.xlabel('time [minute]')
    plt.ylabel('mmHg')
    plt.title(str(patient_num) + "- systolic and diastolic BP train on self")
    plt.legend()
    plt.savefig(figures_path + '/' + graph_name)


def Train_and_Test(path_bp, path_ppg, patient_num, train_start_time):
    enableBatch = True
    enableTrain = False  # TODO: train or test?

    # calculate train and test size
    train_start_samp = train_start_time * 60 * 125
    train_size_125Hz = 2 ** 16  # 5*60*125#
    test_size_125Hz = 2 ** 16  # 5*60*125 #

    # load BP and PPG
    row_count_bp, data_bp = read_csv(path_bp)
    row_count_ppg, data_ppg = read_csv(path_ppg)

    # cut the signals to get them the same length
    row_count = min(row_count_ppg, row_count_bp)
    data_bp = data_bp[:row_count]
    data_ppg = data_ppg[:row_count]

    # SKlearn min max scaler (normalization)
    scaler_BP = MinMaxScaler(feature_range=(-1, 1))
    data_bp_scaled = scaler_BP.fit_transform(data_bp)
    scaler_PPG = MinMaxScaler(feature_range=(-1, 1))
    data_ppg_scaled = scaler_PPG.fit_transform(data_ppg)

    # divide the data to train and test
    train_bp_scaled = data_bp_scaled[train_start_samp:train_start_samp + train_size_125Hz, :]
    train_ppg_scaled = data_ppg_scaled[train_start_samp:train_start_samp + train_size_125Hz, :]
    test_bp = data_bp[train_start_samp + train_size_125Hz:train_start_samp + train_size_125Hz + test_size_125Hz, :]
    test_ppg_scaled = data_ppg_scaled[train_start_samp + train_size_125Hz:train_start_samp + train_size_125Hz + test_size_125Hz, :]

    # Creating lstm class
    class CustomLSTM(nn.Module):
        # Ctor
        def __init__(self, hidden_size, input_size, output_size):
            super(CustomLSTM, self).__init__()
            self.hidden_dim = hidden_size
            self.output_size = output_size
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
            self.act = nn.Tanh()
            self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)
            self.linear01 = nn.Linear(in_features=input_size, out_features=input_size)

        # Forward function
        def forward(self, x):
            seqLength = x.size(0)
            batchSize = x.size(1)
            y = torch.zeros(x.size())
            pred, (hidden, context) = self.lstm(x)
            out = torch.zeros(seqLength, batchSize, self.output_size).cuda()
            for s in range(seqLength):
                out[s] = self.linear(pred[s])
            return out


    # Creating the lstm nn
    r = CustomLSTM(hidden_size, input_dim, output_dim).to(device)

    predictions = []
    optimizer = torch.optim.Adam(r.parameters(), lr=1e-3)
    loss_func = nn.MSELoss().to(device)
    tau = 0  # future estimation time
    loss_vec = []
    running_loss = 0.0

    # TRAIN
    if enableBatch:
        inp_ppg = (torch.tensor(train_ppg_scaled)).reshape(-1, 2 ** 11).transpose(1, 0).unsqueeze_(-1)
        out = (torch.tensor(train_bp_scaled)).reshape(-1, 2 ** 11).transpose(1, 0).unsqueeze_(-1)
        inp_ppg = inp_ppg.float().cuda()
        out = out.float().cuda()
    else:
        inp_ppg = torch.Tensor(train_ppg_scaled.reshape((train_ppg_scaled.shape[0], -1, 1))).cuda()
        out = torch.Tensor(train_bp_scaled.reshape((train_bp_scaled.shape[0], -1, 1))).cuda()

    if enableTrain:
        for t in range(2500):
            startTime = time.time()
            hidden = None
            optimizer.zero_grad()
            pred = r(inp_ppg)
            predictions.append(pred.data.cpu().numpy())
            loss = loss_func(pred, out)
            loss.backward()
            optimizer.step()
            endTime = time.time()
            print('single iter time: %f ms' % (1000 * (endTime - startTime)))
            # print statistics
            running_loss = 0.0
            running_loss += loss.item()
            loss_vec.append(running_loss)
            print(t, running_loss)
            if t % 100 == 0:
                plt.clf()
                if True:  # enableBatch:
                    plt.plot(pred[:1000, 0, 0].data.cpu().numpy(), label='prediction_BP')
                    plt.plot(out[:1000, 0, 0].data.cpu().numpy(), label='train_BP')
                    plt.plot(inp_ppg[:1000, 0, 0].data.cpu().numpy(), label='train_PPG')
                else:
                    plt.plot(pred[:(5 * 60 * 125), 0].data.cpu().numpy(), label='prediction_BP')
                    plt.plot(out[:(5 * 60 * 125), 0].data.cpu().numpy(), label='train_BP')
                plt.title('iteration ' + str(t))
                plt.legend()
                plt.savefig(
                    figures_path + '/iteration_' + str(t) + " from minute " + str(train_start_time))

        # Save the net
        torch.save(r.state_dict(), r_path)
        # Mean Loss
        plt.clf()
        plt.plot(loss_vec)
        plt.title("Mean loss")
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig(figures_path + '/Mean_loss' + " from minute " + str(train_start_time))
    else:
        r.load_state_dict(torch.load(r_path))  # Loading the net

    # TEST- on the same patient as the train
    x_test_PPG = torch.Tensor(test_ppg_scaled.reshape((test_ppg_scaled.shape[0], -1, 1))).cuda()
    pred_t = r(x_test_PPG)

    # Test loss
    runningLossTest = 0.0
    lossTest = loss_func(pred_t, (torch.Tensor(test_bp.reshape((test_bp.shape[0], -1, 1)))).cuda())
    runningLossTest += lossTest.item()

    # now we have pred_t as our predicted BP, but it is scaled.
    pred_t = scaler_BP.inverse_transform(pred_t[:, :, 0].data.cpu().numpy())

    # convert to BP unit
    pred_t = (pred_t * 0.0625) - 40
    test_bp = (test_bp * 0.0625) - 40

    start_time = (train_start_samp + train_size_125Hz) / (60 * 125)
    end_time = (train_start_samp + train_size_125Hz + test_size_125Hz) / (60 * 125)
    start_time = (train_start_samp) / (60 * 125)
    end_time = (train_start_samp + test_size_125Hz) / (60 * 125)

    X_axis = np.linspace(start_time, end_time, (end_time - start_time) * 60 * 125)

    # plot ZOOM
    plt.clf()
    plt.plot(X_axis[:1000], pred_t[:1000, 0], label='prediction_BP')
    plt.plot(X_axis[:1000], test_bp[:1000, 0], label='expected_BP')
    plt.xlabel('time [minute]')
    plt.ylabel('mmHg')
    plt.title("Zoom: BP Prediction based on PPG - train on self")
    plt.legend()
    plt.savefig(figures_path + '/ZOOM: ' + str(patient_num) + ' Prediction and Expected BP train on self- 0-9 min')

    # plot FINEL
    plt.clf()
    plt.plot(X_axis, pred_t[:, 0], label='prediction_BP')
    plt.plot(X_axis, test_bp[:, 0], label='expected_BP')
    plt.xlabel('time [minute]')
    plt.ylabel('mmHg')
    plt.title(" BP Prediction based on PPG - train on self")
    plt.legend()
    plt.savefig(figures_path + '/FINAL: ' + str(patient_num) + ' Prediction and Expected BP train on self-0-9 min')
    plt.clf()

    calc_loss_precentage(test_bp[:, 0], pred_t[:, 0], test_size_125Hz, patient_num, train_start_time, "test on self")

    # TEST ON A DIFFERENT PATIENT
    row_count_bp_test1, data_bp_test1 = read_csv('/home/shirili/Downloads/ShirirliDorin/project_A/data_125Hz/2642420-8079/test_on_others/1887524-1978-MDC_PRESS_BLD_ART_ABP-125.csv') #TODO insert path
    row_count_ppg_test1, data_ppg_test1 = read_csv('/home/shirili/Downloads/ShirirliDorin/project_A/data_125Hz/2642420-8079/test_on_others/1887524-1978-MDC_PULS_OXIM_PLETH-125.csv') #TODO insert path
    row_count_test1 = min(row_count_bp_test1, row_count_ppg_test1)
    data_bp_test1 = data_bp_test1[:row_count_test1]
    data_ppg_test1 = data_ppg_test1[:row_count_test1]
    test1_bp_segment = data_bp_test1[train_start_samp + train_size_125Hz:train_start_samp + train_size_125Hz + test_size_125Hz, :]
    test1_ppg_segment = data_ppg_test1[train_start_samp + train_size_125Hz:train_start_samp + train_size_125Hz + test_size_125Hz, :]
    scaler_PPG_test1 = MinMaxScaler(feature_range=(-1, 1))
    data_ppg_test_1_scaled = scaler_PPG_test1.fit_transform(test1_ppg_segment)
    t_inp_ppg_test_1 = torch.Tensor(data_ppg_test_1_scaled.reshape((data_ppg_test_1_scaled.shape[0], -1, 1))).cuda()
    pred_BP_test_1 = r(t_inp_ppg_test_1)
    pred_BP_test_1 = scaler_BP.inverse_transform(pred_BP_test_1[:, :, 0].data.cpu().numpy())
    pred_BP_test_1 = (pred_BP_test_1 * 0.0625) - 40
    test1_bp_segment = (test1_bp_segment * 0.0625) - 40
    calc_loss_precentage(test1_bp_segment[:, 0], pred_BP_test_1[:, 0], test_size_125Hz, patient_num, train_start_time, "test_on_1887524-1978_0-9min") #TODO write headline

Train_and_Test(
    '/home/shirili/Downloads/ShirirliDorin/project_A/data_125Hz/2642420-8079/2642420-8079-MDC_PRESS_BLD_ART_ABP-125.csv',
    '/home/shirili/Downloads/ShirirliDorin/project_A/data_125Hz/2642420-8079/2642420-8079-MDC_PULS_OXIM_PLETH-125.csv',
    '2642420-8079',0)




















