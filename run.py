__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"

import numpy as np
import pdb
import os
import json
import time
import math
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, abs_data, prediction_len, prediction_interval, sequence_length):
    fig = plt.figure(facecolor='white')
    #ax = fig.add_subplot(121)
    #ax.plot(true_data, label='True Data')
    ax2 = fig.add_subplot(111)
    ax2.plot(abs_data[:, 0])

    #new_data = []
    #for i in range(len(true_data)):
    #    new_data.append((1+true_data[i]) * abs_data[i][0])
    #ax2.plot(new_data)

    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        shift_width = i * prediction_interval
        #for i in range(len(data)):
        #ax.plot([None] * shift_width + data, label='Prediction')
        predicted_abs = abs_data[shift_width:shift_width+prediction_len, 0] * (1 + np.array(data))
        coef = np.vstack([np.arange(0,prediction_len), np.ones(prediction_len)]).T
        m, c = np.linalg.lstsq(coef, predicted_abs, rcond=None)[0]
        line_values = m * np.arange(0,prediction_len) + c

        ax2.plot([None] * (shift_width + sequence_length) + list(line_values)[:10])
        #ax2.plot([None] * (shift_width + sequence_length) + list(abs_data[shift_width, 0] * (1 + np.array(data))))
        #plt.legend()
    plt.show()


def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    model = Model()
    model.build_model(configs)
    #model.load_model("saved_models/AAPL-b64-e12.h5")
    #model.load_model("saved_models/25042021-203944-e20.h5")
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    '''
        # in-memory training
        model.train(
        	x,
        	y,
        	epochs = configs['training']['epochs'],
        	batch_size = configs['training']['batch_size'],
        	save_dir = configs['model']['save_dir']
        )
        '''
    # out-of memory generative training
    steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    model.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=configs['data']['sequence_length'],
            batch_size=configs['training']['batch_size'],
            normalise=configs['data']['normalise']
        ),
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
        save_dir=configs['model']['save_dir']
    )

    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['prediction']['length'], configs['prediction']['interval'])
    # predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    # predictions = model.predict_point_by_point(x_test)

    plot_results_multiple(predictions, y_test, data.data_test, configs['prediction']['length'], configs['prediction']['interval'], configs['data']['sequence_length'])
    # plot_results(predictions, y_test)


if __name__ == '__main__':
    main()
