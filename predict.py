from models.CNNGRU import CNNGRU
import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def normalize(tensor):
    # 최솟값과 최댓값을 이용하여 정규화
    scaled_tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    return scaled_tensor, tensor.min(), tensor.max()


def unnormalize(scaled_tensor, original_min, original_max):
    # 원래 범위로 되돌리는 역변환 수행
    reversed_tensor = scaled_tensor * (original_max - original_min) + original_min

    return reversed_tensor


def data_processing(args):
    data_csv = pd.read_csv(args.data_path)
    full_data = data_csv[args.target].values
    full_data = torch.FloatTensor(full_data)
    proceeding_step = len(full_data[args.given_len+1:])
    real_value = full_data
    input_data = real_value[:args.given_len]
    answer = input_data[-1:]
    return input_data, answer, real_value, proceeding_step


def predict_plot(predicts, real_values):
    plt.figure(figsize=(12, 8))
    plt.plot(predicts, color='red', label='Predict')
    plt.plot(real_values, color='blue', label='Real')
    plt.legend(loc='center left')
    plt.savefig('sample_plot.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--length', type=int, default=1)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--target', type=str, default='close')
    parser.add_argument('--given_len', type=int, default=1440)
    parser.add_argument('--predict_continue', type=bool, default=True)
    parser.add_argument('--teacher_forcing', type=bool, default=True)

    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt_path)
    model_weights = checkpoint["state_dict"]
    hyper_parameters = checkpoint["hyper_parameters"]["model_config"]

    rename_model_weights = model_weights.copy()
    for weight_name in model_weights.keys():
        rename_model_weights[weight_name[6:]] = model_weights[weight_name]
        del rename_model_weights[weight_name]

    model = CNNGRU(hyper_parameters)
    model.load_state_dict(rename_model_weights)
    model.eval()


    if args.predict_continue:
        if not args.teacher_forcing:
            x, answer, real_series, proceeding_step = data_processing(args)
            for step in tqdm(range(proceeding_step)):
                input, min_value, max_value = normalize(x)
                input = input.unsqueeze(-1)
                with torch.no_grad():
                    y_hat = model(input)
                output = torch.cat([input, y_hat.unsqueeze(-1)], dim=0)
                output = unnormalize(output, min_value, max_value)
                input = x[1:]
                x = torch.cat([x, output[-1]], dim=0)
            predict_plot(x, real_series)
        else:
            x, answer, real_series, proceeding_step = data_processing(args)
            for step in tqdm(range(proceeding_step)):
                input, min_value, max_value = normalize(real_series[step:step+args.given_len])
                input = input.unsqueeze(-1)
                with torch.no_grad():
                    y_hat = model(input)
                output = torch.cat([input, y_hat.unsqueeze(-1)], dim=0)
                output = unnormalize(output, min_value, max_value)
                x = torch.cat([x, output[-1]], dim=0)
            predict_plot(x, real_series)


    else:
        x, answer, real_series = data_processing(args)
        with torch.no_grad():
            y_hat = model(x)
        predict_series = torch.cat([x, y_hat.unsqueeze(-1)], dim=0)
        predict_series = unnormalize(predict_series, min_val, max_val)

        predict_plot(predict_series, real_series)

        print(y_hat, answer)

