from models.CNNGRU import CNNGRU
import torch
import argparse
import pandas as pd

def data_processing(args):
    data_csv = pd.read_csv(args.data_path)
    full_data = data_csv[args.target].values
    full_data = torch.tensor(full_data)
    input_data = full_data[:args.given_len]
    answer = full_data[args.given_len:args.given_len+1]
    return input_data, answer



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--length', type=int, default=1)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--target', type=str, default='close')
    parser.add_argument('--given_len', type=int, default=1440)

    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt_path)
    model_weights = checkpoint["state_dict"]
    hyper_parameters = checkpoint["model"]["init_args"]["model_config"]

    model = CNNGRU(hyper_parameters)
    model.load_state_dict(model_weights)
    model.eval()

    x = data_processing(args)

    with torch.no_grad():
        y_hat = model(x)