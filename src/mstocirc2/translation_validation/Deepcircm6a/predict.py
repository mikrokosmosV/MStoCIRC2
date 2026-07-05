from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model_one_hot_NCP_EIIP import CNN51_RNN
from seq_load_one_hot_NCP_EIIP import load_data_bicoding_with_header


def _load_checkpoint(checkpoint_path: Path):
    load_kwargs = {"map_location": torch.device("cpu")}
    try:
        return torch.load(checkpoint_path, weights_only=False, **load_kwargs)
    except TypeError:
        return torch.load(checkpoint_path, **load_kwargs)


def _build_model(model_path: Path) -> CNN51_RNN:
    hidden_num = 128
    layer_num = 3
    fc_dropout = 0.5
    rnn_dropout = 0.5
    checkpoint = _load_checkpoint(model_path / "checkpoint.pth.tar")
    model = CNN51_RNN(hidden_num, layer_num, fc_dropout, rnn_dropout, "LSTM")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _predict(model: CNN51_RNN, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        fx = model.forward(x)
        prob_data = F.log_softmax(fx, dim=1).cpu().data.numpy()
    return np.exp(prob_data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-predict_fa", "--predict_fasta", dest="predict_fa", required=True)
    parser.add_argument("-model_path", "--model_path", dest="model_path", required=True)
    parser.add_argument("-outfile", "--outfile", dest="outfile", required=True)
    args = parser.parse_args()

    model = _build_model(Path(args.model_path))
    x_test, fa_header = load_data_bicoding_with_header(args.predict_fa)
    x_array = np.array(x_test).reshape(len(x_test), 51, 8)
    x_tensor = torch.from_numpy(x_array).float()

    batch_size = 256
    i = 0
    total = x_tensor.shape[0]
    with open(args.outfile, "w", encoding="utf-8") as fw:
        while i < total:
            x_batch = x_tensor[i : i + batch_size]
            header_batch = fa_header[i : i + batch_size]
            prob_data = _predict(model, x_batch)
            for idx, probs in enumerate(prob_data):
                fw.write(f"{header_batch[idx]}\t{probs[1]}\n")
            i += batch_size
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
