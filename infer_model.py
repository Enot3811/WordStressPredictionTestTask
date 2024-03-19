"""Script that perform inference of the trained model."""


from pathlib import Path
from functools import partial
import json

import torch
from torch.utils.data import DataLoader

from train_model import (
    AccentuationModel, LetterAccentuationDataset, dset_collate_func)


def main():
    # Read config
    config_pth = Path('stress_predictor/configs/config_12.json')
    with open(config_pth, 'r') as f:
        config_str = f.read()
    config = json.loads(config_str)
    torch.random.manual_seed(config['rnd_seed'])

    work_dir_name = 'train_' + config_pth.name.split('_')[-1].split('.')[0]
    work_dir = Path(config['work_dir']) / work_dir_name
    model_cfg = work_dir / 'model_config.json'
    with open(model_cfg, 'r') as f:
        model_cfg = f.read()
    model_cfg = json.loads(model_cfg)

    # Get the model
    model = AccentuationModel.from_config(model_cfg)
    ckpt_pth = work_dir / 'ckpts' / 'best_checkpoint.pth'
    state_dict = torch.load(ckpt_pth)['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    # Get the data
    train = False
    file = 'train.csv' if train else 'test.csv'
    test_csv = Path('data/stress_predictor/') / file
    dset = LetterAccentuationDataset(test_csv, train=train)
    collate_fn = partial(
        dset_collate_func, seq_len=model_cfg['seq_len'], train=train)
    d_loader = DataLoader(dset, collate_fn=collate_fn, batch_size=64)

    # Iterate
    for sample in d_loader:
        if train:
            word, lemma, stress = sample
        else:
            word, lemma = sample

        preds = model(word, lemma)
        _, pred_classes = torch.max(preds.data, 1)

        for i, w in enumerate(word.tolist()):
            w = list(dset.idxs_to_word(w))
            pred_idx = pred_classes[i].item()
            w[pred_idx] = w[pred_idx].upper()
            w = ''.join(w)
            if train:
                print(w, pred_idx, stress[i].item())
            else:
                print(w, pred_idx)


if __name__ == '__main__':
    
    main()
