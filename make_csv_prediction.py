"""Script that perform predict and write the results into csv file."""


from pathlib import Path
from functools import partial
import json

from tqdm import tqdm
import pandas as pd
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
    d_loader = DataLoader(dset, collate_fn=collate_fn, batch_size=1)
    result_df = pd.DataFrame(columns=['id', 'stress'])
    csv_pth = work_dir / 'predict.csv'

    # Iterate
    for j, sample in tqdm(enumerate(d_loader)):
        if train:
            words, lemmas, stresses = sample
        else:
            words, lemmas = sample

        preds = model(words, lemmas)
        _, pred_classes = torch.max(preds.data, 1)

        source_word = dset.idxs_to_word(words[0])
        pred_idx = pred_classes[0].item()
        syllable_idx = dset.letter_to_syllable(source_word, pred_idx)

        result_df.loc[j] = [j, syllable_idx]
    
    result_df.to_csv(csv_pth, sep=',', index=False)


if __name__ == '__main__':
    main()
