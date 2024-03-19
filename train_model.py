"""File that contain the whole solution of the task.

This include model architecture, dataset, some helper function
and train script.
"""


from pathlib import Path
from typing import List, Tuple, Union, Dict, Any
from functools import partial
import shutil
import json
import math

import pandas as pd
from tqdm import tqdm
import torch
from torch import FloatTensor, IntTensor, Tensor
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, Metric


class LetterAccentuationDataset(Dataset):
    """Dataset class for the word accentuation task."""

    def __init__(self, csv_pth: Path, train: bool = True) -> None:
        """Initialize dataset.

        Parameters
        ----------
        csv_pth : Path
            A path to a dataset csv file.
        train : bool, optional
            Is the dataset for training or testing. Train dataset has target
            data whereas test dataset has not. By default is `True`.
        """
        super().__init__()
        self.let_to_idx = {'pad': 0}
        for i, let in enumerate(range(ord('а'), ord('я') + 1)):
            self.let_to_idx[chr(let)] = i + 1
        self.let_to_idx['ё'] = len(self.let_to_idx)
        self.idx_to_let = {val: key for key, val in self.let_to_idx.items()}
        self.vowels = set(['у', 'е', 'ы', 'а', 'о', 'э', 'я', 'и', 'ю', 'ё'])

        df = pd.read_csv(csv_pth)

        self.data = []
        self.word_max_len = 0
        for _, row in df.iterrows():
            word = row['word']
            lemma = row['lemma']

            sample = {
                'word': torch.tensor(self.word_to_idxs(word)),
                'lemma': torch.tensor(self.word_to_idxs(lemma))
            }

            if train:
                stress = row['stress']
                stress = self.syllable_to_letter(word, stress)
                sample['stress'] = torch.tensor(stress)

            self.data.append(sample)

            self.word_max_len = max(self.word_max_len,
                                    max(len(word), len(lemma)))

    def __getitem__(self, index: int) -> Tuple[List[int], List[int], int]:
        """Get a sample from the dataset.

        Parameters
        ----------
        index : int
            An index of the sample.

        Returns
        -------
        Tuple[List[int], List[int], int]
            Return a tuple consists of word and lemma in List[int]
            representation. If the dataset is for train,
            the index of target stress letter is added.
        """
        sample = self.data[index]
        return tuple(sample.values())
    
    def __len__(self) -> int:
        """Get length of the dataset.

        Returns
        -------
        int
            The length of the dataset
        """
        return len(self.data)
    
    def word_to_idxs(self, word: str) -> List[int]:
        """Convert a str word to a List[int] letter indexes.

        Parameters
        ----------
        word : str
            The source word string.

        Returns
        -------
        List[int]
            The converted word in List[int] format.
        """
        return [self.let_to_idx[let] for let in word]
    
    def idxs_to_word(self, idxs: Union[List[int], Tensor]) -> str:
        """Convert a List[int] letter indexes to an origin str word.

        Parameters
        ----------
        idxs : Union[List[int], Tensor]
            The list of the letter indexes.

        Returns
        -------
        str
            The origin word string.
        """
        if isinstance(idxs, Tensor):
            idxs = map(int, idxs.tolist())
        return ''.join([self.idx_to_let[idx] for idx in idxs if idx != 0])
    
    def syllable_to_letter(self, word: str, stress: int) -> int:
        """Convert syllable index to letter index.

        Parameters
        ----------
        word : str
            The word string.
        stress : int
            The index of accentuated syllable.

        Returns
        -------
        int
            The index of accentuated letter.
        """
        for i, let in enumerate(word):
            if let in self.vowels:
                stress -= 1
            if stress == 0:
                return i
        else:
            raise

    def letter_to_syllable(self, word: str, stress: int) -> int:
        """Convert letter index to syllable index.

        Parameters
        ----------
        word : str
            The word string.
        stress : int
            The index of accentuated letter.

        Returns
        -------
        int
            The index of accentuated syllable.
        """
        if stress >= len(word):  # if predict not existed letter
            stress = len(word) - 1
        syllable = 0
        for i in range(stress + 1):
            if word[i] in self.vowels:
                syllable += 1
        return syllable


def dset_collate_func(
    batch: Tuple[Tuple[IntTensor, IntTensor, IntTensor]],
    seq_len: int,
    train: bool = True
) -> Tuple[IntTensor, IntTensor, IntTensor]:
    """A collate function for `LetterAccentuationDataset`.

    Parameters
    ----------
    batch : Tuple[Tuple[IntTensor, IntTensor, IntTensor]]
        Not batched data.
    seq_len : int
        A length of the longest expected word in dataset.
    train : bool, optional
        Is the dataset for training or testing. Train dataset has target
        data whereas test dataset has not. By default is `True`.

    Returns
    -------
    Tuple[IntTensor, IntTensor, IntTensor]
        Batched data.
    """
    batch = tuple(zip(*batch))  # words, lemmas, stresses
    pad_words = []
    pad_lemmas = []
    for sample in zip(*batch):
        word = sample[0]
        lemma = sample[1]
        pad_words.append(torch.zeros(seq_len, dtype=torch.int32))
        pad_lemmas.append(torch.zeros(seq_len, dtype=torch.int32))
        pad_words[-1][:word.shape[0]] = word
        pad_lemmas[-1][:lemma.shape[0]] = lemma
    result = [torch.stack(pad_words), torch.stack(pad_lemmas)]
    if train:  # If there are targets
        result.append(torch.stack(batch[2]))
    return tuple(result)


class AccentuationModel(nn.Module):
    """A model class for the word accentuation task."""

    def __init__(
        self, num_lets: int, seq_len: int, embed_dim: int, num_heads: int,
        dropout: float, hidden_dims: List[int]
    ) -> None:
        """Initialize the model.

        Parameters
        ----------
        num_lets : int
            A number of letters in the using alphabet.
        seq_len : int
            A sequence length or the length of the longest word in dataset.
        embed_dim : int
            A size of letter embeddings.
        num_heads : int
            A number of heads in attention layer.
        dropout : float
            A dropout probability.
        hidden_dims : List[int]
            A list with hidden linear layers dimensions.
        """
        super().__init__()
        self._config = {
            'num_lets': num_lets,
            'seq_len': seq_len,
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'dropout': dropout,
            'hidden_dims': hidden_dims}

        self.embed_layer = nn.Embedding(
            num_embeddings=num_lets + 1, embedding_dim=embed_dim,
            padding_idx=0)
        self.position_layer = PositionalEncoding(
            embed_dim, dropout, seq_len, batch_first=True)
        
        # Create linear layers
        self.word_layers = nn.ModuleList()
        self.lemma_layers = nn.ModuleList()
        in_features = embed_dim * seq_len
        for out_features in hidden_dims:
            self.word_layers.append(
                nn.Linear(in_features, out_features, bias=False))
            self.word_layers.append(nn.Dropout(p=dropout, inplace=True))
            self.word_layers.append(nn.BatchNorm1d(out_features))

            self.lemma_layers.append(
                nn.Linear(in_features, out_features, bias=False))
            self.lemma_layers.append(nn.Dropout(p=dropout, inplace=True))
            self.lemma_layers.append(nn.BatchNorm1d(out_features))

            in_features = out_features

        # Final layers
        attn_features = out_features // seq_len
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=attn_features, num_heads=num_heads, dropout=dropout,
            batch_first=True)
        self.final_linear = nn.Linear(attn_features, 1)
        
    def forward(self, words: IntTensor, lemmas: IntTensor) -> FloatTensor:
        """Forward pass.

        Parameters
        ----------
        words : IntTensor
            A converted words tensor with shape `(b, seq_len)`.
        lemmas : IntTensor
            A converted lemmas tensor with shape `(b, seq_len)`.

        Returns
        -------
        FloatTensor
            Result logits tensor with shape `(b, seq_len)`.
        """
        b_dim = words.shape[0]
        # (b, seq_len, embed_dim)
        x = self.embed_layer(words)
        y = self.embed_layer(lemmas)
        x = self.position_layer(x)
        y = self.position_layer(y)
        # (b, seq_len * embed_dim)
        x = x.reshape(words.shape[0], -1)
        y = y.reshape(lemmas.shape[0], -1)

        for word_layer, lem_layer in zip(self.word_layers, self.lemma_layers):
            x = word_layer(x)
            y = lem_layer(y)

        # (b, seq_len, hidden_dims[-1])
        x = x.reshape(b_dim, self._config['seq_len'], -1)
        y = y.reshape(b_dim, self._config['seq_len'], -1)
        out, _ = self.attention_layer(query=x, key=y, value=x,
                                      need_weights=False)
        out = self.final_linear(out)
        out = out.squeeze(-1)
        return out
    
    def get_config(self):
        """Get config from which model can be created."""
        return self._config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AccentuationModel':
        """Create model instance from its config."""
        return cls(**config)
    

class PositionalEncoding(nn.Module):
    """A positional encoding layer."""

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        batch_first: bool = False
    ):
        """Initialize the layer.

        Parameters
        ----------
        d_model : int
            A size of embedding dimension.
        dropout : float, optional
            A dropout probability.
        max_len : int, optional
            A length of the longest expected sequence.
        batch_first : bool, optional
            Whether the batch dimension is first. By default is `False`.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = (torch.exp(torch.arange(0, d_model, 2) *
                              (-math.log(10000.0) / d_model)))
        if batch_first:
            pe = torch.zeros(1, max_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            An input tensor with shape `(seq_len, b, embed_dim)`
            if batch_first equals `False` and `(b, seq_len, embed_dim)`
            if batch_first is `True`.

        Returns
        -------
        Tensor
            An output tensor with the same shape as the input.
        """
        x = x + self.pe
        return self.dropout(x)


class LossMetric(Metric):
    """TorchMetric for loss collecting."""

    def __init__(self):
        """Initialize the metric."""
        super().__init__()
        self.add_state('loss',
                       default=torch.tensor(0, dtype=torch.float32),
                       dist_reduce_fx='sum')
        self.add_state('n_total',
                       default=torch.tensor(0, dtype=torch.float32),
                       dist_reduce_fx='sum')

    def update(self, batch_loss: FloatTensor):
        self.loss += batch_loss
        self.n_total += 1

    def compute(self):
        return self.loss / self.n_total


def main(config_pth: Path):
    # Read config
    with open(config_pth, 'r') as f:
        config_str = f.read()
    config = json.loads(config_str)

    # Environment parameters
    torch.random.manual_seed(config['rnd_seed'])
    device = torch.device(config['device'])
    work_dir_name = 'train_' + config_pth.name.split('_')[-1].split('.')[0]
    work_dir = Path(config['work_dir']) / work_dir_name
    tensorboard_dir = work_dir / 'tensorboard'
    ckpt_dir = work_dir / 'ckpts'

    # Dataset parameters
    train_csv = Path('data/stress_predictor/train.csv')
    b_size = config['b_size']
    shuffle_train = config['shuffle_train']
    shuffle_val = config['shuffle_val']
    train_val_proportions = config['train_val_proportions']
    num_lets = config['num_lets']

    # Model parameters
    embed_dim = config['embed_dim']
    num_heads = config['num_heads']
    drop_out = config['drop_out']
    hidden_dims = config['hidden_dims']

    # Training parameters
    lr = config['lr']
    epochs = config['epochs']
    weight_decay = config['weight_decay']

    # Prepare some stuff
    if work_dir.exists():
        input('Specified directory already exists. '
              'Сontinuing to work will delete the data located there. '
              'Press enter to continue.')
        shutil.rmtree(work_dir)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Get tensorboard
    log_writer = SummaryWriter(str(tensorboard_dir))

    # Get the dataset and loaders
    dset = LetterAccentuationDataset(train_csv)
    train_dset, val_dset = random_split(dset, train_val_proportions)

    collate_fn = partial(dset_collate_func, seq_len=dset.word_max_len)
    train_loader = DataLoader(
        train_dset, batch_size=b_size, shuffle=shuffle_train,
        collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(
        val_dset, batch_size=b_size, shuffle=shuffle_val,
        collate_fn=collate_fn, drop_last=True)

    # Get the model
    model = AccentuationModel(num_lets=num_lets, seq_len=dset.word_max_len,
                              embed_dim=embed_dim, num_heads=num_heads,
                              dropout=drop_out, hidden_dims=hidden_dims)
    with open(work_dir / 'model_config.json', 'w') as f:
        json.dump(model.get_config(), f)  # Save model config separately
    model.to(device=device)
    
    # Get the optimizer and the loss function
    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = nn.CrossEntropyLoss()

    # Get the metrics
    train_loss_metric = LossMetric()
    train_loss_metric.to(device=device)
    val_loss_metric = LossMetric()
    val_loss_metric.to(device=device)
    train_accuracy_metric = Accuracy(
        task="multiclass", num_classes=dset.word_max_len)
    train_accuracy_metric.to(device=device)
    val_accuracy_metric = Accuracy(
        task="multiclass", num_classes=dset.word_max_len)
    val_accuracy_metric.to(device=device)

    # Get the lr scheduler
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    # Training
    best_metric = None
    for e in range(epochs):
        print(f'Epoch {e + 1}')
        
        # Train
        model.train()
        for batch in tqdm(train_loader, 'Train step'):
            words, lemmas, stresses = batch
            words = words.to(device=device)
            lemmas = lemmas.to(device=device)
            stresses = stresses.to(device=device)

            preds = model(words, lemmas)
            loss = loss_func(preds, stresses)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            _, pred_classes = torch.max(preds.data, 1)
            train_accuracy_metric.update(pred_classes, stresses)
            train_loss_metric.update(loss)

        # Validation
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, 'Val step'):
                words, lemmas, stresses = batch
                words = words.to(device=device)
                lemmas = lemmas.to(device=device)
                stresses = stresses.to(device=device)

                preds = model(words, lemmas)
                loss = loss_func(preds, stresses)

                _, pred_classes = torch.max(preds.data, 1)
                val_accuracy_metric.update(pred_classes, stresses)
                val_loss_metric.update(loss)

        # Lr scheduler step
        lr_scheduler.step()

        # Log epoch metrics
        train_loss = train_loss_metric.compute()
        train_loss_metric.reset()
        val_loss = val_loss_metric.compute()
        val_loss_metric.reset()
        log_writer.add_scalars('loss', {
            'train': train_loss,
            'val': val_loss
        }, e)

        train_acc = train_accuracy_metric.compute()
        train_accuracy_metric.reset()
        val_acc = val_accuracy_metric.compute()
        val_accuracy_metric.reset()
        log_writer.add_scalars('accuracy', {
            'train': train_acc,
            'val': val_acc
        }, e)

        print('TrainLoss:', train_loss.item())
        print('ValLoss:', val_loss.item())
        print('TrainAcc:', train_acc.item())
        print('ValAcc:', val_acc.item())

        # Save model
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'epoch': e + 1
        }
        torch.save(checkpoint, ckpt_dir / 'last_checkpoint.pth')

        if best_metric is None or best_metric < val_acc.item():
            torch.save(checkpoint, ckpt_dir / 'best_checkpoint.pth')
            best_metric = val_acc.item()

    # log_writer.add_scalar('best_accuracy', best_metric)
    log_writer.add_text('best_accuracy',
                        f'Best accuracy on validation set: {best_metric:.3f}')
    log_writer.close()


if __name__ == '__main__':
    config_pth = Path('stress_predictor/configs/config_test.json')
    main(config_pth)
