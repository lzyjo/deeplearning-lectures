#!/usr/bin/env python3
# coding: utf-8

# Standard imports
import os
import functools
import logging
import operator
from pathlib import Path
from typing import Union, Tuple
import pickle
import time


# External imports
import torch.nn as nn
from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence,
    pad_packed_sequence,
    PackedSequence,
)
import torch.utils.data
import torchaudio
from torchaudio.datasets import COMMONVOICE
from torchaudio.transforms import (
    Spectrogram,
    AmplitudeToDB,
    MelSpectrogram,
    FrequencyMasking,
    TimeMasking,
)
import matplotlib.pyplot as plt
import multiprocess


_DEFAULT_COMMONVOICE_ROOT = "/mounts/Datasets4/CommonVoice/"
_DEFAULT_COMMONVOICE_VERSION = "v15.0"
_DEFAULT_RATE = 16000  # Hz
_DEFAULT_WIN_LENGTH = 40  # ms
_DEFAULT_WIN_STEP = 10  # ms
_DEFAULT_NUM_MELS = 20  #


def unpack_ravel(tensor: PackedSequence):
    unpacked_tensor, lens_tensor = pad_packed_sequence(tensor)  # T, B, *
    raveled = torch.cat(
        [tensori[:leni] for tensori, leni in zip(unpacked_tensor, lens_tensor)], 0
    )
    # raveled is (Tcum, num_features)
    return raveled


def load_dataset(
    fold: str,
    commonvoice_root: Union[str, Path],
    commonvoice_version: str,
    lang: str = "fr",
) -> torch.utils.data.Dataset:
    """
    Load the commonvoice dataset within the path
    commonvoice_root/commonvoice_version/lang

    In this folder, we expect to find the tsv files of CommonVoice

    Args:
        fold (str): the fold to load, e.g. train, dev, test, validated, ..
        commonvoice_root

    Returns:
        torch.utils.data.Dataset: ``dataset``
    """
    datasetpath = os.path.join(commonvoice_root, commonvoice_version, lang)
    return COMMONVOICE(root=datasetpath, tsv=fold + ".tsv")


def is_sample_in_timerange(i, ds, min_duration, max_duration):
    (w, r, _) = ds[i]
    return i, (not min_duration or min_duration <= (w.squeeze().shape[0] / r)) and (
        not max_duration or (w.squeeze().shape[0] / r) <= max_duration
    )


class DatasetFilter(object):
    """
    Dataset object filtering an original dataset based on the
    durations of its waveform
    """

    def __init__(
        self,
        ds: torch.utils.data.Dataset,
        min_duration: float,
        max_duration: float,
        cacheprefix: str,
        overwrite_index: bool,
        logger=None,
    ) -> None:
        """
        Args:
            ds: the dataset to filter
            min_duration : the minimal duration in seconds
            max_duration : the maximal duration in seconds
            cacheprefix : the prefix of the cache file to which save the index files
        """

        def log(txt):
            if logger is not None:
                logger.info(txt)

        # At construction we build a list of indices
        # of valid samples from the original dataset
        cachepath = cacheprefix + f"-{min_duration}-{max_duration}.idx"
        if os.path.exists(cachepath) and not overwrite_index:
            log("Loading the pre-generated index file {cachepath}")
            self.valid_indices = pickle.load(open(cachepath, "rb"))
        else:
            log(
                f"Generating the index files, processing {len(ds)} files, saved in {cachepath}"
            )
            # self.valid_indices = [
            #     i
            #     for i, (w, r, _) in tqdm.tqdm(enumerate(ds))
            #     if (not min_duration or min_duration <= (w.squeeze().shape[0] / r))
            #     and (not max_duration or (w.squeeze().shape[0] / r) <= max_duration)
            # ]

            indices = list(range(len(ds)))
            t0 = time.time()
            with multiprocess.Pool(processes=None) as pool:
                results = list(
                    pool.map(
                        lambda idx, ds=ds, min_duration=min_duration, max_duration=max_duration: is_sample_in_timerange(
                            idx, ds, min_duration, max_duration
                        ),
                        indices,
                    ),
                )
            t1 = time.time()
            log(f"Elapsed : {t1-t0} seconds")
            self.valid_indices = [i for i, vi in results if vi]

            pickle.dump(self.valid_indices, open(cachepath, "wb"))
        self.ds = ds

    def __getitem__(self, idx):
        return self.ds[self.valid_indices[idx]]

    def __len__(self):
        return len(self.valid_indices)


class CharMap(object):
    """
    Object in charge of performing the char <-> int conversion
    It holds the vocabulary and the functions required for performing
    the conversions in the two directions
    """

    _BLANK = 172  # Corresponds to '¬'
    _SOS = 182  # Corresponds to '¶'
    _EOS = 166  # Corresponds to '¦'

    def __init__(self):
        ord_chars = frozenset().union(
            range(97, 123),  # a-z
            range(48, 58),  # 0-9
            [32, 39, 44, 46],  # <space> <,> <.> <'>
            [self._SOS],  # <sos>¶
            [self._EOS],  # <eos>¦
            [10060],  # <unk> ❌
        )

        # The pad symbol is added first to guarantee it has idx 0
        self.idx2char = [chr(self._BLANK)] + [chr(i) for i in ord_chars]
        self.char2idx = {c: idx for (idx, c) in enumerate(self.idx2char)}

        self.equivalent_char = {}
        for i in range(224, 229):
            self.equivalent_char[chr(i)] = "a"
        for i in range(232, 236):
            self.equivalent_char[chr(i)] = "e"
        for i in range(236, 240):
            self.equivalent_char[chr(i)] = "i"
        for i in range(242, 247):
            self.equivalent_char[chr(i)] = "o"
        for i in range(249, 253):
            self.equivalent_char[chr(i)] = "u"
        # Remove the punctuation marks
        for c in ["!", "?", ";"]:
            self.equivalent_char[c] = "."
        for c in ["-", "…", ":"]:
            self.equivalent_char[c] = " "
        self.equivalent_char["—"] = ""
        # This 'œ' in self.equivalent_char returns False... why ?
        # self.equivalent_char['œ'] = 'oe'
        # self.equivalent_char['ç'] = 'c'
        self.equivalent_char["’"] = "'"

    @property
    def vocab_size(self):
        return len(self.idx2char)

    @property
    def eoschar(self):
        return chr(self._EOS)

    @property
    def eos(self):
        return self.char2idx[self.eoschar]

    @property
    def soschar(self):
        return chr(self._SOS)

    @property
    def blankid(self):
        return self.char2idx[chr(self._BLANK)]

    def encode(self, utterance):
        utterance = self.soschar + utterance.lower() + self.eoschar

        # Remove the accentuated characters
        utterance = [
            self.equivalent_char[c] if c in self.equivalent_char else c
            for c in utterance
        ]
        # Replace the unknown characters
        utterance = ["❌" if c not in self.char2idx else c for c in utterance]
        return [self.char2idx[c] for c in utterance]

    def decode(self, tokens):
        return "".join([self.idx2char[it] for it in tokens])


class WaveformProcessor(object):
    def __init__(
        self,
        rate: float,
        win_length: float,
        win_step: float,
        nmels: int,
        augment: bool,
        spectro_normalization: Tuple[float, float],
    ):
        """
        Args:
            rate: the sampling rate of the waveform
            win_length: the length in second of the window for the STFT
            win_step: the length in second of the step size of the STFT window
            nmels:  the number of mel scales to consider
            augment (bool) : whether to use data augmentation or not
        """
        self.nfft = int(win_length * rate)
        self.nstep = int(win_step * rate)
        self.spectro_normalization = spectro_normalization

        ###########################
        #### START CODING HERE ####
        ###########################
        # @TEMPL@self.transform_tospectro = None
        # @SOL
        modules = [
            MelSpectrogram(
                sample_rate=rate, n_fft=self.nfft, hop_length=self.nstep, n_mels=nmels
            ),
            AmplitudeToDB(),
        ]
        self.transform_tospectro = nn.Sequential(*modules)
        # SOL@

        self.transform_augment = None
        if augment:
            time_mask_duration = 0.1  # s.
            time_mask_nsamples = int(time_mask_duration / win_step)
            nmel_mask = nmels // 4

            modules = [FrequencyMasking(nmel_mask), TimeMasking(time_mask_nsamples)]
            self.transform_augment = nn.Sequential(*modules)
        ##########################
        #### STOP CODING HERE ####
        ##########################

    def get_spectro_length(self, waveform_length: int):
        """
        Computes the length of the spectrogram given the length
        of the waveform

        Args:
            waveform_lengths: the number of samples of the waveform

        Returns:
            int: the number of time samples in the spectrogram
        """
        return waveform_length // self.nstep + 1

    def __call__(self, waveforms: torch.Tensor):
        """
        Apply the transformation on the input waveform tensor
        The time dimension is smalled because of the hop_length given
        to the MelSpectrogram object.

        Args:
            waveforms(torch.Tensor) : (Tx, B) waveform
        Returns:
            spectrograms(torch.Tensor): (Tx//nstep + 1, B, n_mels)
        """
        # Compute the spectorgram
        waveforms = waveforms.transpose(0, 1)  # from (T, B) to (B, T)
        spectro = self.transform_tospectro(waveforms)  # (B, n_mels, T)

        # Normalize the spectrogram
        if self.spectro_normalization is not None:
            spectro = (
                spectro - self.spectro_normalization[0]
            ) / self.spectro_normalization[1]

        # Apply data augmentation
        if self.transform_augment is not None:
            spectro = self.transform_augment(spectro)

        # spectrograms is (B, n_mel, T)
        # we permute it to be (T, B, n_mel)
        return spectro.permute(2, 0, 1)


class BatchCollate(object):
    """
    Collator for the individual data to build up the minibatches
    """

    def __init__(
        self,
        nmels: int,
        augment: bool,
        spectro_normalization: Tuple[float, float] = None,
    ):
        """
        Args:
            nmels (int) : the number of mel scales to consider
            augment (bool) : whether to use data augmentation or not
            spectro_normalization (tuple(float, float)): mean, std
        """
        self.waveform_processor = WaveformProcessor(
            _DEFAULT_RATE,
            _DEFAULT_WIN_LENGTH * 1e-3,
            _DEFAULT_WIN_STEP * 1e-3,
            nmels,
            augment,
            spectro_normalization,
        )
        self.charmap = CharMap()

    def __call__(self, batch):
        """
        Builds and return a minibatch of data as a tuple (inputs, targets)
        All the elements are padded to be of equal time

        Returns:
            a tuple (spectros, targets) with :
                spectors : (Batch size, time, n_mels)
                targets : (Batch size, time)
        """
        # Extract the subcomponents
        # The CommonVoice dataset returns (waveform, sample_rate, dictionnary)
        # waveform is (1, seq_len)
        # dictionnary has the 'sentence' key for the transcript
        waveforms = [w.squeeze() for w, _, _ in batch]
        rates = [r for _, r, _ in batch]
        transcripts = [
            torch.LongTensor(self.charmap.encode(d["sentence"])) for _, _, d in batch
        ]

        # We resample the signal to the _DEFAULT_RATE
        waveforms = [
            torchaudio.transforms.Resample(r, _DEFAULT_RATE)(w)
            if r != _DEFAULT_RATE
            else w
            for w, r in zip(waveforms, rates)
        ]

        # Sort the waveforms and transcripts by decreasing waveforms length
        wt_sorted = sorted(
            zip(waveforms, transcripts), key=lambda wt: wt[0].shape[0], reverse=True
        )
        waveforms = [wt[0] for wt in wt_sorted]
        transcripts = [wt[1] for wt in wt_sorted]

        # Compute the lengths of the spectrograms from the lengths
        # of the waveforms
        waveforms_lengths = [w.shape[0] for w in waveforms]
        spectro_lengths = [
            self.waveform_processor.get_spectro_length(wl) for wl in waveforms_lengths
        ]
        transcripts_lengths = [t.shape[0] for t in transcripts]

        ###########################
        #### START CODING HERE ####
        ###########################

        ##
        # Step 1 : pad the sequence of tensors waveforms so that they all have the same
        #          length. The resulting tensor must be (T, B) where T is the maximal
        #          duration of the elements in waveforms.
        #          (1 line)
        # @TEMPL@waveforms = None
        waveforms = pad_sequence(waveforms)  # @SOL@

        # Step 2 : Apply the waveform_processor transform to the
        #          waveform tensor. Note the resulting tensor is (T, B, n_mels)
        #          (1 line)
        # @TEMPL@spectrograms = None
        spectrograms = self.waveform_processor(waveforms)  # @SOL@

        # Step 3 : pack the tensor of spectrograms given their length
        #          as computed in spectro_lengths
        #          (1 line)
        # @TEMPL@spectrograms = None
        # @SOL
        spectrograms = pack_padded_sequence(spectrograms, lengths=spectro_lengths)
        # SOL@

        # Step 3 : pad the sequence of tensors transcripts, so that all the rows
        #          of the resulting padded tensor have the same legnth. The resulting
        #          tensor is (Ty, B)
        #          (1 line)
        # @TEMPL@transcripts = None
        transcripts = pad_sequence(transcripts)  # @SOL@

        # Step 4 : pack the tensor of transcripts given their length as
        #          computed in transcripts_length
        #          Note : this packed tensor must be given enforce_sorted=False
        #          to ensure the i-th transcript stay aligned with the i-th
        #          spectrogram. Otherwise, torch would sort the transcripts
        #          independently from the spectrograms and the alignement between the
        #          spectrograms and the transcripts would be messed up
        #          (1 line)

        # @TEMPL@transcripts = None
        # @SOL
        transcripts = pack_padded_sequence(
            transcripts, lengths=transcripts_lengths, enforce_sorted=False
        )
        # SOL@

        ##########################
        #### STOP CODING HERE ####
        ##########################
        return spectrograms, transcripts


def get_dataloaders(
    commonvoice_root: str,
    commonvoice_version: str,
    cuda: bool,
    batch_size: int = 64,
    n_threads: int = 4,
    min_duration: float = 1,
    max_duration: float = 5,
    small_experiment: bool = False,
    train_augment: bool = False,
    nmels: int = _DEFAULT_NUM_MELS,
    logger=None,
    normalize=True,
    overwrite_index=False,
):
    """
    Build and return the pytorch dataloaders

    Args:
        commonvoice_root (str or Path) : the root directory where the dataset
                                         is stored
        commonvoice_version (str) : the version of the dataset to consider, e.g. 1, 6.1, ..
        cuda (bool) : whether to use cuda or not. Used for creating tensors
                      on the right device
        batch_size (int) : the number of samples per minibatch
        n_threads (int) : the number of threads to use for dataloading
        min_duration (float) : the minimal duration of the recordings to
                               consider
        max_duration (float) : the maximal duration of the recordings to
                               consider
        small_experiment (bool) : whether or not to use small subsets, usefull for debug
        train_augment (bool) : whether to use SpecAugment
        nmels (int) : the number of mel scales to consider
        logger : an optional logging logger
        normalize : wheter or not to center reduce the spectrograms
        overwrite_index: whether or not to overwrite the cache files for the index of sequences to consider
    """

    if small_experiment:
        min_duration = None
        max_duration = None

    def dataset_loader(fold, version, overwrite_index):
        ds = load_dataset(
            fold,
            commonvoice_root=commonvoice_root,
            commonvoice_version=commonvoice_version,
        )
        if not min_duration and not max_duration:
            return ds
        else:
            return DatasetFilter(
                ds=ds,
                min_duration=min_duration,
                max_duration=max_duration,
                cacheprefix=fold + "-" + version,
                overwrite_index=overwrite_index,
                logger=logger,
            )

    valid_dataset = dataset_loader("dev", commonvoice_version, overwrite_index)
    train_dataset = dataset_loader("train", commonvoice_version, overwrite_index)
    test_dataset = dataset_loader("test", commonvoice_version, overwrite_index)
    if small_experiment:
        indices = range(batch_size)

        train_dataset = torch.utils.data.Subset(train_dataset, indices=indices)
        valid_dataset = torch.utils.data.Subset(valid_dataset, indices=indices)
        test_dataset = torch.utils.data.Subset(test_dataset, indices=indices)

    if normalize:
        # Compute the normalization on the training set
        # batch_collate_norm = BatchCollate(nmels, augment=False)
        # norm_loader = torch.utils.data.DataLoader(
        #     train_dataset,
        #     batch_size=batch_size,
        #     shuffle=False,
        #     num_workers=n_threads,
        #     collate_fn=batch_collate_norm,
        #     pin_memory=cuda,
        # )
        # mean_spectro, std_spectro = 0, 0
        # N_elem = 0
        # for spectros, _ in tqdm.tqdm(norm_loader):
        #     unpacked_raveled = unpack_ravel(spectros)
        #     mean_spectro += unpacked_raveled.sum().item()
        #     N_elem += functools.reduce(operator.mul, unpacked_raveled.shape, 1)
        # mean_spectro /= N_elem

        # for spectros, _ in tqdm.tqdm(norm_loader):
        #     unpacked_raveled = unpack_ravel(spectros)
        #     std_spectro += ((unpacked_raveled - mean_spectro) ** 2).sum()
        # std_spectro = (torch.sqrt(std_spectro / N_elem)).item()

        # Fix for speeding up debuggin
        mean_spectro = -31
        std_spectro = 32
        normalization = (mean_spectro, std_spectro)
    else:
        normalization = None

    if logger is not None:
        logger.info(f"Normalization coefficients : {mean_spectro}, {std_spectro}")

    batch_collate_train_fn = BatchCollate(
        nmels, augment=train_augment, spectro_normalization=normalization
    )
    batch_collate_infer_fn = BatchCollate(
        nmels, augment=False, spectro_normalization=normalization
    )

    print(f"Building a train loader with batch size = {batch_size}")
    print(f"The dataset contains {len(train_dataset)} samples")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_threads,
        collate_fn=batch_collate_train_fn,
        pin_memory=cuda,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_threads,
        collate_fn=batch_collate_infer_fn,
        pin_memory=cuda,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_threads,
        collate_fn=batch_collate_infer_fn,
        pin_memory=cuda,
    )

    return train_loader, valid_loader, test_loader


def plot_spectro(
    spectrogram: torch.Tensor,
    transcript: torch.Tensor,
    win_step: float,
    charmap: CharMap,
) -> None:
    """
    Args:
        spectrogram (time, n_mels) tensor
        trancript (target_len, ) LongTensor
        win_step is the stride of the windows, in seconds, for computing the
                 spectrogram
        charmap : object for converting between int and char for the transcripts
    """
    fig = plt.figure(figsize=(10, 2))
    ax = fig.add_subplot()

    im = ax.imshow(
        spectrogram.T,
        extent=[0, spectrogram.shape[0] * win_step, 0, spectrogram.shape[1]],
        aspect="auto",
        cmap="magma",
        origin="lower",
    )
    ax.set_ylabel("Mel scale")
    ax.set_xlabel("TIme (s.)")
    ax.set_title("{}".format(charmap.decode(transcript)))
    plt.colorbar(im)
    plt.tight_layout()


# @SOL
def ex_charmap():
    charmap = CharMap()

    # Some encoding/decoding tests
    utterance = "Je vais m'éclater avec des RNNs !"
    encoded = charmap.encode(utterance)
    decoded = charmap.decode(encoded)
    print(f'"{utterance}" -> "{encoded}" -> "{decoded}" ')

    # For some reasons, the replacement of œ fails
    print(
        charmap.decode(
            charmap.encode(
                "nous sommes heureux de vous souhaiter nos meilleurs vœux pour 2021"
            )
        )
    )
    print("œ" in charmap.char2idx)

    print(f"The vocabulary contains {charmap.vocab_size} characters")

    print(charmap.decode([16, 20, 3, 22, 36, 37, 1, 29, 32, 26, 31, 5, 2]))


def ex_waveform_spectro():
    dataset = load_dataset(
        "train", _DEFAULT_COMMONVOICE_ROOT, _DEFAULT_COMMONVOICE_VERSION
    )

    # Take one of the waveforms
    idx = 10
    waveform, rate, dictionary = dataset[idx]
    walker = dataset._walker
    path_to_audio = os.path.join(dataset._path, dataset._folder_audio, walker[idx][1])
    print(f"I will be loading {path_to_audio}, with the transcript {walker[idx][2]}")

    n_begin = rate  # 1 s.
    n_end = 3 * rate  # 2 s.
    waveform = waveform[:, n_begin:n_end]  # B, T

    nfft = int(_DEFAULT_WIN_LENGTH * 1e-3 * _DEFAULT_RATE)
    # nmels = _DEFAULT_NUM_MELS
    nstep = int(_DEFAULT_WIN_STEP * 1e-3 * _DEFAULT_RATE)
    trans_spectro = nn.Sequential(
        Spectrogram(n_fft=nfft, hop_length=nstep), AmplitudeToDB()
    )
    spectro = trans_spectro(waveform)  # B, n_mels, T

    trans_mel_spectro = WaveformProcessor(
        rate=rate,
        win_length=_DEFAULT_WIN_LENGTH * 1e-3,
        win_step=_DEFAULT_WIN_STEP * 1e-3,
        nmels=_DEFAULT_NUM_MELS,
        augment=False,
        spectro_normalization=None,
    )
    mel_spectro = trans_mel_spectro(waveform.transpose(0, 1))  # T, B, n_mels
    plot_spectro(mel_spectro[:, 0, :], [], _DEFAULT_WIN_STEP * 1e-3, CharMap())

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 3))

    ax = axes[0]
    ax.plot([i / rate for i in range(n_begin, n_end)], waveform[0])
    ax.set_xlabel("Time (s.)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")

    ax = axes[1]
    im = ax.imshow(
        spectro[0],
        extent=[n_begin / rate, n_end / rate, 0, spectro.shape[1]],
        aspect="auto",
        cmap="magma",
        origin="lower",
    )
    ax.set_ylabel("Frequency bins")
    ax.set_xlabel("TIme (s.)")
    ax.set_title("Spectrogram (dB)")
    fig.colorbar(im, ax=ax)

    ax = axes[2]
    im = ax.imshow(
        mel_spectro[:, 0, :].T,
        extent=[n_begin / rate, n_end / rate, 0, mel_spectro.shape[0]],
        aspect="auto",
        cmap="magma",
        origin="lower",
    )
    ax.set_ylabel("Mel scales")
    ax.set_xlabel("TIme (s.)")
    ax.set_title("Mel-Spectrogram (dB)")
    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("waveform_to_spectro.png")
    logging.info("Image saved to waveform_to_spectro.png")
    plt.show()


def ex_spectro():

    charmap = CharMap()

    # Data loading
    batch_size = 4
    loaders = get_dataloaders(
        _DEFAULT_COMMONVOICE_ROOT,
        _DEFAULT_COMMONVOICE_VERSION,
        cuda=False,
        n_threads=4,
        min_duration=None,
        max_duration=None,
        batch_size=batch_size,
        train_augment=True,
        normalize=False,
    )
    train_loader, valid_loader, test_loader = loaders

    X, y = next(iter(train_loader))
    # X is (Tx, batch_size, n_mels)
    X, lens_X = pad_packed_sequence(X)
    # Y is (Ty, batch_size)
    y, lens_y = pad_packed_sequence(y)

    print("Some decoder texts from the LongTensors")
    for iy, li in enumerate(lens_y):
        print(charmap.decode(y[:li, iy]))

    fig, axes = plt.subplots(nrows=batch_size, ncols=1, sharex=True, figsize=(10, 7))
    for iax, ax in enumerate(axes):
        # spectroi is of shape (Tx, n_mels)
        print(X.shape)
        spectroi = X[:, iax, :]
        im = ax.imshow(
            spectroi.T,
            extent=[
                0,
                spectroi.shape[0] * _DEFAULT_WIN_STEP * 1e-3,
                0,
                spectroi.shape[1],
            ],
            aspect="auto",
            cmap="magma",
            origin="lower",
        )
        # vmin=-100, vmax=10)
        ax.set_ylabel("Mel scale")
        ax.set_title("{}".format(charmap.decode(y[: lens_y[iax], iax])))
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.xlabel("Time (s.)")
    plt.savefig("spectro.png")
    logging.info("Image saved to spectro.png")
    plt.show()


def ex_augmented_spectro():
    charmap = CharMap()

    # Data loading
    batch_size = 4
    loaders = get_dataloaders(
        _DEFAULT_COMMONVOICE_ROOT,
        _DEFAULT_COMMONVOICE_VERSION,
        cuda=False,
        n_threads=4,
        min_duration=None,
        max_duration=None,
        batch_size=batch_size,
        train_augment=True,
        normalize=False,
    )
    train_loader, valid_loader, test_loader = loaders

    # From the validation set
    X, y = next(iter(valid_loader))

    # X is (T, B, n_mels)
    X, lens_X = pad_packed_sequence(X)

    # Y is (T, B)
    y, lens_y = pad_packed_sequence(y)
    idx = 1
    plot_spectro(X[:, idx, :], y[: lens_y[idx], idx], _DEFAULT_WIN_STEP * 1e-3, charmap)
    print("spectro valid")
    plt.savefig("spectro_valid.png")

    # From the validation set
    X, y = next(iter(train_loader))

    # X is (T, B, n_mels)
    X, lens_X = pad_packed_sequence(X)
    # Y is (T, B)
    y, lens_y = pad_packed_sequence(y)
    idx = 0
    print(X.shape, _DEFAULT_WIN_STEP * 1e-3)
    plot_spectro(X[:, idx, :], y[: lens_y[idx], idx], _DEFAULT_WIN_STEP * 1e-3, charmap)
    print("spectro train")
    plt.savefig("spectro_train.png")
    logging.info("Image saved to spectro_train.png")

    plt.show()


def order_by_length():
    dataset_loader = functools.partial(
        load_dataset,
        commonvoice_root=_DEFAULT_COMMONVOICE_ROOT,
        commonvoice_version=_DEFAULT_COMMONVOICE_VERSION,
    )

    def forder(ds):
        idx_lens = [(w.shape[1], itrain) for itrain, (w, _, _) in enumerate(ds)]
        return sorted(idx_lens, key=lambda wi: wi[0])

    for k in ["dev", "test", "train"]:
        print(f"Ordering {k}")
        sorted_idx = forder(dataset_loader(k))
        with open(f"sorted_idx_{k}", "w") as f:
            f.write("\n".join(f"{idxi},{leni}" for leni, idxi in sorted_idx))


def test_spectro():
    dataset = load_dataset(
        "train", _DEFAULT_COMMONVOICE_ROOT, _DEFAULT_COMMONVOICE_VERSION
    )

    # Take one of the waveforms
    idx = 10
    waveform, rate, dictionary = dataset[idx]

    waveform = waveform.transpose(0, 1)
    print(waveform.shape)

    win_step = _DEFAULT_WIN_STEP * 1e-3
    trans_mel_spectro = WaveformProcessor(
        rate=rate,
        win_length=_DEFAULT_WIN_LENGTH * 1e-3,
        win_step=win_step,
        nmels=_DEFAULT_NUM_MELS,
        augment=False,
        spectro_normalization=None,
    )
    mel_spectro = trans_mel_spectro(waveform).squeeze()  # (T, N_MELS)

    fig = plt.figure(figsize=(10, 2))
    ax = fig.add_subplot()

    im = ax.imshow(
        mel_spectro.T,
        extent=[0, mel_spectro.shape[0] * win_step, 0, mel_spectro.shape[1]],
        aspect="auto",
        cmap="magma",
        origin="lower",
    )
    ax.set_ylabel("Mel scale")
    ax.set_xlabel("TIme (s.)")
    ax.set_title("Log mel spectrogram")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("test_spectro.png")
    logging.info("Image saved to test_spectro.png")
    plt.show()


# SOL@
if __name__ == "__main__":
    # @TEMPL@pass
    # @SOL
    # order_by_length()
    ex_charmap()
    test_spectro()
    ex_waveform_spectro()
    ex_spectro()
    ex_augmented_spectro()
    # SOL@
