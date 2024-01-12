#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import argparse
import pathlib
import glob
import h5py
import hdf5plugin
import numpy as np

import torch

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer


def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-sequence representations and model outputs for sequences in multiple FASTA files"  # noqa
    )
    parser.add_argument("--toks_per_batch", type=int, default=4096, help="maximum batch size")
    parser.add_argument(
        "--repr_layers",
        type=int,
        default=[33],
        nargs="+",
        help="layers indices from which to extract representations (0 to num_layers, inclusive)",
    )
    parser.add_argument(
        "--truncation_seq_length",
        type=int,
        default=1022,
        help="truncate sequences longer than the given value",
    )

    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    parser.add_argument('--species', type=str, default='')
    parser.add_argument('--quantise', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    return parser


def idct_protein_tokens(token_representations, i, tokens_len, layers=(25, 13), layer_shapes=((5, 44), (3, 85))):
    from scipy.fftpack import dct, idct
    # Straight from: https://github.com/MesihK/prost/blob/master/src/pyprost/prosttools.py
    def standard_scale(v):
        M = np.max(v)
        m = np.min(v)
        return (v - m) / float(M - m)

    def iDCTquant(v, n):
        f = dct(v.T, type=2, norm='ortho')
        trans = idct(f[:,:n], type=2, norm='ortho')
        for i in range(len(trans)):
            trans[i] = standard_scale(trans[i])
        return trans.T
    
    def quant2D(emb, n=5, m=44):
        # First and last tokens are BoS and EoS
        dct = iDCTquant(emb[1: len(emb) - 1], n)
        ddct = iDCTquant(dct.T, m).T
        ddct = ddct.reshape(n*m)
        # Convert 0-1 to actual uint8
        return (ddct * 127).astype('int8')

    quants = []
    for layer, (n, m) in zip(layers, layer_shapes):
        token_layer = token_representations[layer][i]
        token_layer_numpy = token_layer.to(device="cpu")[0].detach().numpy()
        import ipdb; ipdb.set_trace()
        quant = quant2D(token_layer_numpy, n, m)
        quants.append(quant)
    return np.concatenate(quants)



if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()

    fasta_files = glob.glob('peptide_sequences/*.fasta')
    if args.species:
        species = args.species.split(',')
        tmp = set(fasta_files)
        fasta_files = []
        for organism in species:
            for fn in tmp:
                if organism in fn:
                    break
            else:
                raise ValueError(f'Species not found: {organism}')
            fasta_files.append(fn)
            tmp.remove(fn)

    if args.quantise:
        fn_out = pathlib.Path(f'embeddings_quantise.h5')
    else:
        fn_out = pathlib.Path(f'embeddings.h5')

    model, alphabet = pretrained.esm2_t33_650M_UR50D()
    model.eval()
    if isinstance(model, MSATransformer):
        raise ValueError(
            "This script currently does not handle models with MSA input (MSA Transformer)."
        )

    print('Transfer model to GPU')
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")

    for fasta_file in fasta_files:
        species = fasta_file.split('/')[-1].split('.')[0]
        print(species)

        with h5py.File(fn_out, 'a') as h5:
            if (not args.overwrite) and species in h5:
                print('Already exists, skipping.')
                continue

        fasta_file = pathlib.Path(fasta_file)

        print('Read FASTA')
        dataset = FastaBatchedDataset.from_file(fasta_file)
        batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=alphabet.get_batch_converter(args.truncation_seq_length), batch_sampler=batches
        )
        print(f"Finished reading {fasta_file} with {len(dataset)} sequences")

        if args.quantise:
            repr_layers = [25, 13]
        else:
            assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
            repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]

        sequence_labels = []
        sequence_representations = []

        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in enumerate(data_loader):
                print(
                    f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
                )
                if torch.cuda.is_available() and not args.nogpu:
                    toks = toks.to(device="cuda", non_blocking=True)

                out = model(toks, repr_layers=repr_layers, return_contacts=False)

                print('Extract per-residue representations')
                token_representations = {
                    layer: t.to(device="cpu") for layer, t in out["representations"].items()
                }

                print('Generate per-sequence representations via averaging')
                # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
                # Similarly, the last token is end-of-sequence I guess
                batch_lens = (toks != alphabet.padding_idx).sum(1)
                for i, tokens_len in enumerate(batch_lens):
                    sequence_labels.append(labels[i])
                    # NOTE: each protein is ultimately represented as the average of all tokens, this might
                    # not be the best option (e.g. PROST)
                    if args.quantise:
                        sequence_repr = idct_protein_tokens(token_representations, i, tokens_len)
                    else:
                        sequence_repr = token_representations[33][i, 1 : tokens_len - 1].mean(0)
                    sequence_representations.append(sequence_repr)

        print('Store to file')
        comp_kwargs = hdf5plugin.Zstd(clevel=22)
        with h5py.File(fn_out, 'a') as h5:
            if species in h5:
                h5.pop(species)
            speciesg = h5.create_group(species)
            speciesg.create_dataset('features', data=np.array(sequence_labels).astype('S'))

            mat = np.array(torch.vstack(sequence_representations))
            if not args.quantise:
                mat = mat.astype('f4')
                dtype = 'f4'
            else:
                dtype = 'u1'
            speciesg.create_dataset('embeddings', data=mat, dtype=dtype)
