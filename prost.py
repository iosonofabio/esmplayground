import os
import sys
import argparse
import pathlib
import glob
import h5py
import hdf5plugin
import numpy as np
from scipy.fftpack import dct, idct
import torch
from esm.data import Alphabet


def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-sequence representations and model outputs for sequences in multiple FASTA files"  # noqa
    )

    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    parser.add_argument('--species', type=str, default='')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--max-sequences', type=int, default=-1)
    return parser


# Straight from: https://github.com/MesihK/prost/blob/master/src/pyprost/prosttools.py
def iterate_fasta(fastafile):
    from itertools import groupby
    with open(fastafile) as fh:
        faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))
        for header in faiter:
            header = next(header)[1:].strip()
            seq = "".join(s.strip() for s in next(faiter))
            yield header, seq


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

    fn_out = pathlib.Path(f'embeddings_prost.h5')

    #https://github.com/pytorch/pytorch/issues/52286
    #torch._C._jit_set_bailout_depth(0) # Use _jit_set_fusion_strategy, bailout depth is deprecated.
    
    model = torch.jit.load('traced_esm1b_25_13.pt').eval()
    alphabet = Alphabet.from_architecture("ESM-1b")
    batch_converter = alphabet.get_batch_converter()
    
    #https://stackoverflow.com/a/63616077
    #This prevents memory leak
    for param in model.parameters():
        param.grad = None
        param.requires_grad = False
    
    print('Transfer model to GPU')
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")
    else:
        torch._C._jit_set_profiling_mode(False)
        model = torch.jit.freeze(model)
        model = torch.jit.optimize_for_inference(model)
    
    def _embed(seq):
        _, _, toks = batch_converter([("prot",seq)])
        if torch.cuda.is_available() and not args.nogpu:
            toks = toks.to(device="cuda", non_blocking=True)
        results = model(toks)
        for i in range(len(results)):
            results[i] = results[i].to(device="cpu")[0].detach().numpy()
        return results
    
    def embed(seq):
        l = len(seq)
        embtoks = None
        if l > 1022:
            piece = int(l/1022)+1
            part = l/piece
            for i in range(piece):
                st = int(i*part)
                sp = int((i+1)*part)
                results = _embed(seq[st:sp])
                if embtoks is not None:
                    for i in range(len(results)):
                        embtoks[i] = np.concatenate((embtoks[i][:len(embtoks[i])-1],results[i][1:]),axis=0)
                else:
                    embtoks = results
        else:
            embtoks = _embed(seq)
        return embtoks

    for fasta_file in fasta_files:
        species = fasta_file.split('/')[-1].split('.')[0]
        print(species)

        with h5py.File(fn_out, 'a') as h5:
            if (not args.overwrite) and species in h5:
                print('Already exists, skipping.')
                continue

        print('Read FASTA')
        fasta_file = pathlib.Path(fasta_file)
        fasta_iter = iterate_fasta(fasta_file)

        sequence_labels = []
        sequence_representations = []
        with torch.no_grad():
            for i, (name, seq) in enumerate(fasta_iter):
                print(
                    f"Processing sequence {i + 1}"
                )

                # Embed in chunks if longer than 1022
                if seq:
                    esm_output = embed(seq.upper())
                    q25_544 = quant2D(esm_output[1], 5, 44)
                    q13_385 = quant2D(esm_output[0], 3, 85)
                    quantised = np.concatenate([q25_544,q13_385])
                else:
                    quantised = np.zeros(475, 'u1')

                # Accumulate outputs
                sequence_labels.append(name)
                sequence_representations.append(quantised)

                if i + 1 == args.max_sequences:
                    break

        sequence_representations = np.vstack(sequence_representations)

        print('Store to file')
        comp_kwargs = hdf5plugin.Zstd(clevel=compression)
        with h5py.File(fn_out, 'a') as h5:
            if species in h5:
                h5.pop(species)
            speciesg = h5.create_group(species)
            speciesg.create_dataset(
                'features',
                data=np.array(sequence_labels).astype('S'),
            )
            speciesg.create_dataset(
                'embeddings',
                data=sequence_representations,
                dtype='u1',
            )

