import argparse
import torch
import esm



if __name__ == '__main__':


    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
    )
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    parser.add_argument("--layer", type=int, default=33, help="Layer to use for representation")
    args = parser.parse_args()

    print('Load ESM-2 model')
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    
    print('Transfer model to GPU')
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")
    
    print('Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)')
    data = [
        ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
        ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ("protein3",  "K A <mask> I S Q"),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    
    print('Extract per-residue representations')
    with torch.no_grad():
        if torch.cuda.is_available() and not args.nogpu:
            batch_tokens = batch_tokens.to(device="cuda", non_blocking=True)
    
        results = model(batch_tokens, repr_layers=[args.layer], return_contacts=True)
        token_representations = results["representations"][args.layer]
    
    print('Generate per-sequence representations via averaging')
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
    
    
    def get_index(k, n):
        p = n-1
        kk = k
        j = 1
        i = 0
        while kk >= p:
            kk = kk-p
            p = p-1
            j += 1
            i += 1
        return i, j+kk
    
    ds = torch.nn.functional.pdist(torch.vstack(sequence_representations))
    amin = ds.argmin()
    
