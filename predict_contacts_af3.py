import os
import json
import numpy as np
import datetime
from pathlib import Path

import jax
from jax import numpy as jnp
import scipy
import gc
from alphafold3.common import folding_input
from alphafold3.model import model, params
from alphafold3.model.components import utils
from alphafold3.data import featurisation
from alphafold3.constants import chemical_components

import haiku as hk

"""
Modified predict_contacts_af3.py script that:
- Accepts AlphaFold2-style config JSON
- Parses paired MSA from a single A3M file
- Splits it into unpaired MSAs for each chain based on Ls
- Predicts inter-chain contacts using AlphaFold3

JSON format:
{
  "n_recycle": 3,
  "prefix_template": "{stem}",
  "outdir": "/path/to/output",
  "items": [
    {
      "a3m": "/path/to/paired.a3m",
      "Ls": [lenA, lenB],
      "prefix": "name"
    }
  ]
}
"""

NUM_DIFFUSION_SAMPLES = 1
FLASH_ATTENTION_IMPLEMENTATION = "triton"
CONTACT_BIN_CUTOFF = 32


def parse_paired_a3m(a3m_path):
    with open(a3m_path) as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    entries = []
    desc, seq = None, ""
    for line in lines:
        if line.startswith(">"):
            if desc is not None:
                entries.append((desc, seq))
            desc = line[1:].strip()
            seq = ""
        else:
            seq += line
    if desc and seq:
        entries.append((desc, seq))

    return entries  # List of (description, full_sequence)


def split_msa_to_a3m_strings(entries, Ls):
    num_chains = len(Ls)
    split_msas = [[] for _ in range(num_chains)]

    for desc, full_seq in entries:
        assert len(full_seq) == sum(Ls), (
            f"MSA sequence length {len(full_seq)} does not match sum(Ls) = {sum(Ls)}"
        )
        offset = 0
        for i, L in enumerate(Ls):
            subseq = full_seq[offset:offset + L]
            split_msas[i].append(f">{desc}\n{subseq}")
            offset += L

    return ["\n".join(msa_list) + "\n" for msa_list in split_msas]


def make_model_runner(model_dir, n_recycle):
    cfg = model.Model.Config()
    cfg.global_config.flash_attention_implementation = FLASH_ATTENTION_IMPLEMENTATION
    cfg.num_recycles = n_recycle
    cfg.heads.diffusion.eval.num_samples = NUM_DIFFUSION_SAMPLES
    cfg.heads.distogram.enabled = True

    device = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices("cpu")[0]

    class Runner:
        def __init__(self):
            self.params = params.get_model_haiku_params(model_dir=Path(model_dir))

            @hk.transform
            def forward_fn(batch):
                return model.Model(cfg)(batch)

            self._apply = jax.jit(forward_fn.apply, device=device)

        def run(self, features, seed=0):
            batch = jax.device_put(
                jax.tree_util.tree_map(jnp.asarray, utils.remove_invalidly_typed_feats(features)),
                device,
            )
            rng_key = jax.random.PRNGKey(seed)
            result = self._apply(self.params, rng_key, batch)
            result = jax.tree.map(np.asarray, result)
            result = dict(result)
            result["__identifier__"] = self.params["__meta__"]["__identifier__"].tobytes()
            return batch, result

    return Runner()


def build_fold_input(seqs, a3m_strings, ids, name, seed):
    chains = []
    for seq, a3m, cid in zip(seqs, a3m_strings, ids):
        chains.append(
            folding_input.ProteinChain(
                sequence=seq,
                id=cid,
                unpaired_msa=a3m,
                paired_msa="",
                templates=[],
                ptms=[]
            )
        )

    return folding_input.Input(
        name=name,
        rng_seeds=[seed],
        chains=chains
    )


def extract_inter_contact(token_chain_ids, disto, Ls):
    a_idx = 'A'
    b_idx = 'B'
    idxA = np.where(token_chain_ids == a_idx)[0]
    idxB = np.where(token_chain_ids == b_idx)[0]
    contact = disto[np.ix_(idxA, idxB)] 
    contact = contact[:, :, :CONTACT_BIN_CUTOFF].sum(axis=-1)
    # Check lengths
    if contact.shape != (Ls[0], Ls[1]):
        raise ValueError(f"Contact map shape {contact.shape} does not match expected ({len(idxA)}, {len(idxB)})")
    return contact, idxA, idxB

def get_config():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--config_json", required=True,
        help="Path to a JSON file describing all runs."
    )
    args = parser.parse_args()

    with open(args.config_json, "r") as f:
        cfg = json.load(f)
    
    # Validate & normalize
    if "items" not in cfg or not isinstance(cfg["items"], list) or len(cfg["items"]) == 0:
        raise SystemExit("config_json must contain a non-empty 'items' list.")
    
    # Defaults (all optional)
    cfg.setdefault("n_recycle", 3)
    cfg.setdefault("prefix_template", "{stem}")
    cfg.setdefault("outdir", ".")   # global default = cwd
    cfg.setdefault("model_dir", ".")

    # Normalize and ensure outdir exists
    cfg["outdir"] = os.path.expanduser(str(cfg["outdir"]))
    os.makedirs(cfg["outdir"], exist_ok=True)

    for i, it in enumerate(cfg["items"]):
        if "a3m" not in it:
            raise SystemExit(f"items[{i}] missing required key 'a3m'.")
        if "Ls" not in it or not isinstance(it["Ls"], list) or not all(isinstance(x, int) for x in it["Ls"]):
            raise SystemExit(f"items[{i}] must include 'Ls' as a list of integers.")
        # Normalize absolute path early (optional)
        it["a3m"] = os.path.expanduser(str(it["a3m"]))
        if not os.path.exists(it["a3m"]):
            raise SystemExit(f"A3M not found: {it['a3m']}")
        # Optional per-item prefix; fall back to template
        it.setdefault("prefix", None)

    return cfg

def main():
    # Read config
    cfg = get_config()
    # Initialize model runner
    runner = make_model_runner(cfg["model_dir"], cfg["n_recycle"])
    ccd = chemical_components.Ccd()
    # Process each item
    for item in cfg["items"]:
        a3m_path = os.path.expanduser(item["a3m"])
        Ls = item["Ls"]
        stem = os.path.splitext(os.path.basename(a3m_path))[0]
        prefix = item["prefix"] if item["prefix"] is not None else cfg["prefix_template"].format(stem=stem)
        # Parse paired A3M and split into unpaired MSAs
        entries = parse_paired_a3m(a3m_path)
        a3m_strings = split_msa_to_a3m_strings(entries, Ls)
        # Extract sequences for each chain
        query_seqs = [entries[0][1][sum(Ls[:i]):sum(Ls[:i+1])] for i in range(len(Ls))]
        print(f"Processing {prefix}: chains {len(Ls)}, lengths {Ls}")
        chain_ids = ["A", "B"][:len(Ls)]
        name = prefix
        seed = 0
        # Build input features
        finput = build_fold_input(query_seqs, a3m_strings, chain_ids, name, seed)
        features = featurisation.featurise_input(
            fold_input=finput,
            ccd=ccd,
            buckets=None,
            max_template_date=datetime.date(2025, 1, 1),
            conformer_max_iterations=None,
            verbose=False
        )[0]
        # Run model
        batch, result = runner.run(features, seed=seed)
        # Process distogram to contact probabilities
        disto_logits = result["distogram"]["distogram"]
        disto_prob = scipy.special.softmax(disto_logits, axis=-1)

        inference_result = list(model.Model.get_inference_result(
            batch=features,
            result=result,
            target_name=name
        ))[0]
        token_chain_ids = np.asarray(inference_result.metadata["token_chain_ids"])

        contact, idxA, idxB = extract_inter_contact(token_chain_ids, disto_prob, Ls)
        # Save contact map
        np.savez_compressed(
            os.path.join(cfg["outdir"], f"{prefix}_contact.npz"),
            contact_prob=contact.astype(np.float16)
        )
        # Cleanup
        del contact, idxA, idxB, token_chain_ids, disto_prob, disto_logits
        del result, batch, features, finput, a3m_strings, entries, query_seqs
        gc.collect()
        jax.clear_caches()
        print(f"Saved contact map for {prefix} → {prefix}_contact.npz")


if __name__ == "__main__":
    main()