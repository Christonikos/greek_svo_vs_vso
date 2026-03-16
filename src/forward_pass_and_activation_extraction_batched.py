from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_greek_sentences(csv_path="../stimuli/greek_sentences.csv"):
    """
    Load Greek sentences from CSV file.

    Args:
        csv_path (str): Path to the CSV file containing Greek sentences

    Returns:
        pd.DataFrame: DataFrame containing the loaded sentences
    """
    try:
        # Convert to absolute path for better error handling
        abs_path = Path(__file__).parent / csv_path

        # Check if file exists
        if not abs_path.exists():
            raise FileNotFoundError(f"CSV file not found at: {abs_path}")

        # Load the CSV file
        df = pd.read_csv(abs_path)

        print(f"Successfully loaded {len(df)} sentences from {csv_path}")
        print(f"Columns: {list(df.columns)}")

        return df

    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None


def load_model_and_tokenizer(
    model_name: str = "ilsp/Llama-Krikri-8B-Base",
    optimize_memory: bool = False,
    device: str = None,
):
    """Load the HuggingFace model & tokenizer and move to appropriate device."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # bf16 if supported, else fp16
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    print(f"Loading model '{model_name}' on device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if optimize_memory:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
        )
        model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
            dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        )

    model.eval()

    return model, tokenizer, device, dtype


def batched_aggregations(
    sentences, model, tokenizer, device, dtype, max_length=None, optimize_memory=True
):
    enc = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True if max_length else False,
        max_length=max_length,
    ).to(device)

    with torch.inference_mode():
        ctx = torch.autocast("cuda", dtype=dtype) if device == "cuda" else nullcontext()
        with ctx:
            out = model(
                **enc, output_hidden_states=True, use_cache=False, return_dict=True
            )

    hs = out.hidden_states  # tuple: (L+1) each (B, T, H)
    attn = enc["attention_mask"]  # (B, T)
    lengths = attn.sum(dim=1)  # (B,)
    last_idx = (lengths - 1).clamp(min=0)

    results = []
    for b in range(attn.size(0)):
        Lb = int(lengths[b].item())
        ids = enc["input_ids"][b, :Lb]
        tokens = tokenizer.convert_ids_to_tokens(ids.to("cpu"))

        per_layer_last = []
        per_layer_mean = []
        per_layer_hs_cpu = []

        for layer_hs in hs:
            x = layer_hs[b, :Lb, :]  # (Lb, H) no pads

            per_layer_last.append(x[last_idx[b]].detach().to("cpu").float())

            mean_vec = x.mean(dim=0)
            per_layer_mean.append(mean_vec.detach().to("cpu").float())

            if optimize_memory:
                per_layer_hs_cpu.append(x.detach().to("cpu").to(torch.float16))
            else:
                per_layer_hs_cpu.append(x.detach().to("cpu").to(torch.float32))

        results.append(
            {
                "tokens": tokens,
                "hidden_states": per_layer_hs_cpu,  # list length L+1, each (Lb, H)
                "last_token": per_layer_last,
                "mean": per_layer_mean,
            }
        )

    return results


def extract_activations_for_sentence(
    sentence: str, model, tokenizer, optimize_memory: bool, device: str = "cuda"
):
    """Return hidden states for the sentence."""
    # Tokenize
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    # outputs.hidden_states -> tuple(length = num_layers + 1)
    hidden_states = outputs.hidden_states  # each tensor: (batch, seq_len, hidden_size)

    for li, hs in enumerate(hidden_states):
        if not torch.isfinite(hs).all():
            print(f"[WARNING] Non-finite values already in hidden_states at layer {li}")

    if optimize_memory:
        # Move to CPU and half precision to save memory
        hidden_states_cpu = [hs.squeeze(0).to("cpu").half() for hs in hidden_states]
    else:
        hidden_states_cpu = [
            hs.squeeze(0).detach().to("cpu", dtype=torch.float32)
            for hs in hidden_states
        ]

    # Tokens list for reference
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0).to("cpu"))

    # Aggregations for quick inspection
    last_token_vec = [hs[-1] for hs in hidden_states_cpu]  # (hidden_size,)
    mean_vec = [hs.mean(dim=0) for hs in hidden_states_cpu]  # (hidden_size,)

    return {
        "sentence": sentence,
        "tokens": tokens,
        "hidden_states": hidden_states_cpu,
        "last_token": last_token_vec,
        "mean": mean_vec,
    }


def process_and_save_activations(
    df,
    model,
    tokenizer,
    device="cuda",
    dtype=torch.float32,
    save_dir="activations",
    optimize_memory=True,
    batch_size=8,
    max_length=None,
):
    """Batch over DataFrame rows, extract activations and save to disk (one file per sentence)."""
    save_dir_path = Path(save_dir)
    # append parent folder if the user only gave a folder name, otherwise use the absolute path
    save_path = (
        save_dir_path
        if save_dir_path.is_absolute()
        else Path(__file__).parent / save_dir
    )
    save_path.mkdir(exist_ok=True)

    # faster than iterrows, and handle NaNs
    sentences = df["Sentence"].fillna("").astype(str).tolist()

    for start in tqdm(range(0, len(sentences), batch_size), desc="Processing batches"):
        batch = sentences[start : start + batch_size]

        batch_out = batched_aggregations(
            batch,
            model,
            tokenizer,
            device=device,
            dtype=dtype,
            max_length=max_length,
            optimize_memory=optimize_memory,
        )

        for j, feats in enumerate(batch_out):
            idx = start + j
            activations = {
                "sentence": batch[j],
                "tokens": feats["tokens"],
                "hidden_states": feats["hidden_states"],  # <-- restored
                "last_token": feats["last_token"],
                "mean": feats["mean"],
            }
            torch.save(activations, save_path / f"sentence_{idx}.pt")


# def process_and_save_activations(
#     df, model, tokenizer, device="cuda", save_dir="activations", optimize_memory=True
# ):
#     """Iterate over DataFrame rows, extract activations and save to disk."""
#     save_path = Path(__file__).parent / save_dir
#     save_path.mkdir(exist_ok=True)

#     for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing sentences"):
#         activations = extract_activations_for_sentence(
#             row["Sentence"], model, tokenizer, optimize_memory, device
#         )
#         # Save using torch.save for fidelity & compression
#         torch.save(activations, save_path / f"sentence_{idx}.pt")
#         # Free GPU memory if needed
#         # torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract hidden activations for Greek sentences."
    )
    parser.add_argument(
        "--model", default="ilsp/Llama-Krikri-8B-Base", help="HF model name"
    )
    parser.add_argument(
        "--csv",
        default="../stimuli/greek_sentences.csv",
        help="Path to stimuli CSV",
    )
    parser.add_argument(
        "--save_dir",
        default="activations",
        help="Directory to save activations",
    )
    parser.add_argument(
        "--optimize-memory",
        action="store_true",
        help="Use fp16 model + fp16 activation storage on CPU to save on mem (default: False).",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--max_length", type=int, default=None, help="Truncate very long inputs"
    )
    args = parser.parse_args()

    df = load_greek_sentences(args.csv)

    if df is None:
        raise RuntimeError("Failed to load sentences CSV.")

    model, tokenizer, device, dtype = load_model_and_tokenizer(
        args.model, args.optimize_memory
    )
    process_and_save_activations(
        df,
        model,
        tokenizer,
        device=device,
        dtype=dtype,
        save_dir=args.save_dir,
        optimize_memory=args.optimize_memory,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print("\nActivation extraction complete.")
