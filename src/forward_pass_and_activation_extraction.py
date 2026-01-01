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
    model_name: str = "ilsp/Llama-Krikri-8B-Base", device: str = None
):
    """Load the HuggingFace model & tokenizer and move to appropriate device."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model '{model_name}' on device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    model.to(device)
    model.eval()

    return model, tokenizer, device


def extract_activations_for_sentence(
    sentence: str, model, tokenizer, device: str = "cuda", precision: str = "float32"
):
    """Return hidden states for the sentence.
    
    Args:
        sentence: Input sentence
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        device: Device to run on
        precision: Storage precision - "float16" (smaller, may overflow) or "float32" (safer)
                   NOTE: Gemma models require float32 due to large activation values
    """
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

    # Move to CPU with specified precision
    # WARNING: float16 can overflow for models with large activations (e.g., Gemma)
    if precision == "float16":
        hidden_states_cpu = [hs.squeeze(0).to("cpu").half() for hs in hidden_states]
    else:  # float32 (default, safer)
        hidden_states_cpu = [hs.squeeze(0).to("cpu").float() for hs in hidden_states]

    # Tokens list for reference
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))

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
    df, model, tokenizer, device="cuda", save_dir="activations", precision="float32"
):
    """Iterate over DataFrame rows, extract activations and save to disk.
    
    Args:
        df: DataFrame with sentences
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        device: Device to run on
        save_dir: Directory to save activations
        precision: Storage precision - "float16" or "float32" (default, recommended)
    """
    save_path = Path(__file__).parent / save_dir
    save_path.mkdir(exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing sentences"):
        activations = extract_activations_for_sentence(
            row["Sentence"], model, tokenizer, device, precision=precision
        )
        # Save using torch.save for fidelity & compression
        torch.save(activations, save_path / f"sentence_{idx}.pt")
        # Free GPU memory if needed
        torch.cuda.empty_cache()


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
        "--precision",
        default="float32",
        choices=["float16", "float32"],
        help="Storage precision. Use float32 (default) for Gemma and other models with large activations. float16 saves space but can overflow.",
    )
    args = parser.parse_args()

    df = load_greek_sentences(args.csv)

    if df is None:
        raise RuntimeError("Failed to load sentences CSV.")

    model, tokenizer, device = load_model_and_tokenizer(args.model)
    
    print(f"\nUsing precision: {args.precision}")
    if args.precision == "float16":
        print("WARNING: float16 can overflow for models with large activations (e.g., Gemma)")
    
    process_and_save_activations(
        df, model, tokenizer, device=device, save_dir=args.save_dir, precision=args.precision
    )

    print("\nActivation extraction complete.")
