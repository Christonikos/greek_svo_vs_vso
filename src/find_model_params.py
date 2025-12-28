import argparse
from transformers import AutoConfig


parser = argparse.ArgumentParser(
    description="Run probing / evaluation on a Hugging Face model."
)

parser.add_argument(
    "--model-id",
    type=str,
    default="google/gemma-3-12b-pt",
    help='Hugging Face model repo id (e.g., "google/gemma-3-12b-pt", "ilsp/Llama-Krikri-8B-Base").',
)

args = parser.parse_args()

cfg = AutoConfig.from_pretrained(args.model_id)

num_hidden_layers = getattr(cfg, "num_hidden_layers", None)
if num_hidden_layers:                   
    print("num_hidden_layers:", getattr(cfg, "num_hidden_layers", None))
else:
    print(cfg)
