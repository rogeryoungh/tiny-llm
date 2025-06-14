import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(
        description="Load a local Transformer model and run inference, printing tokens and outputs for comparison."
    )
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Path to the local model directory (with config.json, pytorch_model.bin, etc.)"
    )
    parser.add_argument(
        "--prompt", type=str, default="Hello, world!",
        help="Text prompt to run through the model"
    )
    parser.add_argument(
        "--max_length", type=int, default=50,
        help="Maximum number of tokens to generate"
    )
    args = parser.parse_args()

    # Load tokenizer and model
    print(f"Loading tokenizer from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    print(f"Loading model from {args.model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    model.eval()

    # Tokenize input
    inputs = tokenizer(
        args.prompt, return_tensors="pt"
    )
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # Print input tokens
    token_ids = input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    print("Input token IDs:", token_ids)
    print("Input tokens:     ", tokens)

    # Run inference (generate)
    with torch.no_grad():
        print("Generating output...")
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=args.max_length,
            do_sample=False,
            num_beams=1,
        )

    # Print full sequence (prompt + generation)
    out_ids = output_ids[0].tolist()
    out_tokens = tokenizer.convert_ids_to_tokens(out_ids)
    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)

    print("\n=== Generation Result ===")
    print("Output token IDs:", out_ids)
    print("Output tokens:     ", out_tokens)
    print("Decoded text:      ", out_text)

    # Print logits for each generated token (optional)
    try:
        print("\nComputing token logits for each position...")
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits  # shape [1, seq_len, vocab_size]
        # For each position, print the top-5 token logits
        for i, logit_vec in enumerate(logits[0]):
            topk = torch.topk(logit_vec, k=5)
            top_ids = topk.indices.tolist()
            top_scores = topk.values.tolist()
            top_tokens = tokenizer.convert_ids_to_tokens(top_ids)
            print(f"Position {i}: token '{tokens[i]}' -> top5 tokens: {list(zip(top_tokens, top_scores))}")
    except Exception as e:
        print(f"Could not compute logits: {e}")

if __name__ == "__main__":
    main()
