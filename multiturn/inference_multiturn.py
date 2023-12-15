import sys
import argparse
import torch
from transformers import  AutoModelForCausalLM, AutoTokenizer
from multiturn_utils import generate_prompt_with_history, sample_decode

device = "cuda" if torch.cuda.is_available() else "cpu"


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16, trust_remote_code=True)

    model.eval().to(device)
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)
    return model, tokenizer


def evaluate(args, model, tokenizer):
    history = []
    while True:
        try:
            input_ids, text = generate_prompt_with_history(tokenizer, device, history,
                                                           args.max_context_length_tokens)
            with torch.no_grad():
                for x in sample_decode(input_ids,
                                       model,
                                       tokenizer,
                                       stop_words=["[|Human|]", "[|AI|]", "</s>"],
                                       max_length=args.max_new_tokens,
                                       temperature=args.temperature,
                                       top_p=args.top_p, ):
                    pass
                    # if is_stop_word_or_prefix(x, ["[|Human|]", "[|AI|]"]) is False:
                    #     print(x)
                if "[|Human|]" in x:
                    x = x[: x.index("[|Human|]")].strip()
                if "[|AI|]" in x:
                    x = x[: x.index("[|AI|]")].strip()
                x = x.strip(" ")
                history.append([text, x])
                print('TripGPT: ' + x)
        except KeyboardInterrupt:
            break


def set_args():
    """
    Sets up the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='', help='base model to use')

    parser.add_argument('--temperature', default=0.8, type=float, required=False,
                        help='The value used to modulate the next token probabilities.')
    parser.add_argument('--top_k', default=8, type=int, required=False,
                        help='The number of highest probability vocabulary tokens to keep for top-k-filtering.')
    parser.add_argument('--top_p', default=0.85, type=float, required=False,
                        help='If set to float < 1, only the smallest set of most probable tokens with probabilities '
                             'that add up to top_p or higher are kept for generation.')
    parser.add_argument("--num_beams", type=int, default=1,
                        help='Number of beams for beam search. 1 means no beam search.')
    parser.add_argument('--repetition_penalty', default=1.2, type=float, required=False,
                        help="The parameter for repetition penalty. 1.0 means no penalty.")
    parser.add_argument('--min_new_tokens', type=int, default=128,
                        help='The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt.')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                        help='The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.')
    parser.add_argument('--max_context_length_tokens', type=int, default=2048,
                        help='The maximum numbers of history tokens.')
    return parser.parse_args()


if __name__ == "__main__":
    args = set_args()
    model, tokenizer = init_model(args)
    evaluate(args, model, tokenizer)