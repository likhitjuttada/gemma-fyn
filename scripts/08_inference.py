import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

load_dotenv()

token = os.getenv('HF_TOKEN')
base_model_id = 'unsloth/gemma-4-E2B-it'
adapter_path = 'gemma-lora'

tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=token)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
    device_map=device,
    token=token,
)
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()


def ask(question: str) -> str:
    messages = [{'role': 'user', 'content': question}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors='pt',
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
        )
    new_tokens = output_ids[0][inputs.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


if __name__ == '__main__':
    import sys
    question = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else 'Should I invest in index funds or individual stocks?'
    print(ask(question))
