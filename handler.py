import llama
from typing import Dict, List, Any

MODEL = 'decapoda-research/llama-7b-hf'


# MODEL = 'decapoda-research/llama-13b-hf'
# MODEL = 'decapoda-research/llama-30b-hf'
# MODEL = 'decapoda-research/llama-65b-hf'

class EndpointHandler():
    def __init__(self, path=""):
        self.tokenizer = llama.LLaMATokenizer.from_pretrained(MODEL)
        self.model = llama.LLaMAForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, load_in_8bit=True)
        self.model.to('cuda')

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        inputs = data.pop("inputs", data)
        parameters = data.pop("parameters", None)

        input_ids = self.tokenizer(inputs, return_tensors="pt", add_special_tokens=False).input_ids.cuda()

        if parameters is not None:
            outputs = self.model.generate(input_ids, **parameters)
        else:
            outputs = self.model.generate(input_ids)

        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return [{"generated_text": prediction}]
