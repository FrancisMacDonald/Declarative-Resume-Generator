import torch
import transformers

from transformers import LlamaForCausalLM, LlamaTokenizer
class LlamaPlugin:
    def __init__(self, path_to_model):
        self.path_to_model = path_to_model
        self.pipeline, self.tokenizer = self.init_llama3(path_to_model)

    def init_llama(self, path_to_model):
        # load the model
        model = LlamaForCausalLM.from_pretrained(path_to_model)
        tokenizer = LlamaTokenizer.from_pretrained(path_to_model)

        # initialize the text-gen pipeline
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        return pipeline, tokenizer

    def extract_skills_from_experience(self, experience, prompt):
        # TOTO: implement this method
        raise NotImplementedError("The method extract_skills_from_experience is not implemented yet.")

        sequences = self.pipeline(
            prompt + '\n',
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=400,
        )

        for seq in sequences:
            print(f"{seq['generated_text']}")
