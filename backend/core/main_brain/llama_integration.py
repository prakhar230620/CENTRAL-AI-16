import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from backend.core.main_brain.llama_output_analyzer import EnhancedLlamaOutputAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLamaBrain:
    def __init__(self, model_name="meta-llama/Meta-Llama-3.1-8B"):
        self.model_name = model_name
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.info(f"Added new pad_token: {self.tokenizer.pad_token}")

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.context = []
        self.analyzer = EnhancedLlamaOutputAnalyzer()
        logger.info("Model and analyzer loaded successfully")

    async def process_input(self, text):
        logger.info(f"Processing input: {text}")
        inputs = self.tokenizer.encode_plus(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        input_length = inputs['input_ids'].shape[1]
        max_new_tokens = 1000
        max_length = input_length + max_new_tokens

        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1
        )
        raw_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Analyze the output using the EnhancedLlamaOutputAnalyzer
        is_relevant, confidence, filtered_response = self.analyzer.analyze_output(text, raw_response)

        if is_relevant:
            logger.info(f"Generated relevant response (confidence: {confidence:.2f})")
            self.context.append((text, filtered_response))
            return filtered_response
        else:
            logger.info(f"Generated response not relevant (confidence: {confidence:.2f}). Routing to other module.")
            self.analyzer.route_to_other_module(text)
            return None

    def clear_context(self):
        self.context = []
        logger.info("Cleared the conversation context.")


# Example usage
async def main():
    llama_brain = LLamaBrain()
    response = await llama_brain.process_input("What is the capital of France?")
    if response:
        logger.info(f"LLama response: {response}")
    else:
        logger.info("No relevant response generated. Input routed to another module.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())