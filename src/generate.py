import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from typing import List, Optional

class Generator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_name = model_name
        self.device = self._get_device()
        self.tokenizer = None
        self.model = None
        self.pipe = None

        print(f"Initializing generator with device: {self.device}")

    def _get_device(self):
        """Get the best available device, preferring MPS for Apple Silicon."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def load_model(self):
        """Load the model."""
        print(f"Loading {self.model_name}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto" if self.device not in ["mps", "cpu"] else None,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        # Move to appropriate device
        if self.device in ["mps", "cpu"]:
            self.model = self.model.to(self.device)

        # Create pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto" if self.device != "mps" else None
        )

        print("Model loaded successfully")

    def create_prompt(self, context: str, question: str) -> str:
        """Create a formatted prompt for the model."""
        prompt_template = """<|system|>
You are a helpful assistant answering questions about quantitative finance research papers.

Context from papers:
{context}

Question: {question}

Provide a clear, accurate answer based only on the context above. Cite which paper(s) you drew from.
<|user|>
{question}
<|assistant|>
"""

        return prompt_template.format(context=context, question=question)

    def generate_answer(self, context: str, question: str, max_length: int = 512) -> str:
        """Generate an answer given context and question."""
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        prompt = self.create_prompt(context, question)

        # Generate response
        try:
            outputs = self.pipe(
                prompt,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )

            answer = outputs[0]["generated_text"].strip()

            # Clean up the answer if it contains template artifacts
            if "<|assistant|>" in answer:
                answer = answer.split("<|assistant|>")[-1].strip()

            return answer

        except Exception as e:
            print(f"Error generating answer: {e}")
            return "I apologize, but I encountered an error while generating the answer."

class SimpleGenerator:
    """Fallback generator for environments where the full model doesn't work."""

    def __init__(self):
        self.loaded = True

    def load_model(self):
        pass

    def generate_answer(self, context: str, question: str, max_length: int = 512) -> str:
        """Simple template-based response for testing."""
        return f"""Based on the provided context, I can see information related to your question: "{question}"

The relevant papers discuss various aspects that may address your query. However, I'm currently running in fallback mode.

Sources mentioned in the context include the papers referenced above.

For a more detailed analysis, please ensure the full language model is properly loaded."""

if __name__ == "__main__":
    import sys

    # Simple test
    if len(sys.argv) < 3:
        print("Usage: python generate.py <context> <question>")
        sys.exit(1)

    context = sys.argv[1]
    question = sys.argv[2]

    # Try to load the full generator
    try:
        generator = Generator()
        generator.load_model()
        print("Using full generator")
    except Exception as e:
        print(f"Could not load full generator: {e}")
        print("Using simple generator")
        generator = SimpleGenerator()
        generator.load_model()

    # Generate answer
    answer = generator.generate_answer(context, question)

    print("\nQuestion:", question)
    print("\nAnswer:")
    print("-" * 40)
    print(answer)