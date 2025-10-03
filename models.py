# models.py
from transformers import pipeline, AutoProcessor, AutoModelForVisualQuestionAnswering
from PIL import Image
import torch

# --------------------------
# Text-to-Image Model
# --------------------------
class TextToImageModel:
    def __init__(self):
        # Load the model lazily when needed
        self.pipe = pipeline(
            "image-text-to-text",
            model="smolagents/SmolVLM2-2.2B-Instruct-Agentic-GUI"
        )

    def predict(self, image_path, question):
        # Hugging Face pipeline accepts {"type": "image", "image": <PIL.Image>}
        image = Image.open(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]
        output = self.pipe(messages)
        # Extract generated text safely
        try:
            # Adjust indexing if output format changes
            answer = output[0]['generated_text'][1]['content']
        except Exception:
            answer = str(output)
        return answer


# --------------------------
# Visual Question Answering Model
# --------------------------
class ImageClassifierModel:
    def __init__(self):
        # Use smaller VQA model for faster loading
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-small")
        self.model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip-vqa-small")

    def predict(self, image_path, question):
        image = Image.open(image_path)
        inputs = self.processor(images=image, text=question, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        answer = self.processor.decode(outputs.logits.argmax(-1))
        return answer

