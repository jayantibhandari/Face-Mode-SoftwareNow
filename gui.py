import tkinter as tk
from tkinter import filedialog
from PIL import Image
import torch
from transformers import pipeline, AutoProcessor, AutoModelForVisualQuestionAnswering

# ----------------------------
# GUI CLASS
# ----------------------------
class ImageToTextGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Image Question Answering")
        self.geometry("600x500")
        self.configure(bg="black")

        # Path to selected image
        self.image_path = None

        # MODELS (lazy-loaded)
        self.image_to_text_model = None
        self.vqa_model = None
        self.vqa_processor = None

        # ----------------------------
        # WIDGETS
        # ----------------------------
        # Image path label
        self.label_image = tk.Label(
            self,
            text="Selected Image:",
            fg="white",
            bg="black"
        )
        self.label_image.pack(pady=5)

        # Show selected file path
        self.selected_image_label = tk.Label(
            self,
            text="No image selected",
            fg="white",
            bg="black",
            wraplength=400
        )
        self.selected_image_label.pack(pady=5)

        # Browse button
        self.btn_browse = tk.Button(
            self,
            text="Browse Image",
            command=self.browse_image
        )
        self.btn_browse.pack(pady=5)

        # Question entry
        self.label_question = tk.Label(
            self,
            text="Question:",
            fg="white",
            bg="black"
        )
        self.label_question.pack(pady=5)

        self.entry_question = tk.Entry(self, width=60)
        self.entry_question.pack(pady=5)

        # Buttons for models
        self.btn_image_to_text = tk.Button(
            self,
            text="Image-to-Text Model",
            command=self.run_image_to_text
        )
        self.btn_image_to_text.pack(pady=5)

        self.btn_vqa = tk.Button(
            self,
            text="VQA Model",
            command=self.run_vqa
        )
        self.btn_vqa.pack(pady=5)

        # Output label
        self.output_label = tk.Label(
            self,
            text="Please wait!! Answer will appear here",
            fg="white",
            bg="black",
            wraplength=500,
            justify="left"
        )
        self.output_label.pack(pady=20)

    # ----------------------------
    # FUNCTIONS
    # ----------------------------
    def browse_image(self):
        """Browse a image file."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.image_path = file_path
            self.selected_image_label.config(text=file_path)

    def run_image_to_text(self):
        """Run the Image-to-Text model."""
        if not self.image_path:
            self.output_label.sconfig(text="Please select an image first.")
            return

        question = self.entry_question.get().strip()
        if not question:
            question = "Describe the image."

        # Lazy-load model
        if self.image_to_text_model is None:
            self.output_label.config(text="Loading Image-to-Text model, please wait...")
            self.update()
            device = 0 if torch.backends.mps.is_available() else -1
            self.image_to_text_model = pipeline(
                "image-text-to-text",
                model="smolagents/SmolVLM2-2.2B-Instruct-Agentic-GUI",
                device=device
            )

        # Run prediction
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.image_path},
                    {"type": "text", "text": question}
                ]
            }
        ]

        result = self.image_to_text_model(text=messages)
        answer = result[0]['generated_text'][-1]['content'] if isinstance(result[0]['generated_text'], list) else result[0]['generated_text']
        self.output_label.config(text=f"Image-to-Text Answer:\n{answer}")

    def run_vqa(self):
        """Run the Visual Question Answering model."""
        if not self.image_path:
            self.output_label.config(text="Please select an image first.")
            return

        question = self.entry_question.get().strip()
        if not question:
            question = "What is in the image?"

        # Lazy-load VQA model
        if self.vqa_model is None or self.vqa_processor is None:
            self.output_label.config(text="Loading VQA model, please wait...")
            self.update()
            device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
            self.vqa_processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-small")
            self.vqa_model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip-vqa-small").to(device)
            self.device = device

        # Preprocess input
        image = Image.open(self.image_path).convert("RGB")
        inputs = self.vqa_processor(images=image, text=question, return_tensors="pt").to(self.device)

        # Generate output
        with torch.no_grad():
            out = self.vqa_model.generate(**inputs)
        answer = self.vqa_processor.decode(out[0], skip_special_tokens=True)

        self.output_label.config(text=f"VQA Answer:\n{answer}")


# ----------------------------
# MAIN file to run
# ----------------------------
if __name__ == "__main__":
    app = ImageToTextGUI()
    app.mainloop()
