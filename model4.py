from transformers import pipeline

# Use BLIP image captioning model (small and open access)
pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

# Input image
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

# Run pipeline
result = pipe(image_url)

print(result)
