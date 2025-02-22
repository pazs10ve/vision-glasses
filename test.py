import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import matplotlib.pyplot as plt
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

cap = cv2.VideoCapture(0)  

ret, frame = cap.read()
cap.release() 

if not ret:
    raise Exception("Could not capture image from camera.")

image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
image = Image.fromarray(image)

prompt = "<CAPTION>"
inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    num_beams=3,
    do_sample=False
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
parsed_answer = processor.post_process_generation(generated_text, task="<CAPTION>", image_size=(image.width, image.height))

plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')  

plt.text(10, 10, parsed_answer['<CAPTION>'], fontsize=12, color='red', alpha=0.7)

plt.show()
