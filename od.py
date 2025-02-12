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

if not cap.isOpened():
    raise Exception("Could not open video device.")

plt.ion()  
fig, ax = plt.subplots(figsize=(10, 10))

while True:
    ret, frame = cap.read() 
    if not ret:
        break 

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    prompt = "<OD>"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))

    ax.clear()
    ax.imshow(image)
    ax.axis('off')

    # Iterate through bboxes and labels
    for bboxes, label in zip(parsed_answer['<OD>']['bboxes'], parsed_answer['<OD>']['labels']):
        x1, y1, x2, y2 = bboxes
        
        ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                     fill=False, edgecolor='red', linewidth=2))
        
        ax.text(x1, y1-10, label, fontsize=10, 
                color='red', backgroundcolor='white')

    plt.pause(0.01)  

cap.release() 
plt.ioff() 
plt.show()