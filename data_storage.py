import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
from datetime import datetime, timedelta


def retrieve_data_by_timerange(csv_path, start_time, end_time):
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
    return df.loc[mask]


def display_retrieved_data(filtered_df, rows=3, cols=3):
    n_images = min(len(filtered_df), rows * cols)
    if n_images == 0:
        print("No data found for the specified time range.")
        return

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.ravel()

    for idx, (_, row) in enumerate(filtered_df.head(n_images).iterrows()):
        img = cv2.imread(row['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[idx].imshow(img)
        axes[idx].axis('off')
        title = f"Time: {row['timestamp']}\nCaption: {row['caption']}"
        axes[idx].set_title(title, fontsize=8, wrap=True)

    for idx in range(n_images, rows * cols):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()



def main():
    os.makedirs('captured_frames', exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

    df = pd.DataFrame(columns=['timestamp', 'image_path', 'caption'])

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception("Could not open video device.")

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = datetime.now()
            timestamp = current_time.strftime("%Y%m%d_%H%M%S_%f")
            
            image_filename = f"captured_frames/frame_{timestamp}.jpg"
            cv2.imwrite(image_filename, frame)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

            prompt = "<CAPTION>"
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=2048,
                num_beams=3,
                do_sample=True
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(generated_text, task="<CAPTION>", image_size=(image.width, image.height))
            caption = parsed_answer['<CAPTION>']

            new_row = pd.DataFrame({
                'timestamp': [current_time],
                'image_path': [image_filename],
                'caption': [caption]
            })
            df = pd.concat([df, new_row], ignore_index=True)

            display_frame = frame.copy()
            cv2.putText(display_frame, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 255), 2)
            #cv2.putText(display_frame, "Press 'q' to exit", (10, 60), 
             #          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Live Caption', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit key pressed. Stopping capture...")
                break

            frame_count += 1
            
            if frame_count % 10 == 0:
                df.to_csv('caption_log.csv', index=False)

    except KeyboardInterrupt:
        print("Stopping capture...")
    finally:
        df.to_csv('caption_log.csv', index=False)
        cap.release()
        cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames")
    print("Data saved to caption_log.csv")
    print(f"Images saved in captured_frames/ directory")
    print(f'Showing example retrieval')
    example_retrieval()


def example_retrieval():
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
    
    filtered_data = retrieve_data_by_timerange('caption_log.csv', start_time_str, end_time_str)
    
    display_retrieved_data(filtered_data)

if __name__ == "__main__":
    main()
