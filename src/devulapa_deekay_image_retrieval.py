"""
The image retreival section of the project takes a text prompt as input and fetches images from dataset that are semantically close the text prompt.
There are two versions of our code for image_retrieval.
1. The generic python file that has to be run in command line.
2. Gradio version that is deployed providing interface to upload images in the Gradio interface.
"""
# Authors: DeeKay Goswami & Naresh Kumar Devulapally

# Generic Python version

import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

text = input("Please enter your search query: ")
number = int(input("Enter no. of required images: "))
text = [text]
print("Fetching your data, please wait...")
text_inputs = processor(text=text, return_tensors = "pt", padding = True, truncation = True)

text_outputs = model.get_text_features(**text_inputs)
text_features = text_outputs.detach()

# This will set image paths...
image_paths = glob.glob("/Users/deekay93/Desktop/10_Classes/cars/*.JPEG") 

image_features = []

# This will generate image features...
for image_path in image_paths:
    try:
        image_input = Image.open(image_path).convert('RGB')
        image_input = processor(images=image_input, return_tensors = "pt", padding=True, truncation=True)
        outputs = model.get_image_features(**image_input)
        image_features.append(outputs.detach())
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")

# This will check if we have valid image features...
if image_features:
    # This will calculate similarity between text and image features...
    sim_scores = [cosine_similarity(text_features, img_feature)[0][0] for img_feature in image_features]

    # This will get indices of top 5 similar images...
    top_indices = sorted(range(len(sim_scores)), key = lambda i: sim_scores[i])[-number:]

    # This will get the top 5 most similar images...
    top_images = [image_paths[i] for i in top_indices]

    print("Most similar images:")
    plt.figure(figsize = (20,10))
    for i, img_path in enumerate(top_images):
        img = mpimg.imread(img_path)
        plt.subplot(1,number,i+1)
        plt.title(f"Rank {i+1}")
        plt.imshow(img)
        plt.axis("off")

    plt.show()
else:
    print("No valid image features found. Please check your dataset.")





    

# Gradio version
import glob
import numpy as np
import gradio as gr
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to concatenate images into a single numpy array
def concatenate_images(images):
    max_height = max(img.shape[0] for img in images)
    padded_images = [np.pad(img, ((0, max_height - img.shape[0]), (0, 0), (0, 0))) for img in images]
    concatenated_images = np.concatenate(padded_images, axis=1)
    return concatenated_images

def fetch_images(query, number, folder_path):
    text = [query]
    number = int(number)
    text_inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
    text_outputs = model.get_text_features(**text_inputs)
    text_features = text_outputs.detach()

    image_paths = glob.glob(folder_path + "/*.JPEG") 
    image_features = []

    for image_path in image_paths:
        try:
            image_input = Image.open(image_path).convert('RGB')
            image_input = processor(images=image_input, return_tensors="pt", padding=True, truncation=True)
            outputs = model.get_image_features(**image_input)
            image_features.append(outputs.detach())
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")

    if image_features:
        sim_scores = [cosine_similarity(text_features, img_feature)[0][0] for img_feature in image_features]
        top_indices = sorted(range(len(sim_scores)), key=lambda i: sim_scores[i])[-number:]
        top_images = [np.array(Image.open(image_paths[i])) for i in top_indices]
        return concatenate_images(top_images)
    else:
        print("No valid image features found. Please check your dataset.")
        return None

iface = gr.Interface(
    fn=fetch_images,
    inputs=[
        gr.inputs.Textbox(label="Search Query"),
        gr.inputs.Textbox(label="Number of Images"),
        gr.inputs.Textbox(label="Dataset Path")
    ],
        outputs=gr.outputs.Image(type='numpy', label="Most Similar Images"),
)

iface.launch(share = True)
