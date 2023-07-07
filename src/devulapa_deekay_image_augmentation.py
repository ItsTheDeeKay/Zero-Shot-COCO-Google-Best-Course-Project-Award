# This code file represents the first checkpoint submission for our final project.
# The code opens a gradio interface to perform multiple selected augmentations at the same time.
# The screenshots of the results are presented in the report and the presentation.
# Authors: DeeKay Goswami & Naresh Kumar Devulapally

import gradio as gr
import torchvision.transforms as transforms

# All transformation functions are mentioned here...
def apply_padding(image):
    return transforms.Pad(10)(image)

def apply_resize(image):
    return transforms.Resize((256, 256))(image)

def apply_center_crop(image):
    return transforms.CenterCrop(200)(image)

def apply_grayscale(image):
    return transforms.Grayscale()(image)

def apply_color_jitter(image):
    return transforms.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.5)(image)

def apply_gaussian_blur(image):
    return transforms.GaussianBlur(3)(image)

def apply_random_perspective(image):
    return transforms.RandomPerspective()(image)

def apply_random_rotation(image):
    return transforms.RandomRotation(45)(image)

def apply_random_affine_transform(image):
    return transforms.RandomAffine(degrees = 30, translate = (0.1, 0.1), scale = (0.8, 1.2))(image)

def apply_random_crop(image):
    return transforms.RandomCrop(200)(image)

def apply_random_resized_crop(image):
    return transforms.RandomResizedCrop(200)(image)

def apply_random_horizontal_flip(image):
    return transforms.RandomHorizontalFlip()(image)

def apply_random_vertical_flip(image):
    return transforms.RandomVerticalFlip()(image)

# This will define the augmentations...
augmentations = {
    'Image Padding': apply_padding,
    'Image Resize': apply_resize,
    'Center Crop': apply_center_crop,
    'Grayscale': apply_grayscale,
    'Random Color Jitter': apply_color_jitter,
    'Gaussian Blur': apply_gaussian_blur,
    'Random Perspective': apply_random_perspective,
    'Random Rotation': apply_random_rotation,
    'Random Affine Transform': apply_random_affine_transform,
    'Random Crop': apply_random_crop,
    'Random Resized Crop': apply_random_resized_crop,
    'Random Horizontal Flip': apply_random_horizontal_flip,
    'Random Vertical Flip': apply_random_vertical_flip
}

# This function is used to apply the augmentation to the inputted image...
def apply_augmentations(image, selected_augmentations):
    for augmentation_name in selected_augmentations:
        augmentation = augmentations[augmentation_name]
        image = augmentation(image)
    return image

# This will define the Gradio interface for augmentation...
def apply_augmentation_interface(image, selected_augmentations):
    augmented_image = apply_augmentations(image, selected_augmentations)
    return augmented_image

# This will upload the image and pass to one of the augmentation functions...
input_image = gr.inputs.Image(type = "pil", label = "Input Image")
augmentation_choices = gr.inputs.CheckboxGroup(list(augmentations.keys()), label = "Augmentation Choices")
output_image = gr.outputs.Image(type = "pil", label = "Augmented Image")

iface = gr.Interface(fn = apply_augmentation_interface, inputs = [input_image, augmentation_choices], outputs = output_image)
iface.launch(share = True)