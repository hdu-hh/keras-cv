"""
Title: Generate an image from a text prompt using StableDiffusion
Author: fchollet
Date created: 2022/09/24
Last modified: 2022/09/24
Description: Use StableDiffusion to generate an image according to a short text description.
"""

from PIL import Image

from keras_cv.models import StableDiffusionV2

model = StableDiffusionV2(img_height=768, img_width=768, jit_compile=True)
img = model.text_to_image("Photograph of a beautiful horse running through a field")
Image.fromarray(img[0]).save("horse.png")
print("Saved at horse.png")

img = model.text_to_image("a zebra", inpImage=img)
Image.fromarray(img[0]).save("zebra.png")
print("Saved zebra-morphed horse as zebra.png")
