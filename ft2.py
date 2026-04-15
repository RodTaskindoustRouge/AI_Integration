import os
from google import genai
from google.genai import types
from PIL import Image

# 1. Setup Client
# Ensure GEMINI_API_KEY is set in your environment variables
client = genai.Client()

# 2. Prepare Images
# Replace these paths with your actual local files
input_image_path = "jeans.jpg"     # The object you want to restyle
style_image_path = "fabric1.jpg"   # The fabric sample
output_filename = "restyle_result.png"

input_image = Image.open(input_image_path)
style_image = Image.open(style_image_path)

# 3. Define the Prompt
# This combines your intent into the text_input variable
target_object = "the jeans"
text_input = f"""
Create a professional fashion product photo. 
Take the {target_object} from the first image and apply the exact fabric texture, 
color, and weave pattern from the second image onto it. 

Requirements:
- The {target_object} must maintain its original shape, folds, and silhouette.
- Match the lighting and shadows of the original scene for a photorealistic look.
- Keep the background exactly as it is in the first image.
- High-resolution, 8k, sharp focus on fabric detail.
"""

# 4. Generate Content
# We pass the images and the text together in the contents list
response = client.models.generate_content(
    model="gemini-2.0-flash",
    #model="gemini-3.1-flash-image-preview",
    contents=[input_image, style_image, text_input],
)

# 5. Process and Save the Result
for part in response.parts:
    if part.text is not None:
        # This handles any textual reasoning or descriptions the AI provides
        print(f"AI Feedback: {part.text}")
    
    if part.inline_data is not None:
        # This extracts the generated image
        try:
            image = part.as_image()
            image.save(output_filename)
            print(f"✓ Success! Saved to {output_filename}")
        except Exception as e:
            print(f"Error saving image: {e}")