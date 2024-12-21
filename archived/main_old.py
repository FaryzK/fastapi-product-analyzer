# Comments: Price extraction with EasyOCR doesn't work well since we get [DEBUG] Cleaned OCR Text: 99 31 096 27 80. 100 SCREWS atondos LAG
# Image classifier works fine
# OCR doesnt work well
# LLM works fine if the above extract the right features

import os
import re
import torch
from torchvision import models, transforms
from PIL import Image
from openai import OpenAI  # Updated import for OpenAI client
import easyocr
from dotenv import load_dotenv

# -------------------------------------------------------
# 1. SET UP OPENAI CLIENT
# -------------------------------------------------------
# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")  # API key from environment variable
)

# -------------------------------------------------------
# 2. IMAGE CLASSIFICATION LOADER (DEMO MODEL)
# -------------------------------------------------------
def classify_product(image_path: str) -> str:
    """
    Classify the hardware product in the image using a pretrained model.
    """
    print("[DEBUG] Loading pretrained ResNet model...")
    model = models.resnet50(pretrained=True)
    model.eval()  # Set the model to evaluation mode

    # Define preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to model's input size
        transforms.ToTensor(),         # Convert image to tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )  # Normalize to match ImageNet stats
    ])

    print("[DEBUG] Preprocessing image...")
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)  # Get the class index

    # Map class index to human-readable label (ImageNet classes)
    print("[DEBUG] Loading ImageNet class labels...")
    imagenet_classes = load_imagenet_classes()
    predicted_label = imagenet_classes[predicted_class.item()]

    print(f"[DEBUG] Predicted label: {predicted_label}")
    return predicted_label

def load_imagenet_classes() -> list:
    """
    Load ImageNet class labels for ResNet.
    """
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    import requests
    response = requests.get(url)
    return response.json()

# -------------------------------------------------------
# 3. OCR FUNCTION USING EASYOCR
# -------------------------------------------------------
def extract_text_with_easyocr(image_path: str) -> str:
    """
    Extract text from an image using EasyOCR.
    """
    print("[DEBUG] Initializing EasyOCR Reader...")
    reader = easyocr.Reader(['en'])  # Specify the language (e.g., English)

    print("[DEBUG] Performing OCR on the image...")
    results = reader.readtext(image_path)

    # Combine detected text into a single string
    extracted_text = " ".join([res[1] for res in results])
    print(f"[DEBUG] Raw OCR Text: {extracted_text}")
    return extracted_text

# -------------------------------------------------------
# 4. EXTRACT POTENTIAL PRICES
# -------------------------------------------------------
def extract_potential_prices(ocr_text: str) -> list:
    """
    Extract potential price-related information from OCR text.
    """
    print("[DEBUG] Extracting potential prices from OCR text...")
    # Match patterns for prices
    price_patterns = [
        r"\$\d+\.\d{2}",  # Match "$31.99"
        r"\d+\.\d{2}",    # Match "31.99"
        r"\d+c"           # Match "59c"
    ]
    
    # Extract matches from OCR text
    potential_prices = []
    for pattern in price_patterns:
        matches = re.findall(pattern, ocr_text)
        potential_prices.extend(matches)

    print(f"[DEBUG] Potential prices found: {potential_prices}")
    return potential_prices


# -------------------------------------------------------
# 5. BUILD PROMPT AND CALL LLM
# -------------------------------------------------------
def generate_product_info(product_label: str, ocr_text: str, potential_prices: list) -> str:
    """
    Use OpenAI's LLM to generate product information based on OCR text and potential prices.
    """
    # Prepare the prompt
    prompt = f"""
    Analyze the following information extracted from a product image:

    Product Label: {product_label}
    OCR Extracted Text: {ocr_text}

    Potential Prices Extracted:
    {", ".join(potential_prices) if potential_prices else "No prices detected"}

    Based on this information, provide:
    - A concise product description
    - Product price (per item and per box, where available)
    - Product category and subcategory
    - Common uses for this product
    - Any additional relevant details

    Format your output as JSON with keys:
    - product_name
    - product_description
    - price (SGD)
    - category
    - subcategory
    - common_uses
    """

    print("[DEBUG] Sending prompt to OpenAI API...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Replace with your model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1000,
        temperature=0,
    )

     # Access and log token usage
    usage = response.usage
    if usage:
        print(f"[DEBUG] Token Usage:")
        print(f"  - Prompt Tokens: {usage.prompt_tokens}")
        print(f"  - Completion Tokens: {usage.completion_tokens}")
        print(f"  - Total Tokens: {usage.total_tokens}")

        # Log additional token details if available
        if usage.completion_tokens_details:
            print(f"  - Reasoning Tokens: {usage.completion_tokens_details.reasoning_tokens}")
            print(f"  - Accepted Prediction Tokens: {usage.completion_tokens_details.accepted_prediction_tokens}")
            print(f"  - Rejected Prediction Tokens: {usage.completion_tokens_details.rejected_prediction_tokens}")
    else:
        print("[DEBUG] Token usage data not available in response.")


    print("[DEBUG] Received response from OpenAI API.")


    return response.choices[0].message.content

# -------------------------------------------------------
# 6. MAIN EXECUTION
# -------------------------------------------------------
def main():
    image_path = "./assets/item_4.jpeg"

    print("[DEBUG] Classifying product...")
    product_label = classify_product(image_path)  # Dynamically classify the product
    print(f"[DEBUG] Classified product as: {product_label}")

    print("[DEBUG] Extracting text from the image...")
    ocr_text = extract_text_with_easyocr(image_path)

    print("[DEBUG] Extracting potential prices...")
    potential_prices = extract_potential_prices(ocr_text)

    print("[DEBUG] Generating product information using OpenAI...")
    product_info = generate_product_info(product_label, ocr_text, potential_prices)

    print("\n--- Final LLM Output ---")
    print(product_info)

if __name__ == "__main__":
    main()
