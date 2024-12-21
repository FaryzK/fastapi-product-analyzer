import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize FastAPI app
app = FastAPI()

# Define input model for request body
class ImageRequest(BaseModel):
    image_url: str

@app.post("/analyze-product/")
async def analyze_product(image_request: ImageRequest):
    """
    API endpoint to analyze a product from an image URL.
    """
    image_url = image_request.image_url

    # Validate image URL
    if not image_url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid image URL provided.")

    # Prepare the prompt
    prompt = """
    Analyze the attached image of a product and provide the following details when they can be found:
    - Product name
    - Price (as a single string, describing all variations such as per unit, per box, etc., in SGD where available)
    - Product category and subcategory
    - Common uses
    - Any additional relevant details
    - Write a concise and professional product description suitable for an e-commerce listing.

    Only for the price field, do not guess the price when the field isn't available. Should there be no price information, state "null".

    Format your output as JSON with keys:
    - product_name
    - product_description
    - price (string)
    - category
    - subcategory
    - common_uses
    """

    # Send the request to OpenAI API
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
            max_tokens=1000,
            temperature=0,
        )

        # Extract and format the response
        raw_data = response.choices[0].message.content

        # Parse the JSON content from the response
        try:
            formatted_data = json.loads(raw_data.strip("```json").strip("```").strip())
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"Error parsing JSON: {str(e)}")

        # Include token usage in the response
        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        return {
            "status": "success",
            "data": formatted_data,
            "token_usage": token_usage,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
