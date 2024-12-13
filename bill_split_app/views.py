import base64
import requests
import json
from PIL import Image
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.core.files.storage import default_storage

SYSTEM_PROMPT = """Act as an OCR assistant. Analyze the provided receipt image and extract ONLY the purchased items. For each item provide:

1. give JSON format with item name, quantity, and price.
2. do not include any other information in the response.
3. give response in plain text format.
4. data should be in the format of item name, quantity, and price.
5. if there is no quantity, then give 1 as the quantity.
6. data should be as JSON array of objects.
7. Do not add the dollar sign ($) in the price.

Skip all headers, footers, addresses, and other receipt information.

Example output format:
[{"item": "Coffee", "quantity": 1, "price": 3.99}, {"item": "Bagel", "quantity": 2, "price": 4.50}]
"""

class OCRAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def encode_image_to_base64(self, image_path):
        """Convert an image file to a base64 encoded string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def perform_ocr_from_text_response(self, image_path, api_url="http://localhost:11434/api/chat"):
        """Perform OCR on the given image."""
        base64_image = self.encode_image_to_base64(image_path)
        response = requests.post(
            api_url,
            json={
                "model": "llama3.2-vision",
                "messages": [
                    {
                        "role": "user",
                        "content": SYSTEM_PROMPT,
                        "images": [base64_image],
                    },
                ],
                "stream": False
            }
        )

        # print(response.text.json())

        if response.status_code == 200:
            try:
                response_data = response.json()
                content = response_data.get("message", {}).get("content", "")
                sanitized_json_string = content.replace('$', '')
                json_array = json.loads(sanitized_json_string)
                return json_array

            except Exception as e:
                return f"[Error] Failed to parse OCR result: {str(e)}"
        else:
            return f"[Error] API returned status code {response.status_code}: {response.text}"


    def post(self, request):
        file_obj = request.FILES['image']
        temp_file_path = default_storage.save(file_obj.name, file_obj)

        try:
            result = self.perform_ocr_from_text_response(temp_file_path)
            return Response({"result": result})
        except Exception as e:
            return Response({"error": str(e)}, status=500)
        finally:
            default_storage.delete(temp_file_path)
