import fitz  # PyMuPDF
import io
from PIL import Image
import os
from google import genai
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def extract_pdf_data(file_path):
    print("\n-> Starting Data Extraction & Vision AI Pipeline...")
    doc = fitz.open(file_path)
    combined_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        print(f"-> Processing Page {page_num + 1}...")
        
        # 1. Extract Raw Text
        text = page.get_text("text")
        if text.strip():
            combined_text.append(f"--- Page {page_num + 1} Text ---\n{text}")

        # 2. Extract Visuals & Send to Vision LLM
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            print(f"   📸 Found image/chart {img_index + 1}. Analyzing with Vision AI...")
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Convert to a format Gemini can read
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # The prompt to force structured insights from charts
            vision_prompt = """
            Analyze this image from a scientific paper. 
            - If it is a chart or graph: Extract the axis labels, describe the main trends, and list key data points. 
            - If it is a diagram/architecture: Describe the flow and components.
            - If it's just a logo or irrelevant shape, reply with 'SKIP'.
            Be highly detailed but concise.
            """
            
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[image, vision_prompt]
                )
                
                vision_text = response.text.strip()
                if vision_text and vision_text != 'SKIP':
                    combined_text.append(f"--- Page {page_num + 1} Chart/Figure Insights ---\n{vision_text}")
                    print(f"   ✅ Chart understood and documented.")
            except Exception as e:
                print(f"   ⚠️ Vision API Error: {e}")

    final_content = "\n\n".join(combined_text)
    print("-> Multimodal Extraction Complete!\n")
    return final_content