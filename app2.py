from flask import Flask, render_template, request, jsonify, send_file
import openai
import json
import os
import requests
import base64
from werkzeug.utils import secure_filename
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configuration for interior design uploads
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# OpenAI API key
API_KEY = os.getenv("OPENAI_API_KEY")

# E-commerce knowledge base
knowledge_base = """
Phoenix Group in Sri Lanka primarily refers to Phoenix Industries Ltd., a leading plastic manufacturing company. They produce a wide range of plastic products, including furniture, household items, packaging, and more. They are also known for their rigid packaging solutions and toothbrush manufacturing. 
Phoenix Industries Ltd. (Sri Lanka):
Established in 1976:
They have a long history in the Sri Lankan plastics industry, being the first company in South-East Asia to introduce in-mould labelling. 
Manufacturing and Marketing:
They manufacture and market a wide range of resin-based products, including furniture, household items, crates, pallets, and pipes & fittings. 
Key Supplier:
Phoenix Industries is a key supplier of rigid packaging solutions and toothbrushes to large businesses in Sri Lanka. 
Modern Manufacturing:
Their modern manufacturing plant is equipped with advanced machinery for injection molding, blow molding, and a fully-automated PET line. 
Customer Base:
Their customer portfolio includes major brands like Unilevers, Chevron, Hemas, CCS, Cargills, and more. 
Locations:
They have multiple locations including Welisara, Makandura, and Horana. 
"""

# LangChain setup for e-commerce RAG
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo",
    temperature=0.7
)

prompt_template = PromptTemplate(
    input_variables=["knowledge_base", "question"],
    template="""
    You are a helpful assistant that answers questions based on the following knowledge base:

    {knowledge_base}

    Question: {question}
    Answer:
    """
)

chain = LLMChain(llm=llm, prompt=prompt_template)

# List of product images for interior design
PRODUCT_IMAGES = [
    {"url": "https://honor-ai-video-gen.s3.ap-south-1.amazonaws.com/temp_img/phoenix+1.png", "label": "FLO L PACK 1"},
    {"url": "https://honor-ai-video-gen.s3.ap-south-1.amazonaws.com/temp_img/phoenix+2.png", "label": "White Single Seat"},
    {"url": "https://honor-ai-video-gen.s3.ap-south-1.amazonaws.com/temp_img/phoenix+3.png", "label": "FLO L PACK 1"},
    {"url": "https://honor-ai-video-gen.s3.ap-south-1.amazonaws.com/temp_img/phoenix+4.png", "label": "Chair Cambridge"},
    {"url": "https://honor-ai-video-gen.s3.ap-south-1.amazonaws.com/temp_img/phoenix+5.png", "label": "Chair Mint"},
    {"url": "https://honor-ai-video-gen.s3.ap-south-1.amazonaws.com/temp_img/phoenix+6.jpg", "label": "Deco Box"},
    {"url": "https://honor-ai-video-gen.s3.ap-south-1.amazonaws.com/temp_img/phoenix+7.jpg", "label": "10L Storage Box-Coloured"},
    {"url": "https://honor-ai-video-gen.s3.ap-south-1.amazonaws.com/temp_img/phoenix+8.jpg", "label": "FLO L PACK 3"},
    {"url": "https://honor-ai-video-gen.s3.ap-south-1.amazonaws.com/temp_img/phoenix+9.jpg", "label": "6 Seater Table"},
    {"url": "https://honor-ai-video-gen.s3.ap-south-1.amazonaws.com/temp_img/phoenix+10.jpg", "label": "6 Seater Table"},
    {"url": "https://honor-ai-video-gen.s3.ap-south-1.amazonaws.com/temp_img/phoenix+11.jpg", "label": "Shoes Rack"},
    {"url": "https://honor-ai-video-gen.s3.ap-south-1.amazonaws.com/temp_img/phoenix+13.jpg", "label": "6 Seater Table"},
]

# E-commerce product recommendation logic
with open("product.json") as f:
    products = json.load(f)

openai.api_key = os.getenv("OPENAI_API_KEY")


def recommend_products(user_preferences, category=None, max_budget=None):
    """
    Recommend plastic products based on user preferences, category, and budget, using only description and category.
    """
    print(f"Debug - Input parameters:")
    print(f"- User preferences: {user_preferences}")
    print(f"- Category: {category}")
    print(f"- Max budget: {max_budget}")

    recommended = []
    score_tracking = []

    # Convert preferences to lowercase for case-insensitive matching
    user_preferences = [pref.lower() for pref in user_preferences if pref]
    
    # Get all unique categories from products
    all_categories = set(p["category"].lower() for p in products)
    
    # If category is provided, find the closest matching category
    matched_category = None
    if category:
        category = category.lower()
        # Try exact match first
        if category in all_categories:
            matched_category = category
        else:
            # Try partial match
            matches = [c for c in all_categories if category in c or c in category]
            if matches:
                matched_category = matches[0]
        
        print(f"Debug - Matched category: {matched_category}")

    for product in products:
        score = 0
        product_description = product["description"].lower()
        product_category = product["category"].lower()
        
        print(f"\nDebug - Evaluating product: {product['name']}")
        print(f"- Category: {product['category']}")
        print(f"- Price: {product['price']}")
        print(f"- Description: {product_description}")

        # Skip if product is over budget
        if max_budget and product["price"] > max_budget:
            print(f"- Skipped: Over budget ({product['price']} > {max_budget})")
            continue

        # Category filtering - only consider if category matches or no category specified
        if matched_category and matched_category not in product_category:
            print(f"- Skipped: Category mismatch (looking for {matched_category})")
            continue

        # Give base score for category match
        if matched_category and matched_category in product_category:
            score += 5  # Base score for matching the requested category
            print(f"- Category match base score (+5)")

        # Score based on matching preferences in product description
        for pref in user_preferences:
            if pref in product_description:
                score += 2  # Score for description match
                print(f"- Description match: {pref} (+2)")
            if pref in product_category:
                score += 1  # Additional score for category match
                print(f"- Category keyword match: {pref} (+1)")

        # Bonus points for products that are under budget but not too cheap
        if max_budget and product["price"] <= max_budget:
            budget_ratio = product["price"] / max_budget
            if budget_ratio > 0.7:  # Products using 70-100% of budget get bonus
                bonus = 0.5
                score += bonus
                print(f"- Budget optimization bonus: +{bonus}")
            
            # Price efficiency bonus (lower price = higher score)
            efficiency_bonus = (1 - (product["price"] / max_budget)) * 2
            score += efficiency_bonus
            print(f"- Price efficiency bonus: +{efficiency_bonus:.2f}")

        print(f"- Final score: {score}")

        # Include all products that match category and budget
        if score > 0:
            score_tracking.append((product, score))

    # Sort by relevance score and get top matches
    score_tracking.sort(key=lambda x: x[1], reverse=True)
    recommended = [item[0] for item in score_tracking[:3]]
    
    print("\nDebug - Final recommendations:")
    for prod in recommended:
        print(f"- {prod['name']} (Score: {next(score for p, score in score_tracking if p == prod)})")
    
    return recommended


def recommend_products_from_catalog(question):
    if not question:
        return {"error": "User input is required"}, 400

    try:
        messages = [
            {
                "role": "system",
                "content": """You are an assistant that extracts structured preferences from user inputs for plastic products.
                Extract the following information:
                Category: [exact product category mentioned or "None" if not specified]
                Use Case: [mentioned use case or purpose (e.g., storage, seating, household) or "None" if not specified]
                Budget: [maximum price as number only or "None" if not specified]
                Keywords: [comma-separated list of specific keywords or descriptors mentioned related to product use or description]
                
                Example input: "can you suggest a storage crate for kitchen use under 5000?"
                Example output:
                Category: Crates & Pallets
                Use Case: kitchen storage
                Budget: 5000
                Keywords: storage, crate, kitchen"""
            },
            {
                "role": "user",
                "content": f'Extract preferences from: "{question}"'
            }
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=100,
            temperature=0.5
        )

        # Parse preferences from the GPT-4 response
        preferences_text = response["choices"][0]["message"]["content"].strip()
        print(f"Debug - GPT-4 Response:\n{preferences_text}")
        
        # Parse the structured response
        category = None
        max_budget = None
        preferences = []
        
        for line in preferences_text.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                value = value.strip().lower()
                
                if "category" in key.lower():
                    if value != "none":
                        category = value
                    print(f"Debug - Extracted category: {category}")
                elif "budget" in key.lower():
                    if value != "none":
                        try:
                            # Extract numbers from the value, handle currency symbols
                            budget_str = ''.join(c for c in value if c.isdigit() or c == '.')
                            if budget_str:
                                max_budget = float(budget_str)
                                print(f"Debug - Extracted budget: {max_budget}")
                        except ValueError:
                            print(f"Debug - Failed to parse budget from: {value}")
                elif "use case" in key.lower() or "keywords" in key.lower():
                    if value != "none":
                        new_prefs = [v.strip() for v in value.split(",") if v.strip()]
                        preferences.extend(new_prefs)
                        print(f"Debug - Added preferences: {new_prefs}")

        # Get recommended products with category and budget constraints
        recommendations = recommend_products(
            user_preferences=preferences,
            category=category,
            max_budget=max_budget
        )

        if not recommendations:
            print("Debug - No recommendations found")
            return {"error": "No matching products found"}, 404

        return {"answer": recommendations}

    except Exception as e:
        print(f"Debug - Error occurred: {str(e)}")
        return {"error": str(e)}, 500
    

def get_product_ingredients(user_message: str, chat_history: list):
    identify_messages = [
        {
            "role": "system",
            "content": """
            You are a skincare expert. Extract the exact product name from the user's query.
            Only return the product name, nothing else.
            """
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=identify_messages,
            temperature=0.3,
            max_tokens=50
        )
        
        product_query = completion.choices[0].message["content"].lower()
        matching_products = []
        for product in products:
            if product["name"].lower() in product_query or product_query in product["name"].lower():
                matching_products.append(product)

        if len(matching_products) == 0:
            return "I apologize, but I couldn't find that specific product in our catalog. Could you please verify the product name or tell me more about what you're looking for?"
        elif len(matching_products) == 1:
            product = matching_products[0]
            format_messages = [
                {
                    "role": "system",
                    "content": f"""
                    You are a knowledgeable skincare expert. Provide information about the ingredients in {product['name']}.
                    Use this exact ingredient list (do not modify or add to it): {product['ingredients']}
                    
                    Follow these rules:
                    1. Be conversational and friendly
                    2. Start with a brief introduction about the product
                    3. Present the ingredients in a clear way
                    4. Don't add any ingredients that aren't in the list
                    5. Don't make claims about effects not mentioned in the product description
                    6. Use the exact product name
                    """
                },
                {
                    "role": "user",
                    "content": f"Tell me about the ingredients in {product['name']}."
                }
            ]

            response_completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=format_messages,
                temperature=0.7,
                max_tokens=400
            )
            
            return response_completion.choices[0].message["content"]
        else:
            product_names = [p["name"] for p in matching_products]
            response = "I found several products that match your query. Which one would you like to know about?\n\n"
            for name in product_names:
                response += f"• {name}\n"
            return response

    except Exception as e:
        print(f"Error in get_product_ingredients: {str(e)}")
        return "I apologize, but I encountered an error while processing your request. Please try again."
    


# E-commerce function calling
functions = [
    {
        "name": "rag_retrieve_information",
        "description": (
            "Use this function to retrieve information from the knowledge base about company history, manufacturing, customer base, "
            "locations, or other organizational details. "
            "Examples: When was Phoenix Industries established? What types of products does the company manufacture? "
            "Who are the major customers of Phoenix Industries?"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_query": {
                    "type": "string",
                    "description": "The question related to the company's history, manufacturing, customer base, or locations.",
                }
            },
            "required": ["question"]
        }
    },
    {
        "name": "recommend_products",
        "description": (
            "Use this function to recommend plastic products based on user preferences such as category, budget, "
            "and description keywords. "
            "Examples: Can you suggest a storage crate for kitchen use under 5000? What are the best chairs for outdoor seating?"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_query": {
                    "type": "string",
                    "description": "The user query specifying plastic product needs, product type, budget, or description keywords."
                }
            },
            "required": ["question"]
        }
    },
    {
        "name": "get_product_ingredients",
        "description": (
            "Use this function to retrieve the ingredients of a specific product. "
            "Examples: What are the ingredients in Aloe Vera Gel? Show me what's in the Gotukola Under Eye Cream."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_query": {
                    "type": "string",
                    "description": "The question asking about ingredients of a specific product."
                }
            },
            "required": ["product_query"]
        }
    }
]


def retrieve_from_knowledge_base(question):
    try:
        if not question:
            return jsonify({"error": "Question is required"}), 400
        answer = chain.run({"knowledge_base": knowledge_base, "question": question})
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

def get_chatbot_response(user_message: str, chat_history: list):
    messages = [
        {
            "role": "system",
            "content": """
            You are an assistant specializing in company-related information and product recommendations.
            - For questions about company history, manufacturing, customer base, locations, or other organizational details, use the rag_retrieve_information function.
            - For product-related recommendations (e.g., plastic products, budget, category, description), use the recommend_products function.
            - For get product indridients, use the get_product_ingredients function.
            - Always use one of these functions—do not answer directly.
            - Consider previous messages for follow-up questions to maintain context.
            - If the query cannot be answered using the available information, respond with: 'I'm unable to find the required information.'
            """
        }
    ]
    
    for msg in chat_history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            role = "assistant" if msg["role"] == "bot" else msg["role"]
            messages.append({
                "role": role,
                "content": str(msg["content"])
            })
    
    messages.append({
        "role": "user",
        "content": str(user_message)
    })

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=messages,
            functions=functions,
            function_call="auto"
        )
        
        response = completion.choices[0].message

        if "function_call" in response:
            function_name = response.function_call.name
            function_args = json.loads(response.function_call.arguments)
            
            if function_name == "rag_retrieve_information":
                question = function_args.get("user_query")
                if question:
                    result = retrieve_from_knowledge_base(question)
                    if isinstance(result, tuple):
                        return result[0].get_json().get("answer", "Error processing request")
                    return result.get_json().get("answer", "Error processing request")
                    
            elif function_name == "recommend_products":
                question = function_args.get("user_query")
                if question:
                    result = recommend_products_from_catalog(question)
                    if isinstance(result, tuple):
                        return result[0].get("answer", "No matching products found")
                    return result.get("answer", "No matching products found")
                
            elif function_name == "get_product_ingredients":
                product_query = function_args.get("product_query")
                if product_query:
                    return get_product_ingredients(product_query, chat_history)
                  
            return "I couldn't process your query. Please try again."
        else:
            return response.content if hasattr(response, 'content') else "I couldn't process your query. Please try again."
            
    except Exception as e:
        print(f"Error in get_chatbot_response: {str(e)}")
        return f"An error occurred while processing your request. Please try again."

# Interior design helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def download_image(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download image from {url}: {response.status_code}")

def save_image_from_base64(response_json, output_path):
    base64_string = response_json.get("data", [{}])[0].get("b64_json")
    if not base64_string:
        raise ValueError("No Base64 string found in the response.")
    with open(output_path, "wb") as image_file:
        image_file.write(base64.b64decode(base64_string))

def send_openai_image_edit_request(api_key, images, prompt, model="gpt-image-1"):
    url = "https://api.openai.com/v1/images/edits"
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"model": (None, model), "prompt": (None, prompt)}
    
    for i, image_path in enumerate(images):
        with open(image_path, "rb") as image_file:
            files[f"image[{i}]"] = (secure_filename(image_path), image_file.read(), "image/jpeg")
    
    response = requests.post(url, headers=headers, files=files)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")

# E-commerce routes
@app.route("/rag", methods=["POST"])
def rag_endpoint():
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "Question is required"}), 400
    response = retrieve_from_knowledge_base(question)
    return jsonify(response)

@app.route("/recommend", methods=["POST"])
def recommend_endpoint():
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "User input is required"}), 400
    response = recommend_products_from_catalog(question)
    return jsonify(response)

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    try:
        data = request.get_json()
        user_message = data.get("message")
        chat_history = data.get("history", [])
        if not user_message:
            return jsonify({"error": "User message is required"}), 400
        response = get_chatbot_response(user_message, chat_history)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Interior design route
@app.route("/design", methods=["POST"])
def design_endpoint():
    if request.method == "POST":
        room_file = request.files.get("room_image")
        selected_items = request.form.getlist("selected_items")
        prompt = request.form.get("prompt")

        if not room_file or not allowed_file(room_file.filename):
            return jsonify({"error": "Please upload a valid room image (PNG, JPG, JPEG)."}), 400

        if not prompt:
            return jsonify({"error": "Please provide a prompt."}), 400

        if len(selected_items) > 4:
            return jsonify({"error": "You can select up to 4 items only."}), 400
        if not selected_items:
            return jsonify({"error": "Please select at least one product item."}), 400

        room_filename = secure_filename(room_file.filename)
        room_filepath = os.path.join(app.config['UPLOAD_FOLDER'], room_filename)
        room_file.save(room_filepath)

        uploaded_files = [room_filepath]
        for i, item_url in enumerate(selected_items):
            product_filename = f"product_{i}.{item_url.split('.')[-1]}"
            product_filepath = os.path.join(app.config['UPLOAD_FOLDER'], product_filename)
            try:
                download_image(item_url, product_filepath)
                uploaded_files.append(product_filepath)
            except Exception as e:
                return jsonify({"error": f"Failed to download product image: {str(e)}"}), 500

        try:
            response_json = send_openai_image_edit_request(API_KEY, uploaded_files, prompt)
            output_file = os.path.join(app.config['OUTPUT_FOLDER'], "edited_image.png")
            save_image_from_base64(response_json, output_file)

            for file_path in uploaded_files:
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

            with open(output_file, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            return jsonify({"image": image_data})
        except Exception as e:
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Main route to render the combined frontend
@app.route("/")
def index():
    return render_template("new.html", product_images=PRODUCT_IMAGES)

if __name__ == "__main__":
    app.run(debug=True)