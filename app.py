from flask import Flask, render_template, request, jsonify
import openai
import json
import os
from flask import Flask, request, jsonify
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

from flask import Flask, request, render_template, send_file
import os
import requests
import base64
from werkzeug.utils import secure_filename
load_dotenv()
##################################################################################################

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# OpenAI API key (replace with your key or load from environment)
API_KEY = os.getenv("OPENAI_API_KEY")

# List of product images from S3
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

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def download_image(url, save_path):
    """Download an image from a URL and save it locally."""
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download image from {url}: {response.status_code}")

def save_image_from_base64(response_json, output_path):
    """Decode a Base64-encoded image string and save it as an image file."""
    base64_string = response_json.get("data", [{}])[0].get("b64_json")
    if not base64_string:
        raise ValueError("No Base64 string found in the response.")
    with open(output_path, "wb") as image_file:
        image_file.write(base64.b64decode(base64_string))

def send_openai_image_edit_request(api_key, images, prompt, model="dall-e-2"): #gpt-image-1
    """Send a request to OpenAI's API to generate an edited image."""
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
 

# @app.route("/design", methods=["POST"])
# def design():
#     """Handle interior design requests (originally at / in the interior design app)."""
#     # Handle file upload and prompt
#     room_file = request.files.get("room_image")
#     selected_items = request.form.getlist("selected_items")
#     prompt = request.form.get("prompt")

#     # Validate room image
#     if not room_file or not allowed_file(room_file.filename):
#         return jsonify({"error": "Please upload a valid room image (PNG, JPG, JPEG)."}), 400

#     # Validate prompt
#     if not prompt:
#         return jsonify({"error": "Please provide a prompt."}), 400

#     # Validate selected items (up to 4)
#     if len(selected_items) > 4:
#         return jsonify({"error": "You can select up to 4 items only."}), 400
#     if not selected_items:
#         return jsonify({"error": "Please select at least one product item."}), 400

#     # Save the room image
#     room_filename = secure_filename(room_file.filename)
#     room_filepath = os.path.join(app.config['UPLOAD_FOLDER'], room_filename)
#     room_file.save(room_filepath)

#     # Download selected product images from S3
#     uploaded_files = [room_filepath]  # Start with the room image
#     for i, item_url in enumerate(selected_items):
#         product_filename = f"product_{i}.{item_url.split('.')[-1]}"  # e.g., product_0.png
#         product_filepath = os.path.join(app.config['UPLOAD_FOLDER'], product_filename)
#         try:
#             download_image(item_url, product_filepath)
#             uploaded_files.append(product_filepath)
#         except Exception as e:
#             return jsonify({"error": f"Failed to download product image: {str(e)}"}), 500

#     # Send API request
#     try:
#         response_json = send_openai_image_edit_request(API_KEY, uploaded_files, prompt)
#         output_file = os.path.join(app.config['OUTPUT_FOLDER'], "edited_image.png")
#         save_image_from_base64(response_json, output_file)

#         # Clean up temporary files
#         for file_path in uploaded_files:
#             try:
#                 os.remove(file_path)
#             except Exception as e:
#                 print(f"Failed to delete {file_path}: {e}")

#         return send_file(output_file, mimetype="image/png", as_attachment=True)
#     except Exception as e:
#         return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route("/design", methods=["POST"])
def design():
    """Handle interior design requests (originally at / in the interior design app)."""
    # Handle file upload and prompt
    room_file = request.files.get("room_image")
    selected_items = request.form.getlist("selected_items")
    prompt = request.form.get("prompt")

    # Validate room image
    if not room_file or not allowed_file(room_file.filename):
        return jsonify({"error": "Please upload a valid room image (PNG, JPG, JPEG)."}), 400

    # Validate prompt
    if not prompt:
        return jsonify({"error": "Please provide a prompt."}), 400

    # Validate selected items (up to 4)
    if len(selected_items) > 4:
        return jsonify({"error": "You can select up to 4 items only."}), 400
    if not selected_items:
        return jsonify({"error": "Please select at least one product item."}), 400

    # Save the room image
    room_filename = secure_filename(room_file.filename)
    room_filepath = os.path.join(app.config['UPLOAD_FOLDER'], room_filename)
    room_file.save(room_filepath)

    # Download selected product images from S3
    uploaded_files = [room_filepath]  # Start with the room image
    for i, item_url in enumerate(selected_items):
        product_filename = f"product_{i}.{item_url.split('.')[-1]}"  # e.g., product_0.png
        product_filepath = os.path.join(app.config['UPLOAD_FOLDER'], product_filename)
        try:
            download_image(item_url, product_filepath)
            uploaded_files.append(product_filepath)
        except Exception as e:
            return jsonify({"error": f"Failed to download product image: {str(e)}"}), 500

    # Send API request
    try:
        response_json = send_openai_image_edit_request(API_KEY, uploaded_files, prompt)
        output_file = os.path.join(app.config['OUTPUT_FOLDER'], "edited_image.png")
        save_image_from_base64(response_json, output_file)

        # Clean up temporary files
        for file_path in uploaded_files:
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

        return send_file(output_file, mimetype="image/png", as_attachment=True)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

##################################################################################################

# Load environment variables

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

##########################################RAG Part###############################################
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo",
    temperature=0.7
)

knowledge_base = """
Our Origin ->
Our humble journey begins with our founder's deep love and respect for nature. 
Born an adventurous soul in the small, rural village of Morawaka in Sri Lanka, 
Samantha Kumarasinghe grew up strongly aware of the beauty and strength of nature, of its many natural healing wonders, 
and the great need to protect nature for posterity.

These very values of finding inspiration in nature whilst also passionately protecting it represent the ethos of the company he founded, 
and are at the core of the decisions that we take and the products we create as we journey on to reach our vision to become the most environment-friendly cosmetics manufacturer in the world.

Sustainability Initiatives ->
From our company's inception itself we have been determined to invest in the future of our planet and its people. 
Our respect for nature and our many continuous initiatives to protect it have also been nationally recognized, 
with our company being awarded Sri Lanka's highest environmental award; the National Green Award-Gold.

We are passionately committed to further expand our sustainability initiatives to achieve our vision 'to be the most environment-friendly cosmetics manufacturer in the world'. 
Our initiatives also align with the United Nations' Sustainable Development Goals for a better and more sustainable future, especially under its Goal 6 of Clean Water & Sanitation, 
Goal 12 of Responsible Consumption & Production, and Goal 15 of Life on Land.

Awards & Certifications ->
From its inception, Nature's Beauty Creations Ltd aspired to create the finest, 
most unique cosmetics manufacturing facility in Sri Lanka.

Today we still remain the most advanced, ultra-hygienic, eco-friendly and sought-after cosmetics manufacturer, 
our quality standards even surpassing European GMP requirements.

The company has won many accolades for its facilities and initiatives, 
and was also the first cosmetics manufacturer in Sri Lanka to receive international GMP, ISO 9001 and ISO 14001 certification.

Overseas Distribution & Exports ->
We have embarked on an exciting journey to create global beauty and baby care brands from Sri Lanka.

Our brands are currently enjoyed in several countries worldwide, 
so if you would like to see if our products are available in your country for easy purchase, drop us a message and find out!

We are also incredibly motivated to continue to grow our international presence and bring our amazing products to even more consumers overseas. 
If you have an established distribution network and are interested in becoming an agent for our brands in your country we would love to hear from you.

You can easily reach us via exports@nbc.lk
"""

# Create a LangChain Prompt Template
prompt_template = PromptTemplate(
    input_variables=["knowledge_base", "question"],
    template="""
    You are a helpful assistant that answers questions based on the following knowledge base:

    {knowledge_base}

    Question: {question}
    Answer:
    """
)

# Create the LangChain LLM Chain
chain = LLMChain(llm=llm, prompt=prompt_template)

def retrieve_from_knowledge_base(question):
    """Retrieve information from the knowledge base."""
    try:
        if not question:
            return jsonify({"error": "Question is required"}), 400

        answer = chain.run({"knowledge_base": knowledge_base, "question": question})

        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


########################################Product Recommendation################################
app = Flask(__name__)

with open("product.json") as f:
    products = json.load(f)

openai.api_key = os.getenv("OPENAI_API_KEY")

def recommend_products(user_preferences, category=None, max_budget=None):
    """
    Recommend products based on user preferences, category, and budget.
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
        product_features = [feature.lower() for feature in product["features"]]
        product_category = product["category"].lower()
        
        print(f"\nDebug - Evaluating product: {product['name']}")
        print(f"- Category: {product['category']}")
        print(f"- Price: {product['price']}")
        print(f"- Features: {product_features}")

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

        # Score based on matching features/skin types
        for pref in user_preferences:
            if pref in product_features:
                score += 2  # Increased weight for direct feature match
                print(f"- Feature match: {pref} (+2)")
            if pref in product["description"].lower():
                score += 0.5
                print(f"- Description match: {pref} (+0.5)")

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
                "content": """You are an assistant that extracts structured preferences from user inputs for skincare and beauty products.
                Extract the following information:
                Category: [exact product category mentioned or "None" if not specified]
                Skin Type: [mentioned skin type or "None" if not specified]
                Budget: [maximum price as number only or "None" if not specified]
                Features: [comma-separated list of specific features mentioned]
                
                Example input: "can you suggest eye care cream my skin is dry and my budget is 500?"
                Example output:
                Category: eye care
                Skin Type: dry
                Budget: 500
                Features: eye care, dry skin"""
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
                elif "skin type" in key.lower() or "features" in key.lower():
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
    
##############################################END Product Recommendation##############################################


###############################Get Ingredients#######################################
def get_product_ingredients(user_message: str, chat_history: list):
    """
    Handles user queries about product ingredients with chat history.
    """
    # First messages array to identify the product
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
        # First, identify the product
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=identify_messages,
            temperature=0.3,
            max_tokens=50
        )
        
        product_query = completion.choices[0].message["content"].lower()
        
        # Find matching products
        matching_products = []
        for product in products:
            if product["name"].lower() in product_query or product_query in product["name"].lower():
                matching_products.append(product)

        if len(matching_products) == 0:
            return "I apologize, but I couldn't find that specific product in our catalog. Could you please verify the product name or tell me more about what you're looking for?"

        elif len(matching_products) == 1:
            product = matching_products[0]
            
            # Create a new messages array for formatting the response
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

            # Get formatted response
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
    


    
######################################End Ingredients########################################


############################################Function Calling#########################################################
functions = [
    {
        "name": "rag_retrieve_information",
        "description": (
            "Use this function to retrieve information from the knowledge base about company origins, sustainability initiatives, "
            "awards, certifications, and overseas distribution. "
            "Examples: Who is the founder of the company? What sustainability initiatives does the company follow? "
            "What are the company's international certifications?"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_query": {
                    "type": "string",
                    "description": "The question related to the company's background, sustainability, awards, or international presence.",
                }
            },
            "required": ["question"]
        }
    },
    {
        "name": "recommend_products",
        "description": (
            "Use this function to recommend beauty and skincare products based on user preferences such as category, skin type, "
            "budget, and desired features. "
            "Examples: Can you suggest an eye care cream for dry skin under $500? What are the best natural face cleansers?"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_query": {
                    "type": "string",
                    "description": "The user query specifying skincare needs, product type, skin type, budget, or features."
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


def get_chatbot_response(user_message: str, chat_history: list):
    """Handles user queries with chat history using RAG and product recommendation functions."""
    
    messages = [
        {
            "role": "system",
            "content": """
            You are an assistant specializing in company-related information and product recommendations.
            - For questions about company history, sustainability, awards, exports, or other organizational details, use the rag_retrieve_information function.
            - For product-related recommendations (e.g., skincare products, budget, skin type, features), use the recommend_products function.
            - For get product indridients, use the get_product_ingredients function.
            - Always use one of these functions—do not answer directly.
            - Consider previous messages for follow-up questions to maintain context.
            - If the query cannot be answered using the available information, respond with: 'I'm unable to find the required information.'
            """
        }
    ]
    
    for msg in chat_history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            # Convert role if necessary
            role = "assistant" if msg["role"] == "bot" else msg["role"]
            # Add properly formatted message
            messages.append({
                "role": role,
                "content": str(msg["content"])  # Ensure content is string
            })
    
    # Add current user message
    messages.append({
        "role": "user",
        "content": str(user_message)  # Ensure content is string
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
    

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/rag", methods=["POST"])
def rag_endpoint():
    """Endpoint for retrieving information from the knowledge base."""
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    response = retrieve_from_knowledge_base(question)
    return jsonify(response)

@app.route("/recommend", methods=["POST"])
def recommend_endpoint():
    """Endpoint for recommending products based on user preferences."""
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "User input is required"}), 400

    response = recommend_products_from_catalog(question)
    return jsonify(response)
    

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """Chat endpoint that dynamically handles RAG queries and product recommendations."""
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
    

@app.route("/")
def index():
    """Render the main page with the chatbot and interior design panel."""
    return render_template("new.html", product_images=PRODUCT_IMAGES)


if __name__ == "__main__":
    app.run(debug=True)
#####################################END Function calling##################################

