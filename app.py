import os
import logging
import requests
import nest_asyncio
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from pyngrok import ngrok, conf
import uvicorn
from dotenv import load_dotenv

nest_asyncio.apply()
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# API Key for authentication
LOCATION_ID = os.getenv("LOCATION_ID")  # GHL Location id
API_KEY = os.getenv("API_KEY")  # GHL API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
conf.get_default().auth_token = os.getenv("CONF_AUTH_TOKEN")
qdrant_api = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    api_key=qdrant_api,
    url=qdrant_url,
    prefer_grpc=True,
    collection_name="my_test_documents",
)


async def get_conversation_id(contact_id: str) -> str:
    """Fetches the conversation ID for a given contact ID."""
    url = "https://services.leadconnectorhq.com/conversations/search"
    querystring = {"locationId": LOCATION_ID, "contactId": contact_id}
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Version": "2021-04-15",
        "Accept": "application/json",
    }

    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()
        data = response.json()

        # Ensure conversations exist
        conversations = data.get("conversations", [])
        if not conversations:
            logger.warning(f"No conversation found for contact_id: {contact_id}")
            return None  # Return None when no conversation exists

        # Extract conversation ID
        conversation_id = conversations[0].get("id")
        return conversation_id

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching conversation ID: {e}", exc_info=True)
        return None


async def get_conversation_messages(conversation_id: str) -> tuple:
    """Fetches and formats messages for a given conversation ID.
    Checks if user has provided contact information.

    Returns:
    - convo (list): Processed conversation messages.
    - contact_info_missing (bool): True if contact info is missing.
    """
    url = (
        f"https://services.leadconnectorhq.com/conversations/{conversation_id}/messages"
    )
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Version": "2021-04-15",
        "Accept": "application/json",
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        convo = [("system", "You are a helpful assistant for Bharat Giveaway.")]

        contact_info_missing = True  # Assume contact info is missing by default

        for msg in reversed(data.get("messages", {}).get("messages", [])):
            role = "ai" if msg["direction"] == "outbound" else "human"
            body = msg["body"]

            # Check if user has entered contact info
            if role == "human" and body.startswith("Contact Information:"):
                logger.info("User has entered contact information. AI can proceed.")
                contact_info_missing = False  # Contact info is now available

            convo.append((role, body))

        return convo, contact_info_missing

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching conversation messages: {e}", exc_info=True)
        return [], True  # Assume contact info is missing if there's an error


async def generate_ai_response(convo: list) -> str:
    """
    Generates an AI response using Gemini with RAG (Retrieval-Augmented Generation).

    :param convo: The conversation history as a list of tuples (role, message).
    :return: The AI-generated response as a string.
    """

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite-preview-02-05",  # Highest rate limit model
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    try:
        # Extract up to the last 3 messages
        num_messages = min(3, len(convo))
        last_messages = [msg[1] for msg in convo[-num_messages:]]
        question = " ".join(last_messages)  # Merge user context
        last_human_message = next((msg[1] for msg in reversed(convo) if msg[0] == "human"), None)

        # Retrieve relevant documents from the vector store
        results = vector_store.similarity_search(question, k=2)

        # Combine retrieved knowledge into a formatted RAG context
        retrieved_context = "\n\n".join([f"- {res.page_content}" for res in results])

        # Prepare the AI input with knowledge + user's actual query
        prompt = f"""You are a helpful assistant for Bharat Giveaways.
        
Use the following relevant information to answer the user's query if helpful:
{retrieved_context}

User's query:
{last_human_message}

You will provide short responce to questions in 2 to 4 lines.
"""

        # Call the LLM with the updated prompt
        ai_msg = llm.invoke(prompt)
        return ai_msg.content

    except Exception as e:
        logger.error(f"Error generating AI response with RAG: {e}", exc_info=True)
        return "All our agents are busy. Please try again in some time."


async def send_message(contact_id: str, message: str) -> dict:
    """Sends a live chat message to a contact."""
    url = "https://services.leadconnectorhq.com/conversations/messages"
    payload = {"type": "Live_Chat", "contactId": contact_id, "message": message}
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Version": "2021-04-15",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending message: {e}", exc_info=True)
        return {"error": str(e)}


@app.route("/health", methods=["GET", "HEAD"])
async def health_check():
    return {"status": "ok", "message": "FastAPI server is running smoothly! 😊"}


@app.post("/webhook")
async def handle_webhook(request: Request):
    """Handles webhook events and ensures AI waits until contact info is provided."""
    try:
        payload = await request.json()
        contact_id = payload.get("contact_id")
        if not contact_id:
            return JSONResponse(
                status_code=400, content={"error": "Missing contact_id"}
            )

        logger.info(f"Received webhook for contact_id: {contact_id}")

        # Step 1: Get Conversation ID
        conversation_id = await get_conversation_id(contact_id)
        if not conversation_id:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "No conversation found for this user yet. Waiting for user input."
                },
            )

        logger.info(f"Found conversation_id: {conversation_id}")

        # Step 2: Get Conversation Messages & Check if Contact Info is Missing
        convo, contact_info_missing = await get_conversation_messages(conversation_id)

        if contact_info_missing:
            logger.info(
                "User has not submitted contact info. Waiting for lead form submission."
            )
            response = await send_message(
                contact_id, "Waiting for user to submit contact information form."
            )
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Waiting for user to submit contact information in the lead form."
                },
            )

        logger.info(f"Fetched conversation messages: {convo}")

        # Step 3: Generate AI Response
        ai_response = await generate_ai_response(convo)
        logger.info(f"AI Response: {ai_response}")

        # Step 4: Send AI Response as a Message
        response = await send_message(contact_id, ai_response)
        logger.info(f"Message sent response: {response}")

        return JSONResponse(
            status_code=200, content={"message": "Success", "ai_response": ai_response}
        )

    except Exception as err:
        logger.error(f"Error in webhook processing: {err}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Internal Server Error"})


# # Expose the app using ngrok
# public_url = ngrok.connect(8000).public_url
# webhook_url = f"{public_url}/webhook"
# print(f"ngrok tunnel URL: {webhook_url}")

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

