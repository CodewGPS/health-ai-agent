import os
import asyncio
import chainlit as cl
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# System instruction: scope, guardrails, and output style
SYSTEM_PROMPT = (
    "You are a cautious, educational virtual health assistant. Your role is to:\n"
    "1) Ask the user for their age, gender, and symptoms if not provided\n"
    "2) Once you have this information, suggest possible differential diagnoses\n"
    "3) Provide triage recommendations and red-flag indicators\n\n"
    "Important guidelines:\n"
    "- You are NOT a substitute for a clinician\n"
    "- Avoid definitive diagnoses; present differentials with reasoning\n"
    "- Be concise and use bullet points\n"
    "- Always ask for missing information before providing medical suggestions\n"
    "- If symptoms are severe (chest pain, severe bleeding, difficulty breathing, etc.), "
    "immediately advise seeking urgent medical care\n\n"
    "Response format when you have all info:\n"
    "1) List 3-6 possible differential diagnoses ranked by likelihood\n"
    "2) Suggest immediate self-care steps and generate age appropriate medicines.\n"
    "3) Advise when to seek urgent care\n"
    "4) Always end with the disclaimer: 'Education-only disclaimer: This information is for "
    "educational purposes and is not medical advice; it does not replace evaluation by a qualified "
    "clinician. If symptoms are severe, worsening, or you notice red flags (e.g., chest pain, severe "
    "shortness of breath, confusion, fainting, uncontrolled bleeding), seek urgent medical care immediately.'"
)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("messages", [
        {"role": "system", "content": SYSTEM_PROMPT}
    ])
    await cl.Message(
        content=(
            "Hi! I'm your educational health assistant. "
            "Please tell me about your symptoms, age, and gender so I can help you better."
        )
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    
    messages = cl.user_session.get("messages")
    
   
    messages.append({"role": "user", "content": message.content})
    
    response_msg = cl.Message(content="")
    await response_msg.send()
    
    try:
       
        stream = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.3,
            top_p=0.95,
            max_tokens=800,
            stream=True
        )
        
       
        response_content = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                response_content += token
                await response_msg.stream_token(token)
        
        await response_msg.update()
        
      
        messages.append({"role": "assistant", "content": response_content})
        cl.user_session.set("messages", messages)
        
    except Exception as e:
        await response_msg.stream_token(
            f"Sorry, there was an error: {str(e)}"
        )
        await response_msg.update()
