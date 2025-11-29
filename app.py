from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr


load_dotenv(override=True)

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


class Me:

    def __init__(self):
        GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
        google_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini = OpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
        self.name = "Mohammad Enamul Haque"
        self.max_questions_per_user = 10
        reader = PdfReader("me/Profile.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()


    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Don't tell the user that you are a chatbot, just answer the questions as if you are {self.name}. \
For investment related questions also show them the required disclaimer  \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt
    
    def count_user_questions(self, history):
        """Count the number of user questions in the conversation history"""
        count = 0
        for msg in history:
            if isinstance(msg, dict) and msg.get("role") == "user":
                count += 1
        return count
    
    def chat(self, message, history):
        # Count user questions (excluding the current one)
        question_count = self.count_user_questions(history)
        
        # Check if user has reached the limit
        if question_count >= self.max_questions_per_user:
            return f"I appreciate your interest! I've answered {self.max_questions_per_user} questions in this session. To continue our conversation, please reach out to me directly via email 'enamul.promy@gmail.com' I'd love to hear from you!"
        
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            try:
                response = self.gemini.chat.completions.create(model="gemini-2.5-flash", messages=messages, tools=tools)
                if response.choices[0].finish_reason=="tool_calls":
                    message = response.choices[0].message
                    tool_calls = message.tool_calls
                    results = self.handle_tool_call(tool_calls)
                    messages.append(message)
                    messages.extend(results)
                else:
                    done = True
            except Exception as e:
                # Handle API limit errors or other API issues
                error_msg = str(e).lower()
                if "quota" in error_msg or "limit" in error_msg or "rate" in error_msg:
                    return "I apologize, but I've reached my API usage limit. Please try again later or contact me directly via email."
                else:
                    return f"I apologize, but I encountered an error: {str(e)}. Please try again later."
        
        return response.choices[0].message.content
    
    def respond(self, message, history):
        """Wrapper function for Gradio interface"""
        if not message:
            return history
        # Convert Gradio history format to messages format
        messages_history = []
        for msg in history:
            messages_history.append({"role": msg["role"], "content": msg["content"]})
        
        response = self.chat(message, messages_history)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return history, ""


if __name__ == "__main__":
    me = Me()
    
    with gr.Blocks(css="""
        .enter-btn {
            background-color: lightslategray !important;
        }
        .enter-btn:hover {
            background-color: #5f7a8a !important;
        }
    """) as demo:
        chatbot = gr.Chatbot(type="messages")
        with gr.Row():
            msg = gr.Textbox(
                placeholder="⚠️ Experimental: For educational and informational purposes only",
                show_label=False,
                container=False,
                scale=9
            )
            submit_btn = gr.Button("Enter", variant="primary", scale=1, elem_classes="enter-btn")
        
        msg.submit(me.respond, [msg, chatbot], [chatbot, msg])
        submit_btn.click(me.respond, [msg, chatbot], [chatbot, msg])
    
    demo.launch()
    