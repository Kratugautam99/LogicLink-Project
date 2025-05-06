import uuid
import time
import re
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
import modelscope_studio.components.antd as antd
import modelscope_studio.components.antdx as antdx
import modelscope_studio.components.base as ms
import modelscope_studio.components.pro as pro
from config import DEFAULT_LOCALE, DEFAULT_THEME, get_text, user_config, bot_config, welcome_config
from ui_components.logo import Logo
from ui_components.settings_header import SettingsHeader

# Loading the tokenizer and model from Hugging Face's model hub

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Using CUDA for an optimal experience

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Defining a custom stopping criteria class for the model's text generation

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [2]  # IDs of tokens where the generation should stop.
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

# Function to generate model predictions with streaming

def generate_response(user_input, history):
    stop = StopOnTokens()
    messages = "</s>".join([
        "</s>".join([
            "\n<|user|>:" + item["content"] if item["role"] == "user"
            else "\n<|assistant|>:" + item["content"]
            for item in history
        ])
    ])
    messages += f"\n<|user|>:{user_input}\n<|assistant|>:"
    model_inputs = tokenizer([messages], return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        **model_inputs,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop])
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()  # Start generation in a separate thread.
    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        if '</s>' in partial_message:
            break
    return partial_message

# Define the system prompt for seeding the model's context

SYSTEM_PROMPT = (
    "I am LogicLink, Version 5—a state-of-the-art AI chatbot created by "
    "Kratu Gautam (A-27) and Geetank Sahare (A-28) from SY CSE(AIML) GHRCEM. "
    "I am here to assist you with any queries. How can I help you today?"
)

class Gradio_Events:
    _generating = False

    @staticmethod
    def new_chat(state_value):
        # This is CRITICAL - we DO NOT clean up old conversation
        # Instead, we leave it in the state to be accessed later

        # Create a fresh conversation
        new_id = str(uuid.uuid4())
        state_value["conversation_id"] = new_id

        # Add the new conversation to the list with a default name
        state_value["conversations"].append({
            "label": "New Chat",
            "key": new_id
        })

        # Seed it with system prompt
        state_value["conversation_contexts"][new_id] = {
            "history": [{
                "role": "system",
                "content": SYSTEM_PROMPT,
                "key": str(uuid.uuid4()),
                "avatar": None
            }]
        }

        # Return updates
        return (
            gr.update(items=state_value["conversations"]),
            gr.update(value=state_value["conversation_contexts"][new_id]["history"]),
            gr.update(value=state_value),
            gr.update(value="")  # empties input
        )

    @staticmethod
    def add_message(input_value, state_value):
        input_update = gr.update(value="")

        # If input is empty, just return
        if not input_value.strip():
            conversation = state_value["conversation_contexts"].get(state_value["conversation_id"], {"history": []})
            chatbot_update = gr.update(value=conversation["history"])
            state_update = gr.update(value=state_value)
            return input_update, chatbot_update, state_update

        # If there's no active conversation, initialize a new one
        if not state_value["conversation_id"]:
            random_id = str(uuid.uuid4())
            state_value["conversation_id"] = random_id
            state_value["conversation_contexts"][random_id] = {"history": [{
                "role": "system",
                "content": SYSTEM_PROMPT,
                "key": str(uuid.uuid4()),
                "avatar": None
            }]}

            # Set the chat name to the first message from user
            chat_name = input_value[:20] + ("..." if len(input_value) > 20 else "")
            state_value["conversations"].append({
                "label": chat_name,
                "key": random_id
            })
        else:
            # Get current conversation history
            current_id = state_value["conversation_id"]
            history = state_value["conversation_contexts"][current_id]["history"]

            # If this is the first user message (after system message), update the label
            user_messages = [msg for msg in history if msg["role"] == "user"]
            if len(user_messages) == 0:
                # This is the first user message - update the chat name
                chat_name = input_value[:20] + ("..." if len(input_value) > 20 else "")
                for i, conv in enumerate(state_value["conversations"]):
                    if conv["key"] == current_id:
                        state_value["conversations"][i]["label"] = chat_name
                        break

        # Add the message to history
        history = state_value["conversation_contexts"][state_value["conversation_id"]]["history"]
        history.append({
            "role": "user",
            "content": input_value,
            "key": str(uuid.uuid4()),
            "avatar": None
        })

        chatbot_update = gr.update(value=history)
        return input_update, chatbot_update, gr.update(value=state_value)

    @staticmethod
    def submit(state_value):
        if Gradio_Events._generating:
            history = state_value["conversation_contexts"].get(state_value["conversation_id"], {"history": []})["history"]
            return (
                gr.update(value=history),
                gr.update(value=state_value),
                gr.update(value="Generation in progress, please wait...")
            )

        Gradio_Events._generating = True

        # Make sure we have a valid conversation ID
        if not state_value["conversation_id"]:
            Gradio_Events._generating = False
            return (
                gr.update(value=[]),
                gr.update(value=state_value),
                gr.update(value="No active conversation")
            )

        history = state_value["conversation_contexts"][state_value["conversation_id"]]["history"]

        # Assuming the last message is the latest user input
        user_input = history[-1]["content"] if (history and history[-1]["role"] == "user") else ""
        if not user_input:
            Gradio_Events._generating = False
            return (
                gr.update(value=history),
                gr.update(value=state_value),
                gr.update(value="No user input provided")
            )

        # Generate the response from the model
        history, response = Gradio_Events.logiclink_chat(user_input, history)
        state_value["conversation_contexts"][state_value["conversation_id"]]["history"] = history
        Gradio_Events._generating = False
        return (
            gr.update(value=history),
            gr.update(value=state_value),
            gr.update(value=response)
        )

    @staticmethod
    def logiclink_chat(user_input, history):
        if not user_input:
            return history, "No input provided"
        try:
            start = time.time()
            response = generate_response(user_input, history)
            elapsed = time.time() - start
            # Clean and format the response before appending it
            cleaned_response = re.sub(r'\*\(\d+\.\d+s\)\*', '', response).strip()
            response_with_time = f"{cleaned_response}\n\n*({elapsed:.2f}s)*"
            history.append({
                "role": "assistant",
                "content": response_with_time,
                "key": str(uuid.uuid4()),
                "avatar": None
            })
            return history, response_with_time
        except Exception as e:
            error_msg = (
                f"Generation failed: {str(e)}. "
                "Possible causes: insufficient memory, model incompatibility, or input issues."
            )
            history.append({
                "role": "assistant",
                "content": error_msg,
                "key": str(uuid.uuid4()),
                "avatar": None
            })
            return history, error_msg

    @staticmethod
    def clear_history(state_value):
        if state_value["conversation_id"]:
            # Only clear messages after system prompt
            current_history = state_value["conversation_contexts"][state_value["conversation_id"]]["history"]
            if len(current_history) > 0 and current_history[0]["role"] == "system":
                system_message = current_history[0]
                state_value["conversation_contexts"][state_value["conversation_id"]]["history"] = [system_message]
            else:
                state_value["conversation_contexts"][state_value["conversation_id"]]["history"] = []

            # Return the cleared history
            return (
                gr.update(value=state_value["conversation_contexts"][state_value["conversation_id"]]["history"]),
                gr.update(value=state_value),
                gr.update(value="")
            )
        return (
            gr.update(value=[]),
            gr.update(value=state_value),
            gr.update(value="")
        )

    @staticmethod
    def delete_conversation(state_value, conversation_key):
        # Keep a copy of the conversations before removal
        new_conversations = [conv for conv in state_value["conversations"] if conv["key"] != conversation_key]

        # Remove the conversation from the list
        state_value["conversations"] = new_conversations

        # Delete the conversation context
        if conversation_key in state_value["conversation_contexts"]:
            del state_value["conversation_contexts"][conversation_key]

        # If we're deleting the active conversation
        if state_value["conversation_id"] == conversation_key:
            state_value["conversation_id"] = ""
            return gr.update(items=new_conversations), gr.update(value=[]), gr.update(value=state_value)

        # If deleting another conversation, keep the current one displayed
        return (
            gr.update(items=new_conversations),
            gr.update(value=state_value["conversation_contexts"].get(
                state_value["conversation_id"], {"history": []}
            )["history"]),
            gr.update(value=state_value)
        )

# (The remainder of your Gradio UI code remains largely unchanged.)

css = """
:root {
--color-red: #ff4444;
--color-blue: #1e88e5;
--color-black: #000000;
--color-dark-gray: #121212;
}
.gradio-container { background: var(--color-black) !important; color: white !important; }
.gr-textbox textarea, .ms-gr-ant-input-textarea { background: var(--color-dark-gray) !important; border: 2px solid var(--color-blue) !important; color: white !important; }
.gr-chatbot { background: var(--color-dark-gray) !important; border: 2px solid var(--color-red) !important; }
.gr-textbox.output-textbox { background: var(--color-dark-gray) !important; border: 2px solid var(--color-red) !important; color: white !important; margin-bottom: 10px; }
.gr-chatbot .user { background: var(--color-blue) !important; border-color: var(--color-blue) !important; }
.gr-chatbot .bot { background: var(--color-dark-gray) !important; border: 1px solid var(--color-red) !important; }
.gr-button { background: var(--color-blue) !important; border-color: var(--color-blue) !important; }
.gr-chatbot .tool { background: var(--color-dark-gray) !important; border: 1px solid var(--color-red) !important; }
"""

with gr.Blocks(css=css, fill_width=True, title="LogicLinkV5") as demo:
    state = gr.State({
        "conversation_contexts": {},
        "conversations": [],
        "conversation_id": "",
    })
    with ms.Application(), antdx.XProvider(theme=DEFAULT_THEME, locale=DEFAULT_LOCALE), ms.AutoLoading():
        with antd.Row(gutter=[20, 20], wrap=False, elem_id="chatbot"):
            # Left Column
            with antd.Col(md=dict(flex="0 0 260px", span=24, order=0), span=0, order=1):
                with ms.Div(elem_classes="chatbot-conversations"):
                    with antd.Flex(vertical=True, gap="small", elem_style=dict(height="100%")):
                        Logo()
                        with antd.Button(color="primary", variant="filled", block=True, elem_classes="new-chat-btn") as new_chat_btn:
                            ms.Text(get_text("New Chat", "新建对话"))
                            with ms.Slot("icon"):
                                antd.Icon("PlusOutlined")
                        with antdx.Conversations(elem_classes="chatbot-conversations-list") as conversations:
                            with ms.Slot('menu.items'):
                                with antd.Menu.Item(label="Delete", key="delete", danger=True) as conversation_delete_menu_item:
                                    with ms.Slot("icon"):
                                        antd.Icon("DeleteOutlined")
            # Right Column
            with antd.Col(flex=1, elem_style=dict(height="100%")):
                with antd.Flex(vertical=True, gap="small", elem_classes="chatbot-chat"):
                    chatbot = pro.Chatbot(elem_classes="chatbot-chat-messages", height=600,
                                         welcome_config=welcome_config(), user_config=user_config(),
                                         bot_config=bot_config())
                    output_textbox = gr.Textbox(label="LatestOutputTextbox", lines=1,
                                              elem_classes="output-textbox", interactive=True)
                    with antdx.Suggestion(items=[]):
                        with ms.Slot("children"):
                            with antdx.Sender(placeholder="Type your message...", elem_classes="chat-input") as input:
                                with ms.Slot("prefix"):
                                    with antd.Flex(gap=4):
                                        with antd.Button(type="text", elem_classes="clear-btn") as clear_btn:
                                            with ms.Slot("icon"):
                                                antd.Icon("ClearOutlined")
    # Event Handlers
    input.submit(fn=Gradio_Events.add_message, inputs=[input, state],
                outputs=[input, chatbot, state]).then(
        fn=Gradio_Events.submit, inputs=[state],
        outputs=[chatbot, state, output_textbox]
    )
    new_chat_btn.click(fn=Gradio_Events.new_chat,
                     inputs=[state],
                     outputs=[conversations, chatbot, state, input],
                     queue=False)
    clear_btn.click(fn=Gradio_Events.clear_history, inputs=[state],
                   outputs=[chatbot, state, output_textbox])
    conversations.menu_click(
        fn=lambda state_value, e: (
            # If there's no payload, skip
            gr.skip() if (e is None or not isinstance(e, dict) or 'key' not in e._data['payload'][0] or 'menu_key' not in e._data['payload'][1])
            else (
                # Extract keys
                (lambda conv_key, action_key: (
                    # If "delete", remove that convo
                    Gradio_Events.delete_conversation(state_value, conv_key)
                    if action_key == "delete"
                    # If other action, do nothing
                    else (
                        gr.update(items=state_value["conversations"]),
                        gr.update(value=state_value["conversation_contexts"]
                        .get(state_value["conversation_id"], {"history": []})
                        ["history"]),
                        gr.update(value=state_value)
                    )
                ))(
                    e._data['payload'][0]['key'],
                    e._data['payload'][1]['key']
                )
            )
        ),
        inputs=[state],
        outputs=[conversations, chatbot, state],
        queue=False
    )

demo.queue().launch(share=True, debug=True)