# LogicLink: Version 5

LogicLink is a conversational AI chatbot developed by **Kratu Gautam**, an **AIML Engineer**. Powered by the **TinyLlama-1.1B-Chat-v1.0** model, LogicLink provides an interactive and user-friendly interface for engaging conversations, answering queries, and assisting with tasks like planning, writing, and more. Version 5 introduces a sleek GUI, streaming responses, and enhanced features like conversation management.

## Features

- **Conversational AI**: Built on TinyLlama-1.1B-Chat-v1.0, LogicLink delivers natural and engaging responses to a wide range of user queries.
- **Streaming Responses**: Utilizes `TextIteratorStreamer` for real-time response generation, providing a smooth user experience.
- **Customizable GUI**: Features a modern interface with a red/blue/black theme, powered by Gradio and ModelScope Studio components (`pro.Chatbot`, `antdx.Sender`).
- **Conversation Management**:
  - **New Chat**: Start fresh conversations with a dedicated button.
  - **Clear History**: Reset the current conversation’s history.
  - **Delete Conversations**: Remove individual conversations from the conversation list.
- **Single Time Stamp**: Responses include a single processing time stamp (e.g., `*(4.50s)*`), fixed to avoid duplication.
- **CUDA Support**: Optimizes performance on GPU-enabled systems, with fallback to CPU.
- **Error Handling**: Gracefully handles issues like memory shortages or invalid inputs, displaying user-friendly error messages.

## Installation

### Prerequisites

- Python 3.8+
- CUDA-enabled GPU (optional, for faster processing)
- Dependencies:

  ```bash
  pip install gradio torch transformers modelscope-studio
  ```

### Setup

1. **Clone the Repository**:

   ```bash
   git clone Kratugautam99/LogicLink-Project.git
   cd LogicLink-Project
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Directory Structure**: Ensure the following files are present:

   - `app.py`: Main application script.
   - `config.py`: Configuration for GUI components (ensure `DEFAULT_LOCALE`, `DEFAULT_THEME`, `get_text`, `user_config`, `bot_config`, `welcome_config` are defined).
   - `ui_components/logo.py`: Logo component for the GUI.
   - `ui_components/settings_header.py`: Settings header component.

4. **Run the Application**:

   ```bash
   python app.py
   ```

   This launches a web interface via Gradio, providing a public URL (e.g., `https://...gradio.live`) if `share=True`.

## Usage

1. **Launch the Chatbot**:

   - Run `app.py` in a Jupyter notebook, Colab, or terminal.
   - Access the web interface through the provided URL.

2. **Interact with LogicLink**:

   - **Input Queries**: Type questions or tasks in the input field (e.g., "Tell me about Pakistan" or "Who are you?").
   - **Manage Conversations**:
     - Click **New Chat** to start a new conversation.
     - Click **Clear History** to reset the current conversation.
     - Click the **Delete** menu item in the conversation list to remove a conversation.

3. **Example Interaction**:

   - **Input**: "Who are you?"
   - **Output**:

     ```
     I'm LogicLink, Version 5, created by Kratu Gautam, an AIML Engineer. I'm here to help with your questions, so what's up?
     *(4.50s)*
     ```
   - **Input**: "Explain quantum physics briefly"
   - **Output**: A concise explanation of quantum physics, followed by `*(X.XXs)*`.

4. **Performance**:

   - **Response Time**: \~3–5 seconds per query (faster with CUDA).
   - **RAM Usage**: \~2–3 GB on CPU, lower on GPU.

## Technical Details

### Model Architecture

- **Base Model**: TinyLlama-1.1B-Chat-v1.0, a lightweight transformer-based language model with 1.1 billion parameters, optimized for chat applications.
- **Framework**: PyTorch with the Transformers library from Hugging Face.
- **Tokenizer**: `AutoTokenizer` configured with left-padding and EOS token handling to ensure proper input formatting for chat sequences.
- **Response Generation**:
  - Leverages `AutoModelForCausalLM` for next-token prediction.
  - Implements streaming with `TextIteratorStreamer` to output tokens in real-time, enhancing user experience.
  - Uses a custom `StopOnTokens` stopping criterion to halt generation at specific tokens (e.g., token ID 2), preventing unnecessary output.
- **Generation Parameters**:
  - `max_new_tokens=1024`: Limits response length to 1024 tokens.
  - `temperature=0.7`: Balances creativity and coherence in responses.
  - `top_k=50`: Considers the top 50 probable tokens for sampling.
  - `top_p=0.95`: Applies nucleus sampling to focus on the top 95% probability mass.
  - `num_beams=1`: Uses greedy decoding for deterministic output.

### Implementation Specifics

- **Prompt Engineering**:
  - The model is instructed via a system prompt:

    ```
    You are LogicLink, Version 5, created by Kratu Gautam, an AIML Engineer. Respond to the following user input: {user_input}
    ```
  - Conversation history is formatted with `<|user|>` and `<|assistant|>` tags, separated by `</s>`, to maintain context.
- **Threading**: Response generation runs in a separate thread using Python’s `Thread` module to prevent blocking the Gradio interface.
- **Time Stamp Handling**:
  - A regex (`re.sub(r'\*\(\d+\.\d+s\)\*', '', response)`) removes duplicate time stamps, ensuring each response ends with a single `*(X.XXs)*`.
- **Error Handling**:
  - Catches exceptions (e.g., memory errors, model incompatibilities) and appends user-friendly messages to the conversation history.
  - Example: `Generation failed: insufficient memory. Possible causes: ...`

### GUI

- **Framework**: Gradio integrated with ModelScope Studio components for a professional-grade interface.
- **Components**:
  - `pro.Chatbot`: Renders conversation history with distinct user (blue bubbles) and assistant (dark gray with red borders) messages.
  - `antdx.Sender`: Provides an input field with a clear button for user queries.
  - `antdx.Conversations`: A sidebar for managing multiple conversations, with a context menu for deletion.
  - `antd.Button`: Implements the "New Chat" button and other interactive elements.
- **Styling**: Custom CSS defines a red/blue/black theme:
  - User messages: Blue background for visibility.
  - Assistant messages: Dark gray with red borders for contrast.
  - Buttons: Blue with hover effects for interactivity.
- **Layout**: Uses `antd.Row` and `antd.Col` for responsive design, with a fixed 260px sidebar and flexible chat area.

### Performance Optimization

- **CUDA Support**: Automatically detects CUDA-enabled GPUs via `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`, reducing response times to \~3 seconds on GPU compared to \~5 seconds on CPU.
- **Memory Efficiency**: TinyLlama’s 1.1B parameters require \~2–3 GB RAM on CPU, making it suitable for consumer hardware.
- **Threaded Generation**: Offloads model inference to a separate thread, ensuring the GUI remains responsive during processing.

### Key Fixes

- **Single Time Stamp**: Resolved duplicate time stamps using regex to clean responses before appending `*(X.XXs)*`.
- **Delete Functionality**: Fixed `AntdXConversations` event handling by replacing `select` with `menu_click`, ensuring reliable conversation deletion.
- **Metadata**: Embedded model identity in the prompt to consistently identify as LogicLink V5 by Kratu Gautam.

## Troubleshooting

- **Double Time Stamps**:
  - If responses show multiple `*(X.XXs)*`, verify the regex in `logiclink_chat`.
  - Test with inputs like "Tell me about Pakistan" and share the output.
- **Slow Responses**:
  - Use a CUDA-enabled GPU for faster processing.
  - Reduce `max_new_tokens` to 512 if needed.
  - Check RAM usage: `!free -h` in Colab.
- **GUI Issues**:
  - Ensure `config.py` and `ui_components/` are correctly configured.
  - Update dependencies: `pip install --force-reinstall gradio modelscope-studio`.
- **Delete Button Not Working**:
  - Verify the `menu_click` event handler and JavaScript snippet.
  - Share any error messages or tracebacks.
- **Model Errors**:
  - Check for sufficient RAM (\~2–3 GB) and compatible PyTorch/Transformers versions.
  - Run a test generation:

    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    inputs = tokenizer(["Hello"], return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=10)
    print(tokenizer.decode(outputs[0]))
    ```

## Future Improvements

- Add a welcome message displaying LogicLink’s identity via `welcome_config()`.
- Enhance prompt engineering for more context-aware responses.
- Implement persistent storage for conversation history using a database or file system.
- Add support for multimodal inputs (e.g., images) to expand functionality.
- Optimize tokenization and generation for lower latency on CPU.

## Credits

- **Developer**: Kratu Gautam, AIML Engineer
- **Dependencies**:
  - TinyLlama-1.1B-Chat-v1.0 (Hugging Face)
  - Gradio
  - PyTorch
  - Transformers
  - ModelScope Studio
- **Inspiration**: Built to provide an accessible and interactive AI chatbot for students and enthusiasts.

## License

MIT License. See `LICENSE` for details.

---

**LogicLink V5** is a project by Kratu Gautam, showcasing the power of AI in creating intuitive conversational tools. Contributions and feedback are welcome!