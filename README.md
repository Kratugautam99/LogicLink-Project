# LogicLink: Version 5

**LogicLink** is a conversational AI chatbot developed by **Kratu Gautam** (AIML Engineer). Powered by the **TinyLlama-1.1B-Chat-v1.0** model, it provides an interactive interface for engaging conversations, query resolution, and task assistance. Version 5 features streaming responses, conversation management, and a sleek GUI.

<div align="center">
  <img src="https://raw.githubusercontent.com/Kratugautam99/LogicLink-Project/main/assets/FullIcon.jpg" 
       alt="LogicLink Logo" 
       width="500">
</div>


---
## 🔍 Topic Index
- [✨ Key Features](#-key-features)
- [🛠️ Installation](#-installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
  - [Directory Structure](#directory-structure)
- [💬 Usage](#-usage)
- [⚙️ Technical Architecture](#-technical-architecture)
  - [Model Configuration](#model-configuration)
  - [Key Components](#key-components)
- [🧪 Troubleshooting Guide](#-troubleshooting-guide)
- [🚀 Future Roadmap](#-future-roadmap)
- [📜 License](#-license)

--- 
## ✨ Key Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| **🤖 Conversational AI** | TinyLlama-1.1B-Chat-v1.0 powered responses | Natural, engaging dialogue |
| **⚡ Streaming Responses** | Real-time token generation with `TextIteratorStreamer` | Smooth user experience |
| **🎨 Customizable GUI** | Red/blue/black theme with Gradio & ModelScope Studio | Professional interface |
| **🗂️ Conversation Management** | New chat, clear history, delete conversations | Full control over interactions |
| **⏱️ Single Time Stamp** | Regex-cleaned response timing `*(4.50s)*` | Consistent performance metrics |
| **🚀 CUDA Support** | Automatic GPU detection with CPU fallback | Optimized performance |
| **🛡️ Error Handling** | Graceful failure for memory/input issues | Robust user experience |

---
<a id="-installation"></a>
## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (recommended)
- Dependencies:
  ```bash
  pip install gradio torch transformers modelscope-studio
  ```
  
---
### Setup
1. Clone repository:
   ```bash
   git clone https://github.com/Kratugautam99/LogicLink-Project.git
   cd LogicLink-Project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run application:
   ```bash
   python app.py
   ```

---
### Directory Structure
```
LogicLink-Project/
├── app.py              # Main application
├── config.py           # GUI configuration
├── ui_components/
│   ├── logo.py         # Logo component
│   └── settings_header.py # Settings UI
└── requirements.txt    # Dependencies
```

---
## 💬 Usage

```python
# Sample interaction flow
user >> "Who are you?"
LogicLink >> "I'm LogicLink V5, created by Kratu Gautam. How can I assist you today? *(4.50s)*"
```

1. **Interface Controls**:
   - 💬 Input field: Type queries
   - ➕ New Chat: Start fresh conversation
   - 🧹 Clear History: Reset current chat
   - 🗑️ Delete: Remove conversations from sidebar

2. **Performance Metrics**:
   - ⏱️ Response time: 3-5s (GPU), 5-8s (CPU)
   - 💾 RAM usage: 2-3GB (CPU), ~1.5GB (GPU)

---
<a id="-technical-architecture"></a>
<h2>
  <span style="float: right; margin-left:35px;">
    <img src="https://raw.githubusercontent.com/Kratugautam99/LogicLink-Project/main/assets/lll.jpg" alt="icon" width="35" />
  </span>
      Technical Architecture
</h2>

### Model Configuration
```python
# Core model parameters
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16 if cuda else torch.float32
)

# Generation settings
generation_kwargs = {
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95,
    "num_beams": 1
}
```
---
### Key Components
1. **Prompt Engineering**:
   ```
   <|system|>You are LogicLink V5 created by Kratu Gautam</s>
   <|user|>{user_input}</s>
   <|assistant|>
   ```
   
2. **Streaming Pipeline**:
   ```mermaid
   graph LR
   A[User Input] --> B(Tokenizer)
   B --> C{TextIteratorStreamer}
   C --> D[Model Generation]
   D --> E[Real-time Output]
   E --> F[Regex Cleaner]
   F --> G[Timestamp Append]
   ```

3. **GUI Components**:
   - `pro.Chatbot`: Conversation display
   - `antdx.Sender`: Input field
   - `antdx.Conversations`: Sidebar manager
   - `antd.Button`: Action controls

---
## 🧪 Troubleshooting Guide

| Issue | Solution |
|-------|----------|
| Double timestamps | Verify regex: `re.sub(r'\*\(\d+\.\d+s\)\*', '', response)` |
| Slow responses | Enable CUDA, reduce `max_new_tokens` to 512 |
| GUI rendering issues | Update packages: `pip install --upgrade gradio modelscope-studio` |
| Delete button failure | Check `menu_click` event binding in JS |
| Model loading errors | Validate RAM ≥3GB, test with minimal example |

**Minimal Test Script**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
inputs = tokenizer(["Test input"], return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=10)
print(tokenizer.decode(outputs[0]))
```

---
## 🚀 Future Roadmap
- **Persistent Storage**: SQLite conversation history
- **Multimodal Support**: Image/text inputs
- **Enhanced Prompting**: Context-aware responses
- **Deployment Options**: Docker containerization
- **Performance**: Quantization for CPU optimization

---
## 📜 License
MIT License - See [LICENSE](https://github.com/Kratugautam99/LogicLink-Project/LICENSE)

---

<div align="center">
  Developed with 🧠 by <b>Kratu Gautam</b> | AIML Engineer<br>
  <a href="https://github.com/Kratugautam99">GitHub</a> | 
  <a href="https://huggingface.co/TinyLlama">Model Source</a> | 
  <a href="https://modelscope.cn/studios">UI Framework</a>
</div>




