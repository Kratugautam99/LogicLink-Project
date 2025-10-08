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
- [📸 GUI Display](#-gui-display)
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

## 📸 GUI Display

---

### 💬 Full-Fledged Conversation
<p align="center">
  <img src="https://raw.githubusercontent.com/Kratugautam99/LogicLink-Project/main/Screenshots/LogicLinkFullFledgedConversation.png" alt="LogicLink Full Conversation" width="80%" />
</p>

LogicLink engaging in a complete dialogue, handling multiple turns seamlessly.  
This demonstrates its ability to maintain context, respond naturally, and adapt to user intent across an extended session.

---

### 🧑‍💻 Coding Response (Part 1)
<p align="center">
  <img src="https://raw.githubusercontent.com/Kratugautam99/LogicLink-Project/main/Screenshots/LogicLinkCodingResponse1.png" alt="LogicLink Coding Response 1" width="80%" />
</p>

LogicLink generating a structured coding solution.  
Notice how it explains the reasoning step-by-step, making the output not just correct but also **educational**.

---

### 🧑‍💻 Coding Response (Part 2)
<p align="center">
  <img src="https://raw.githubusercontent.com/Kratugautam99/LogicLink-Project/main/Screenshots/LogicLinkCodingResponse2.png" alt="LogicLink Coding Response 2" width="80%" />
</p>

A continuation of the coding workflow, where LogicLink refines and expands on its earlier solution.  
This shows its iterative reasoning ability — improving code quality when prompted.

---

### 🔑 Core Response
<p align="center">
  <img src="https://raw.githubusercontent.com/Kratugautam99/LogicLink-Project/main/Screenshots/LogicLinkCoreResponse.png" alt="LogicLink Core Response" width="80%" />
</p>

A snapshot of LogicLink delivering a **core logical explanation**.  
This highlights its strength in breaking down abstract queries into clear, actionable insights.

---

### ⚡ While Processing
<p align="center">
  <img src="https://raw.githubusercontent.com/Kratugautam99/LogicLink-Project/main/Screenshots/LogicLinkWhileProcessing.png" alt="LogicLink While Processing" width="80%" />
</p>

The system mid‑inference, showing its **real-time feedback loop**.  
This reassures users that LogicLink is actively working on their request.

---

### 🔄 With vs Without Latest Output Text Box
<p align="center">
  <img src="https://raw.githubusercontent.com/Kratugautam99/LogicLink-Project/main/Screenshots/LogicLinkwithLOTB.png" alt="LogicLink with LOTB" width="45%" />
  <img src="https://raw.githubusercontent.com/Kratugautam99/LogicLink-Project/main/Screenshots/LogicLinkwithoutLOTB.png" alt="LogicLink without LOTB" width="45%" />
</p>

A side‑by‑side comparison of LogicLink’s performance **with** and **without LOTB (Latest Output Text Box)**.  
The difference illustrates how LOTB enhances reasoning depth and response clarity.

---

### 📊 Bottom Section
<p align="center">
  <img src="https://raw.githubusercontent.com/Kratugautam99/LogicLink-Project/main/Screenshots/LogicLinkBottom.png" alt="LogicLink Bottom Section" width="80%" />
</p>

The footer view of the interface, where conversation summaries and quick actions are displayed.  
This ties the user experience together, making LogicLink feel like a polished, end‑to‑end assistant.

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
├── LogicLinkVersion5.ipynb
├── README.md
├── app.py
├── config.py
├── .gitattributes
├── requirements.txt
├── assets/
├── Documents/
├── Screenshots/
├── ui_components/
└── Different Versions of LogicLink/  (not expanded)
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
  <a href="https://huggingface.co/spaces/KraTUZen/LogicLink-Project-Space">HFT Space</a> | 
  <a href="https://www.gradio.app/">UI Framework</a>
</div>




