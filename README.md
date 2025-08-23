# Book Library Chatbot

A Streamlit-based chatbot for a book library that answers questions using content from a text file (`data.txt`).  
Built with **LangChain**, **OpenAI embeddings**, and **FAISS** for Arabic and multilingual support.

---

## Features

- Answers questions **only** based on the content of `data.txt`.
- Supports Arabic language queries.
- Interactive chat interface using **Streamlit** and **streamlit-chat**.
- Uses vector-based retrieval (FAISS + OpenAI Embeddings) for accurate responses.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/SamehDheir/book-library-chatbot-phyton.git
cd book-library-chatbot-phyton
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
# Activate on Windows
venv\Scripts\activate
# Activate on Mac/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Setup

1. Place your book data in a file named `data.txt` in the project root.
2. Add your OpenAI API key in `chatbot.py` or set it as an environment variable:

```python
import os
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
```

---

## Usage

Run the Streamlit app:

```bash
streamlit run chatbot.py
```

- Type your questions in the input box.
- The chatbot will respond **only** based on the content of `data.txt`.

---

## File Structure

```
book-library-chatbot/
│
├─ chatbot.py        # Main Streamlit app
├─ data.txt          # Book library content
├─ requirements.txt  # Python dependencies
└─ README.md         # This file
```

---

## Dependencies

- Python 3.10+
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [OpenAI](https://openai.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [streamlit-chat](https://github.com/AI-Yash/st-chat)

---

## License

This project is licensed under the MIT License.

