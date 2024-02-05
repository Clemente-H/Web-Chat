# README for Together AI & Langchain: Search and Image Description Agent

This project integrates Together AI with Langchain to create a powerful Streamlit-based web application that provides users with the capability to search for information and describe images using a conversational interface. It leverages the advanced capabilities of Langchain's structured chat agents, image captioning using BLIP (Berkeley Language and Image Pretraining) models, and the Together language model for processing and responding to user queries.

## Features

- **Information Search**: Utilizes Tavily Search Results tool to fetch and display information based on user queries.
- **Image Description**: Incorporates a BLIP model to generate descriptions for images provided by users through URLs.
- **Conversational Interface**: Offers a chat-based interface where users can ask questions or request image descriptions.
- **Streamlit Integration**: Uses Streamlit for an interactive and user-friendly web application.

## Requirements

- Python 3.7+
- Streamlit
- Langchain
- `transformers` library
- `requests`
- `Pillow` for image processing

## Installation

1. **Clone the repository**:

   ```
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Set up a virtual environment** (optional but recommended):

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies**:

   ```
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:

   Create a `.env` file in the root directory of the project and add the following variables:

   ```
   TOGETHER_API_KEY=<Your Together API Key>
   TAVILY_API_KEY=<Your Tavily API Key>
   ```

   Ensure you replace `<Your Together API Key>` and `<Your Tavily API Key>` with your actual API keys.

## Usage

To run the application, execute:

```
streamlit run app.py
```

Navigate to the URL provided by Streamlit (usually `http://localhost:8501`) in your web browser to start using the application.

## How It Works

- The application initializes the necessary models and tools on startup.
- Users can interact with the application via a chat interface provided by Streamlit.
- For information search, users can enter queries which are processed by the Tavily search tool integrated within a structured Langchain agent.
- For image descriptions, users can provide image URLs. The application uses the BLIP model to generate and display a caption for the image.
- The conversation history and responses are managed by Streamlit components and Langchain's memory and callback mechanisms.

## Contributing

Contributions to improve the application are welcome. Please feel free to fork the repository, make changes, and submit pull requests.

## License

Please include your licensing information here.

---
