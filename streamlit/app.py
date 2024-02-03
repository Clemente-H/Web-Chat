import os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_together import Together
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent, create_structured_chat_agent
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.tools import tool
from langchain.pydantic_v1 import BaseModel, Field
import requests
from PIL import Image


import streamlit as st
st.set_page_config(page_title="LangChain: Chat with search", page_icon="🦜")
st.title("🦜 LangChain: Chat with search")

load_dotenv()
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

class ImageCaptionerInput(BaseModel):
    image_url: str = Field(description="URL of the image that is to be described")

@tool("image_captioner", return_direct=True, args_schema=ImageCaptionerInput)
def image_captioner(image_url: str) -> str:
    """Provides information about the image"""
    raw_image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
    inputs = blip_processor(raw_image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)

TAVILY_API_KEY=os.getenv('TAVILY_API_KEY')
search = TavilySearchResults(max_results=1)


prompt = hub.pull("hwchase17/structured-chat-agent")
prompt[0].prompt.template = prompt[0].prompt.template[0:394] + "```\n\n\nFor example, if you want to use a tool to get an image's caption, your $JSON_BLOB might look like this:\n\n```\n{{\n    'action': 'image_captioner_json', \n    'action_input': {{'query': 'images url'}}\n}}```" + "```\n\n\nIf you want to use a tool to search for information, your $JSON_BLOB might look like this:\n\n```\n{{\n    'action': 'tavily_search_results_json', \n    'action_input': {{'query': 'Example search query'}}\n}}```"  +prompt[0].prompt.template[394:]

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)
if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")
    st.session_state.steps = {}

avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                st.write(step[0].log)
                st.write(step[1])
        st.write(msg.content)

if instruction := st.chat_input(placeholder="Escribir aqui :D"):
    st.chat_message("user").write(instruction)

    llm = Together(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0,
        max_tokens=256,
        top_k=1,
        together_api_key=os.getenv('TOGETHER_API_KEY')
    )

    tools = [image_captioner, search]
    agent = create_structured_chat_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, return_intermediate_steps = True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        cfg = RunnableConfig()
        cfg["callbacks"] = [st_cb]
        response = agent_executor.invoke({"input": instruction},cfg)
        #response = executor.invoke(prompt + instruction, cfg)
        st.write(response["output"])
        #st.chat_message("assistant").write(response["output"])
        st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]