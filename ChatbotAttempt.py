import os
import chainlit as cl
import langchain
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from getpass import getpass

HUGGINGFACEHUB_API_TOKEN = getpass()
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN

model_id = "gpt2-medium"
conv_model = HuggingFaceEndpoint(huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN'],
                            repo_id=model_id,
                            temperature=0.8,
                            max_new_tokens=250)

template = """You are a helpful AI assistant that makes stories by completing the query provided by the user

{query}
"""

@cl.on_chat_start
async def main():
    prompt = PromptTemplate(template=template, input_variables=['query'])
    #conv_chain = langchain.chains.LLMChain(llm=conv_model, prompt=prompt, verbose=True)
    llm = conv_model
    conv_chain = prompt | llm

    cl.user_session.set("llm_chain", conv_chain)

@cl.on_message
async def main(message: str):
    llm_chain = cl.user_session.get("llm_chain")
    res = await llm_chain.ainvoke(message)

    await cl.Message(content=res["text"]).send()