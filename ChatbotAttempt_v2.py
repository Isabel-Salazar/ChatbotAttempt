import os
import transformers
import chainlit as cl
import langchain
from langchain.chains import LLMChain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from getpass import getpass

HUGGINGFACEHUB_API_TOKEN = getpass()
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN

model_id = "gpt2-medium"
conv_model = HuggingFacePipeline.from_model_id(repo_id=model_id,
                                  task="text-generation",
                                  api_token=os.environ['HUGGINGFACEHUB_API_TOKEN'],
                                  model_id=model_id,
                                  model_type="text-generation",
                                  max_new_tokens=250,
                                  temperature=0.8,
                                  top_p=0.9,
                                  top_k=50,
                                  verbose=True)

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
    llm_chain.invoke("Hello")
    res = await llm_chain.ainvoke(message)
    llm_chain.batch([1, 2, 3])
    await llm_chain.abatch([1, 2, 3])

    #await cl.Message(content=res["text"]).send()