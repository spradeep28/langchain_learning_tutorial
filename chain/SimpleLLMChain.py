import os

from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from dotenv import dotenv_values
env_config = dotenv_values(".env")

os.environ['OPENAI_API_KEY'] = env_config["OPENAI_API_KEY"]


def SimpleChainLLM(object, location):
    
    prompt = PromptTemplate(
    input_variables=["object", "location"],
    template="Suggest me a good {object}, located in {location}",
    )
    
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=prompt)
    
    print(chain.run({
        'object': object,
        'location': location
    }))
    
if __name__ == "__main__":
    
    print("Hello LangChain...!!!")
    SimpleChainLLM(object="restaurant",
                   location="Phoenix, USA")