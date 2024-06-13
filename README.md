# LLM with LangChain
Here I will explain the code to myself, so when I return to it later, I understand it faster!
___
## <span style="color: darkcyan;">Introduction to LangChain</span>
* LangChain is an OpenSource framework that allows developers working with AI to combine LLMs (like GPT-4) with external sources of computation and data.
* LLMs alone are often limited in their ability to understand the context, interact with the real world, or learn and adapt.
* LLMs have an impressive general knowledge but are limited to their training data.
* LangChain allows you to connect an LLM like GPT-4 to your own sources of data (data-aware)
* Using LangChain you can make your LLM application take actions (agentic-aware)

### LangChain Use Cases: 
* Chat Bots
* Question Answering Systems
* Summarization Tools

### LangChain main concepts: 
#### LangChain Components:
   1. **LLM Wrappers** (allow us to connect to and use LLMs like GPT-4 from the Hugging Face Hub)
   2. **Prompt Templates** (allow us to create dynamic prompts which are the input to the LLM)
   3. **Indexes** (allow us to extract relevant information for the LLMs)
   4. **Memory** (concept of storing and retrieving data in the process of a conversation) 
      * **Short Term Memory** (how to pass data in the context of a single conversation)
      * **Long Term Memory** (how to fetch and update information between conversations)
#### Chains
* Allow us to combine multiple components together to solve a specific task and build an entire LLM application
#### Agents
* Facilitate interaction between the LLM and external APIs. They play a crucial role in decision-making, determining which actins the LLM should undertake.
* Agents are enabling tools for LLMs 
* This process involves taking an action, observing the result, and then repeating the cycle until completion 

___
## <span style="color: orangered;">Requirements</span>
`requirements.txt` contains all the required libraries for the project. You can install all this libraries running this code in the terminal:
```py
pip install -r .\requirements.txt -q
```

To see the version of a library (e.g. langchain) run the following in the terminal:
```py
pip show langchain
```
Note: because of the popularity of langchain, it is updating very fast!

If you want to update a library to its latest version, run this in the terminal:
```py
pip install langchain --upgrade -q
```
 
## <span style="color: darkcyan;">API Keys</span>
#### How to get an `openai API Key`:
1. go to [openai platform website](https://platform.openai.com/) and sign up
2. go to Your profile
3. go to User API Keys
4. here, you can generate a new API key or invalidate an existing one 

#### How to get an `pinecone API Key`:
1. go to [Pincone](https://www.pinecone.io/) and sign up
2. go to API Keys and generate a new one
3. copy the value
4. for the environment go to [pinecone environment](https://docs.pinecone.io/guides/get-started/quickstart) and add the codes in the `.py` file:
    ```Py
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
    index_name = "docs-quickstart-index"
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    ```
> [!NOTE]
> Due to security reasons create your own API keys when running the program and put them in the .env file


## <span style="color: darkcyan;">Pinecone</span>
High performance, scalable, and distributed **vector store** for LLMs.


## <span style="color: darkcyan;">Python-dotenv</span>
`python-dotenv` is a module that allows you to specify environment variables, as key-value pairs, in a `.env` file within your python project directory. 

It is a convenient and secure way to load and use environment variables in your application.

We will save the API keys in the `.env` file.


### How to create a `.env` file:
1. open a text editor in the current directory and add these
2. `OPENAI_API_KEY=""` in which in the `""` will be the openai API key
3. `PINECONE_API_KEY=""` in which in the `""` will be the pinecone API key
4. Click on `Save As`, choose `Save as type: All Files`, `File Name: .env`


### Loading the environment variables:
```Py
import os
from dotenv import load_dotenv, find_dotenv

# loading the variables found in the .env file: 
# first argument of load_dotenv() is the directory of the .env file, or you can simply use the find_dotenv() as argument 
# second argument of load_dotenv() is override=True to override the value of the variable if you change it in .env
load_dotenv(find_dotenv(), override=True)
```

### Getting and Printing API key
```Py
os.environ.get('PINECONE_API_KEY')
print(os.environ.get('PINECONE_API_KEY'))
```
___

## <span style="color: darkcyan;">ChatModels: GPT-4</span>
ChatModels are a variation on classical language models which expose an interface where chat messages or conversations are the inputs and the outputs.

In the terminal run: `pip install -U langchain-openai`
#### Importing a schema for the `messages` schema:
```Py
from langchain.schema import(
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain_openai import ChatOpenAI
```

Look at this format of an [API call](https://platform.openai.com/docs/guides/text-generation/chat-completions-api):
1. system: this role helps set the behaviour of the assistant (in langchain: SystemMessage)
2. user: what we ask the assistant (in langchain: HumanMessage)
3. assistant: help store prior responses (in langchain: AIMessage)


#### Creating the `chat` object:
```Py
chat = ChatOpenAI(model_name='gpt-4o', temperature=0.7, max_tokens=1024)
```

#### Creating the `messages` list:
```Py
messages = [
    SystemMessage(content='You are a computer scientist and response only in German.'),
    HumanMessage(content='explain API key in one sentence')
]
output = chat.invoke(messages)
print(output.content)
```
___
## <span style="color: darkcyan;">Prompt Templates</span>
* **Prompt** refers to the input to the model
* **Prompt Templates** are a way to create dynamic prompts for LLMs that are more flexible and easier to use 
* A prompt template takes a piece of text and injects the user's input into that piece of text 

#### Importing the required classes:
```Py
from langchain_core.prompts import PromptTemplate
from langchain.schema import(
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain_openai import ChatOpenAI
```

#### Creating the dynamic prompt and `prompt` object:
```Py
template = ''' You are an experienced computer scientist.
Write a few sentences about {concept} in {language}'''

prompt = PromptTemplate(
    input_variables=['concept', 'language'],
    template=template
)
```

#### Initializing `chat` model:
```Py
chat = ChatOpenAI(model_name='gpt-4', temperature=0.7, max_tokens=1024)
```

#### Defining a function to generate output response using dynamic inputs
```Py
def get_response(concept, language):
    formatted_prompt = prompt.format(concept=concept, language=language)
    messages = [
        SystemMessage(content='You are an experienced computer scientist.'),
        HumanMessage(content=formatted_prompt)
    ]
    return chat.invoke(messages)

# Example usage
output = get_response('API key', 'English')
print(output.content)
```
___
## <span style="color: darkcyan;">Simple Chains</span>
* Chains allow us to combine multiple components to create a single and coherent application
#### Importing the required classes:
```Py
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence
```
#### Initializing `llm` model:
```Py
llm = ChatOpenAI(model_name='gpt-4', temperature=0.5)
```

#### Creating the dynamic prompt and `prompt` object:
```Py
template = ''' You are an experienced computer scientist.
Write a few sentences about {concept} in {language}'''

prompt = PromptTemplate(
    input_variables=['concept', 'language'],
    template=template
)
```

#### Create a runnable sequence with the `prompt` and the `llm` and generate output based on input variables:
```Py
sequence = RunnableSequence(prompt | llm)
output = sequence.invoke({'concept': 'API key', 'language': 'English'})
print(output.content)
```

___
## <span style="color: darkcyan;">Sequential Chains</span>
With **sequential chains**, you can make a series of calls to one or more LLMs. You can take the output from one chain and use it as the input to another chain.


There are two types of sequential chains:
1. SimpleSequentialChain
2. General form of sequential chain

### SimpleSequentialChain 
Represents a series of chains, where each individual chain has a single input and a single output, and the output of one step is used as input to the next.

#### Importing the required classes:
```Py
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, SimpleSequentialChain
```
#### Creating 2 chains:
```Py
# creating the first chain
llm1 = ChatOpenAI(model_name='gpt-4', temperature=0.7)
template = ''' You are an experienced computer scientist.
Write a function that implements the concept of {concept}'''

prompt1 = PromptTemplate(
    input_variables=['concept'],
    template=template
)
chain1 = LLMChain(llm=llm1, prompt=prompt1)

# creating the second chain
llm2 = ChatOpenAI(model_name='gpt-4', temperature=0.7)
template = ''' Given the python function {function}, describe it as detailed as possible'''

prompt2 = PromptTemplate(
    input_variables=['function'],
    template=template
)
chain2 = LLMChain(llm=llm2, prompt=prompt2)
```

#### Combining 2 chains using `SimpleSequentialChain`:
```Py
overall_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)
output = overall_chain.invoke('linear regression')
```
___
## <span style="color: darkcyan;">LangChain Agents</span>
LLMs cannot give accurate answers to complicated calculations! Also, LLMs are out of date and can give old information about newly asked questions. 
Solution: LangChain Agents

#### Importing the required classes:
```Py
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_openai import ChatOpenAI
```

#### Creating `llm` model:
```Py
llm = ChatOpenAI(model_name='gpt-4', temperature=0.7)
```

#### Creating `agent_executor`:
```Py
agent_executor = create_python_agent(
    llm=llm,
    tool=PythonREPLTool(),
    verbose=True
)
agent_executor.invoke('Calculate 1.7**5.2')
```
* Creating a Python `agent_executor` using `ChatOpenAI` `llm` allows us to have the language model execute Python code.
* **The `tool` argument:** tools are essentially functions that agents can use to interact with the outside world.






















