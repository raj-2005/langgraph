# OBJECTIVES

# Define state structure with a list of human message objects
# initialize gpt-4o model using LangChain's chatOpenAI
# sending and handling different types of messages
# building and compiling the graph of the agent

# main goal : How to integrate LLM's in our graphs


# Human Message
# it is the message prompt given by humans(us) to the AI

# Import Libraries

from typing import TypedDict,List
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq # in video they used openAI , but we use Groq
from langgraph.graph import StateGraph , START,END
from dotenv import load_dotenv # used to store secret stuff like API keys

load_dotenv()

# this will load our API from environment variables

class AgentState(TypedDict):
    messages:List[HumanMessage] # since our objective is to get list of human messages

# now we will initialise the LLM
llm = ChatGroq(model="llama3-70b-8192")

# define the node
def process(state:AgentState)->AgentState:
    # how we'll call the llm 
    # to run the llm we use invoke in langchain & langgraph
    response = llm.invoke(state['messages']) # the invoke needs to get inputs which will be the messages from the user
    print(f"\nAI: {response.content}")
    return state


# now we need to create a graph

graph = StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)

agent = graph.compile()

user_input = input("Enter: ")
agent.invoke({"messages": [HumanMessage(content=user_input)]})


# this bot does not have the memory capabality , as we are calling the 
# api everytime we want to generate response