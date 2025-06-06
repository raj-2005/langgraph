# OBJECTIVES

# Use different message types - HumanMessage and AIMessage
# Maintain a full conversation history using both message types
# Use Groq Api using LangChain's ChatGroq
# Create sophisticated conversation loop

#Main Goal : Create a form of memory for our Agent


import os
from typing import TypedDict,List,Union
from langchain_core.messages import HumanMessage , AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv

load_dotenv()

# there is 2 new inclusions in this chatbot compared to the previous agent bot
# one is union another is AIMessage

# define the state
class AgentState(TypedDict):
    messages:List[Union[HumanMessage,AIMessage]]

    # HumanMessage and AImessage are a type of
    # data types in langgraph , 
    # with union we say that the key messages
    # can contain list of either humanmessage or 
    # AImessage

    # messages_ai: List[AIMessage]
    # this is the naive way  

# initialise the llm
llm = ChatGroq(model="llama3-70b-8192")


# now lets build the node
def process(state:AgentState)->AgentState:
    """This node will solve the request you want"""
    response = llm.invoke(state["messages"])

    state['messages'].append(AIMessage(content=response.content))
    # what this line does here
    # response.content extracts only the content part of the response
    # the result of the API call of the llm
    # it gets converted to AI Message and gets appended to the 
    # state['message']
    print(f"\nAI: {response.content}")
    return state

# create the graph
graph=StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent = graph.compile()

# conversation history -> this is going to be our memory for the chat
conversation_history=[]

user_input=input("Enter: ")

while user_input!= "exit":
    conversation_history.append(HumanMessage(content=user_input))
    # user input is the human message

    result = agent.invoke({"messages":conversation_history})
    # we are invoking the agent (compiled graph) with the conversation history
    # the entire conversation history is sent , not just the current HumanMessage

    conversation_history=result["messages"]
    user_input=input("Enter: ")
    




