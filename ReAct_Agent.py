# OBJECTIVES

# Learn how to create tools in LangGraph
# How to create a ReAct Graph
# Work with different types of messages such as ToolMessages
# Test out Robustness of our graph


# Main goal =>  create a robust ReAct Agent

# Imports
from typing import Annotated,Sequence,TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph,START,END
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

# Annotated -> provides additional context without affecting the type of the key
# Sequence -> To automatically handle the state updates for sequences 
#             such as by adding new messages to a chat history

# ToolMessage : passes the data back to the llm after it calls a tool
#               such as the content and the tool_call_id

# SystemMessage : Message for providing instructions to the LLM

# # BaseMessage : The foundation class for all message types in Langgraph

# add_messages : it is a reducer function
# it is a rule that controls how updates from nodes are combined with the existing state.
# Tells us how to merge new data into the current state

# without reducer , updates would have replaced the existing value entirely


# building the state
class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage], add_messages]
    # Sequence[BaseMessage] is the datatype
    # add_messages : is the reducing function(metadata)

# lets create the first tool
# we use the decorator , the decorator tells python that this function
# is special

@tool
def add(a:int , b:int):
    """this is an addition function that adds 2 numbers together"""
    return a+b

# lets add some more tools
@tool
def subtract(a:int , b:int):
    """subtraction function"""
    return a-b

@tool
def multiply(a:int , b:int):
    "this is the multiplication function"
    return a*b

# how can we infuse these tools to a llm 

tools=[add,subtract,multiply] # this is used to store all the tools




# how can we tell the llm that these are the tools that u need to use
# we will use in built fcn called (bind_tools)
# this bind_tools() fcn will get the list of tools (list)

model=ChatGroq(model="llama3-70b-8192").bind_tools(tools)


# now create the node that acts as the agent 
def model_call(state:AgentState)-> AgentState:
    # response = model.invoke(["You are my AI assistant , please answer my query to the best of your ability"])
    # the string we inputted to invoke the llm is the system message

    # another method
    System_prompt=SystemMessage(content=
                                "You are my AI assistant , please answer my query to the best of your ability. "
                                )
    # now pass in this System_prompt to while invoking the llm
    response = model.invoke([System_prompt] + state["messages"])
    return {"messages":[response]}
    # this is the another way of writing or updating the messages key of the state
    # this is possible because the add_messages (reducer fcn) will take care of the updation


# now let us do with the conditional edge according to the graph we want to build

def should_continue(state: AgentState)->AgentState:
    messages = state['messages']
    last_message=messages[-1]

    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    # as end and continue are edge names that we define later in the graph
    # we will see through the last messages and we will see if there is any more
    # tools needed to run , if there are we'll go the toolnode , if there is non
    # we'll just end

# now we'll create the graph
graph=StateGraph(AgentState)
graph.add_node("our_agent",model_call)

# now lets define the tool node
tool_node = ToolNode(tools=tools)
graph.add_node("tools",tool_node)
# the tool node is a node that contains all the tools 

# set the entry point and point it to our agent
graph.set_entry_point("our_agent")

# now let us establish the conditional edge
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue":"tools",
        "end":END
    }
)

# since we need a circular connection from our tool to the agent , we establish
# another edge even though the conditional edge is present in the other dirn

graph.add_edge("tools","our_agent")


# compile the graph
app=graph.compile()

# lets define a helper function for printing the status 
def print_stream(stream):
    for s in stream:
        message=s["messages"][-1]
        if isinstance(message , tuple):
            print(message)
        else:
            message.pretty_print()

inputs={"messages":[("user","Add 21+999 and multiply the result by 3 , Also tell me a joke")]}
print_stream(app.stream(inputs,stream_mode="values"))

# as we can see even if we pass something random like other than the tool calls
# it generates the output on that context which is the robustness of langgraph

# it happends because , we have the loop that iterates over the agent and the toolnode
# we creates , so even if the tool node is completed ,the execution goes to the agent 
# and it sees that there is still something in the input , that needs to be
# resolved , so it executes that also and then only it ends
