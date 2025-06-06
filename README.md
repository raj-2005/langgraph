# langgraph
# LangGraph Agent Tutorials

This repository demonstrates how to create intelligent agents using [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain) with the Groq LLM backend. The project progresses from a simple chatbot to a memory-enhanced chatbot and a robust ReAct agent capable of tool usage.

---

## 📂 Project Structure

- [simple_bot.py](./simple_bot.py)
- [chatbot_memory.py](./chatbot_memory.py)
- [react_agent.py](./react_agent.py)

Each script focuses on different learning objectives and gradually increases in complexity.

---

## 🔹 [simple_bot.py](./simple_bot.py)

### Objective
- Learn how to define a simple graph-based agent using LangGraph.
- Integrate a single node that calls a Groq-backed LLM using `ChatGroq`.
- Understand basic state management using `TypedDict` with a list of `HumanMessage` objects.

### Description
This script defines a minimal conversational bot using LangGraph. It accepts one user input and generates a single LLM response without maintaining memory. It illustrates:
- Setting up an LLM (`llama3-70b-8192` via Groq).
- Using LangGraph’s `StateGraph` to define the message flow.
- Returning the LLM’s response using `invoke`.

> ⚠️ Note: This bot has no memory—each interaction is stateless.

---

## 🔹 [chatbot_memory.py](./chatbot_memory.py)

### Objective
- Add memory using `HumanMessage` and `AIMessage` types.
- Maintain conversation history with the agent.
- Implement a loop-based chat session with full state tracking.

### Description
This script builds upon `simple_bot.py` by adding:
- Stateful memory using a conversation history list.
- A loop allowing users to converse with the agent until they type `"exit"`.
- `ChatGroq` as the LLM backend using the LLaMA3 model.

The conversation history is managed via a `Union` of `HumanMessage` and `AIMessage`, allowing the bot to contextually respond based on the full chat history.

---

## 🔹 [react_agent.py](./react_agent.py)

### Objective
- Create a ReAct agent using LangGraph.
- Add tool usage via LangChain tools.
- Use conditional graph edges and robust message handling.
- Test robustness with dynamic looping between agent and tool nodes.

### Description
This is the most advanced agent in the project. It:
- Implements multiple tools (`add`, `subtract`, `multiply`) decorated with `@tool`.
- Binds these tools to the LLM via `bind_tools`.
- Uses conditional branching (`should_continue`) to determine whether to call a tool or return a final answer.
- Demonstrates tool calling via `ToolMessage`, and state management using reducers (`add_messages`).
- Shows how LangGraph can loop between nodes to handle complex tasks (e.g., execute multiple tool calls and natural responses in sequence).

> 💡 This showcases how LangGraph supports the ReAct pattern (Reasoning + Acting) using a circular agent-tool-agent flow.

---

## 💡 Highlights

- ✅ Robust use of LangChain + LangGraph
- ✅ Groq LLaMA3 integration via `ChatGroq`
- ✅ Demonstration of stateless and stateful agents
- ✅ Tool calling with ReAct-style design
- ✅ Clear use of `StateGraph` with conditional logic

---

## 📌 Requirements

Install dependencies before running:

```bash
pip install langchain langgraph python-dotenv
pip install --upgrade langchain-groq
