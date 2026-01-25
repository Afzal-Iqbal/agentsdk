import os
import sys
from openai import AsyncOpenAI
from dotenv import load_dotenv
from agents import Agent, Runner, set_default_openai_client

load_dotenv()

# Configure the client for Groq
client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Set default client and disable tracing (Groq keys don't work with OpenAI tracing)
set_default_openai_client(client, use_for_tracing=False)

def main():
    # Initial Configuration
    initial_model = "llama-3.3-70b-versatile"
    fallback_model = "llama-3.1-8b-instant"
    prompt_limit_for_switch = 10
    memory_keep_turns = 3  # Keep last 3 prompts (turns)
    
    agent = Agent(
        name="Tutor",
        instructions="Explain things like a friendly teacher.",
        model=initial_model,
    )

    memory_list = []
    summary = ""
    prompt_count = 0

    print("--- GroqTutor Agent Started ---")
    print(f"Model: {agent.model}")
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        try:
            user_input = input("\nUser: ").strip()
        except EOFError:
            break

        if user_input.lower() in ["exit", "quit"]:
            print("Task ended. Clearing memory...")
            memory_list.clear()
            summary = ""
            break

        if not user_input:
            continue

        prompt_count += 1

        # Model Switching Logic
        if prompt_count > prompt_limit_for_switch and agent.model != fallback_model:
            print(f"\n[System] Switching model to {fallback_model} (Prompt count: {prompt_count})")
            agent.model = fallback_model

        # Construct Context
        context_str = ""
        if summary:
            context_str += f"PREVIOUS CONVERSATION SUMMARY:\n{summary}\n\n"
        
        if memory_list:
            context_str += "RECENT CONVERSATION:\n"
            for item in memory_list:
                role = "User" if item['role'] == "user" else "Tutor"
                context_str += f"{role}: {item['content']}\n"
        
        # Combine context with current user input
        full_input = context_str + f"\nUser: {user_input}"

        # Run the agent
        print("\nTutor is thinking...")
        result = Runner.run_sync(agent, input=full_input)
        response_text = str(result.final_output)
        
        print(f"Tutor: {response_text}")

        # Update Memory
        memory_list.append({"role": "user", "content": user_input})
        memory_list.append({"role": "assistant", "content": response_text})

        # Summarization Logic (Maintain only last 3 turns = 6 items)
        if len(memory_list) > memory_keep_turns * 2:
            # Separate items to summarize and items to keep
            items_to_summarize = memory_list[:- (memory_keep_turns * 2)]
            memory_list = memory_list[- (memory_keep_turns * 2):]

            # Prepare text for summarization
            text_chunk = "\n".join([f"{i['role']}: {i['content']}" for i in items_to_summarize])
            
            summary_prompt = (
                f"Update the following summary with the new conversation details concisely.\n\n"
                f"Current Summary: {summary}\n\n"
                f"New Conversation:\n{text_chunk}"
            )

            # Run summarization using a temporary agent (uses current model)
            # We use a fresh agent to strictly focus on summarization
            summarizer = Agent(
                name="Summarizer",
                instructions="You are an expert summarizer. Update the summary concisely.",
                model=agent.model
            )
            
            summary_result = Runner.run_sync(summarizer, input=summary_prompt)
            summary = str(summary_result.final_output)
            # print(f"\n[System] Memory Summarized. Summary length: {len(summary)} chars")

if __name__ == "__main__":
    main()