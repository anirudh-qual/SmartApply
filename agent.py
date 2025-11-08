# A conceptual example of your new agent.py
from google.adk.agents import LlmAgent, BaseAgent
from google.adk.tools import tool

# This is a "tool" the agent can use.
@tool
def get_resume_context(application_id: str) -> str:
    """Gets the user's resume highlights based on their application ID."""
    # In a real app, you would fetch this from your database
    # based on the application_id passed during the WebSocket connection.
    if application_id == "app_123":
        return "The user is a Java developer with 5 years of experience at Enphase Energy."
    return "The user's resume is not available."

def create_interview_agent() -> BaseAgent:
    """Builds and returns your interviewer agent."""
    
    # The LlmAgent is the core "brain" powered by Gemini.
    return LlmAgent(
        # This is the agent's core instruction, which it will always follow.
        instructions="""
        You are a friendly, professional human interviewer.
        Your goal is to have a natural, free-flowing conversation.
        
        1. Start by greeting the user and introducing yourself.
        2. Use the 'get_resume_context' tool to understand the user's background.
        3. Ask a relevant opening question based on their resume.
        4. Ask natural follow-up questions based on their answers.
        5. Do NOT sound like a robot. Be conversational.
        6. When the interview is over, say "Thank you, that's all the questions I have."
        """,
        # This tells the agent what tools it is allowed to use.
        tools=[get_resume_context]
    )