import os
from crewai import Agent, Task, Crew, LLM
from crewai_tools import (
    YoutubeVideoSearchTool,
    FileReadTool,
    FileWriteTool
)

# Set up API keys
# os.environ["OPENAI_API_KEY"] = "your-key-here"

# Initialize tools
youtube_tool = YoutubeVideoSearchTool()
file_read_tool = FileReadTool()
file_write_tool = FileWriteTool()

# Initialize the Ollama LLM
ollama_llm = LLM(
    model="ollama/llama3.2:1b",  # or any other model you've pulled
    base_url="http://localhost:11434"
)

# Create Agents with Ollama LLM
researcher = Agent(
    role='YouTube Content Researcher',
    goal='Analyze YouTube video content and extract key information',
    backstory='Expert at analyzing video content and identifying key topics and insights',
    tools=[youtube_tool],
    llm=ollama_llm,  # Using Ollama LLM
    verbose=True
)

writer = Agent(
    role='Content Writer',
    goal='Create engaging blog posts from video content',
    backstory='Experienced writer specialized in converting video content into compelling articles',
    tools=[file_read_tool, file_write_tool],
    llm=ollama_llm,  # Using Ollama LLM
    verbose=True
)

editor = Agent(
    role='Content Editor',
    goal='Polish and optimize content for publication',
    backstory='Senior editor with expertise in SEO and content optimization',
    tools=[file_read_tool, file_write_tool],
    llm=ollama_llm,  # Using Ollama LLM
    verbose=True
)

# Define Tasks
analyze_video = Task(
    description='Analyze the YouTube video and extract key points and insights',
    expected_output='A detailed summary of the video content with main points and timestamps',
    agent=researcher
)

write_article = Task(
    description='Create a blog post based on the video analysis',
    expected_output='A well-structured blog post with introduction, main content, and conclusion',
    agent=writer,
    context=[analyze_video]
)

edit_article = Task(
    description='Polish the blog post and optimize it for SEO',
    expected_output='A final, publication-ready blog post with optimized title and meta description',
    agent=editor,
    context=[write_article],
    output_file='blog-posts/final_post.md'
)

# Create and run the crew
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[analyze_video, write_article, edit_article],
    verbose=True,
    planning=True  # Enable planning feature
)

try:
    # Execute the workflow
    result = crew.kickoff(inputs={"video_url": "YOUR_YOUTUBE_URL"})
except Exception as e:
    print(f"Error occurred: {str(e)}")
    # Implement fallback logic if needed