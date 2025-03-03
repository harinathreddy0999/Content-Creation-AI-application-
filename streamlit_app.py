from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Streamlit page config
st.set_page_config(page_title="Content Researcher & Writer", page_icon="📝", layout="wide")

# Title and description
st.title("✍️ Content Researcher & Writer, powered by CrewAI.")
st.markdown("Generate blog posts about any topic using AI agents.")

# Sidebar
with st.sidebar:
    st.header("Content Settings")
    topic = st.text_area("Enter your topic", height=100, placeholder="Enter the topic")
    st.markdown("### LLM Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    st.markdown("---")

# Generate button
generate_button = st.button("Generate Content", type="primary", use_container_width=True)

# Expander for instructions
with st.expander("💡 How to use:"):
    st.markdown("""
    1. Enter your desired content topic  
    2. Adjust the temperature if needed  
    3. Click 'Generate Content' to start  
    4. Wait for the AI to generate your article  
    5. Download the result as a markdown file  
    """)

# LLM and tools
llm = LLM(model="gpt-4")
search_tool = SerperDevTool(n_results=10)

# Research Agent
senior_research_analyst = Agent(
    role="Senior Research Analyst",
    goal="Research, analyze, and synthesize comprehensive information on {topic} from reliable web sources.",
    backstory="You're an expert research analyst with advanced web research skills, skilled at fact-checking, identifying key insights, and structuring information effectively.",
    tools=[search_tool],
    llm=llm,
    allow_delegation=False,
    verbose=True
)

# Content Writing Agent
content_writer = Agent(
    role="Content Writer",
    goal="Create engaging, informative, and well-structured content based on research findings.",
    backstory="A skilled writer specializing in making complex topics accessible and engaging.",
    llm=llm,
    allow_delegation=False,
    verbose=False
)

# Research Task
research_task = Task(
    description="""
    Conduct in-depth research on {topic}, including:
    - Latest developments and news
    - Key trends and innovations
    - Expert opinions and insights
    - Statistical data and market research
    Ensure all sources are credible and properly cited.
    """,
    expected_output="""
    - A well-structured research brief summarizing key insights
    - Verified sources and citations
    - Clear categorization of information
    """,
    agent=senior_research_analyst
)

# Writing Task
writing_task = Task(
    description="""
    Using the research brief, create a compelling blog post that:
    - Translates technical details into engaging content
    - Includes structured sections with clear headings
    - Preserves citations in [Source: URL] format
    - Maintains factual accuracy
    """,
    expected_output="""
    - A well-written blog post in markdown format
    - Engaging introduction, structured body, and compelling conclusion
    - Inline citations and references section
    """,
    agent=content_writer
)

# Crew setup
crew = Crew(
    agents=[senior_research_analyst, content_writer],
    tasks=[research_task, writing_task],
    verbose=True
)

# Generate content
def generate_content(topic):
    if topic:
        result = crew.kickoff(input={"topic": topic})
        st.markdown("### Generated Content:")
        st.markdown(result, unsafe_allow_html=True)
    else:
        st.warning("Please enter a topic before generating content.")

if generate_button:
    generate_content(topic)
