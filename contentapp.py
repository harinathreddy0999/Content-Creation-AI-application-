from crewai import Agent,Task,Crew,LLM
from crewai_tools import SerperDevtool

from dotenv import load_dotenv

load_dotenv()

topic="Medical Industry using Generative AI"

#Tool 1
llm=LLM(model="gpt-4")

#Tool 2
serach_tool= SerperDevTool(n=15)


#Agent 1
senior_research_analyst = Agent(
    role="Research Analyst",
    goal="Find and summarize information about {topic} from the web sources",
    backstory="You are an experienced researcher with attention to detail",
    tools=[SerperDevTool()],
    verbose=True  # Enable logging for debugging
    llm="gpt-4",  # Default: OPENAI_MODEL_NAME or "gpt-4"
    function_calling_llm=None,  # Optional: Separate LLM for tool calling
    memory=True,  # Default: True
    verbose=False,  # Default: False
    allow_delegation=False,  # Default: False
    max_iter=20,  # Default: 20 iterations
    max_rpm=None,  # Optional: Rate limit for API calls
    max_execution_time=None,  # Optional: Maximum execution time in seconds
    max_retry_limit=2,  # Default: 2 retries on error
    allow_code_execution=False,  # Default: False
    code_execution_mode="safe",  # Default: "safe" (options: "safe", "unsafe")
    respect_context_window=True,  # Default: True
    use_system_prompt=True,  # Default: True
    tools=[SerperDevTool()],  # Optional: List of tools
    knowledge_sources=None,  # Optional: List of knowledge sources
    embedder=None,  # Optional: Custom embedder configuration
    system_template=None,  # Optional: Custom system prompt template
    prompt_template=None,  # Optional: Custom prompt template
    response_template=None,  # Optional: Custom response template
    step_callback=None,  # Optional: Callback function for monitoring
)


#Agent 2
 
content_writer = Agent(
    role="Content Writer",
    goal="Develop informative and engaging content about medical advancements, trends, and healthcare innovations."  
         "Create SEO-optimized articles, blogs, and social media posts to educate and engage the audience."  
          "Translate complex medical information into clear, accessible, and reader-friendly content."  
          "Ensure accuracy, credibility, and compliance with medical writing standards and regulations."  
          "Boost brand authority by crafting compelling narratives on healthcare topics and industry updates."  

    backstory=- "A highly skilled and passionate content writer with deep expertise in the medical industry."  
                 "Excels at transforming complex medical jargon into engaging and reader-friendly content."  
                 "Adheres to the highest standards of accuracy, credibility, and compliance in medical writing."  
                 "Continuously researches industry trends to produce insightful, up-to-date, and valuable content."  
                "Crafts compelling narratives that educate, engage, and build trust with the audience." ,
    llm="gpt-4",  # Default: OPENAI_MODEL_NAME or "gpt-4"
    function_calling_llm=None,  # Optional: Separate LLM for tool calling
    memory=True,  # Default: True
    verbose=False,  # Default: False
    allow_delegation=False,  # Default: False
    
    
)

#Research task

research_task= Task(

    description=("""
                 - Conducts in-depth research on recent developments on {topic} and current news in the medical industry
                - Continuously updates data to ensure accuracy and relevance in all content 
                - Evaluates sources for credibility, reliability, and scientific validity before inclusion.
                - Organizes findings in a structured format with clear insights and key takeaways.
                - Includes relevant citations and sources to support claims and enhance trustworthiness.

        
    """),
    
  expected_output= """
             -  Provides a concise executive summary highlighting key insights.
             -  Delivers a comprehensive analysis of medical industry trends and updates.
             -  Presents a structured list of facts, key takeaways, and supporting links.
             -  Ensures all citations are properly included for credibility and verification.
             -  Organizes content with clear sections, bullet points, and categorized information.
"""
  agent = senior_research_analyst
  
)
  


 #task 2 Content Writing 
 writing_task = Task(
    description = ("""Using the research brief provided, create an engaging blog post that:
    1. Transforms technical information into accessible content
    2. Maintains all factual accuracy and citations from the research
    3. Includes:
        - Attention-grabbing introduction
        - Well-structured body sections with clear headings
        - Compelling conclusion
    4. Preserves all source citations in [Source: URL] format
    5. Includes a References section at the end.
 """      
    ),
    expected_output = """A polished blog post in markdown format that:
   - Engages readers while maintaining accuracy
   - Contains properly structured sections
   - Includes inline citations hyperlinked to the original source URL
   - Presents information in an accessible yet informative way
   - Follows proper markdown formatting, use H1 for the title and H3 for the sub-sections""",
   
   agent =content_writer
   
   
)

crew = crew(
    agents = [ senior_research_analyst,content_writer],
    tasks = [research_task, writing_task],
    verbose = True
)

result = crew.kickoff(input = {"topic": topic })

print(result)
