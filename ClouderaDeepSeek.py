from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from subprocess import getoutput
from dotenv import load_dotenv
import os

load_dotenv()


os.environ["OPENAI_API_KEY"] = getoutput("cdp iam generate-workload-auth-token --workload-name DE 2>/dev/null| jq -r '.token'")
cloudera_endpoint = os.environ["CLOUDERA_ENDPOINT"]
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

from langchain_community.tools import WikipediaQueryRun, YouTubeSearchTool
from langchain_community.utilities import WikipediaAPIWrapper


wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wikipedia = WikipediaQueryRun(description="A tool to explain things in text format. Use this tool if you think the user’s asked concept is best explained through text.", api_wrapper=wiki_api_wrapper)

youtube = YouTubeSearchTool(
   description="A tool to search YouTube videos. Use this tool if you the user asks for a video."
)

llm = ChatOpenAI(temperature=0.7,
                 model_name=model_name,
                 base_url=cloudera_endpoint)


agent = initialize_agent(
    tools=[wikipedia, youtube],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,

)

def solve_problem(problem: str):
    response = agent.invoke({'input':problem})
    return response


if __name__ == "__main__":
    problem = "What is Apache NiFi? Link me a video to learn about"
    result = solve_problem(problem)
    print("Result:", result["output"])