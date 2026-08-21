import os
from typing import Any

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()


def require_env(name: str, legacy_name: str | None = None) -> str:
    value = os.getenv(name)
    if not value and legacy_name:
        value = os.getenv(legacy_name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Set it before running this script."
        )
    return value


def build_agent():
    google_api_key = require_env("Gemini_API_Keyy")
    tavily_api_key = require_env("TAVILY_API_KEY")
    rapid_api_key = require_env("RAPID_API_KEY")

    model = init_chat_model(
        "google_genai:gemini-2.5-flash",
        api_key=google_api_key,
    )

    skill_demand_tool = TavilySearch(
        max_result=10,
        topic="general",
        search_depth="advanced",
        tavily_api_key=tavily_api_key,
    )

    @tool
    def search_jobs(skill: str, location: str) -> list:
        """Search for jobs requiring a specific skill using JSearch API from RapidAPI."""
        print(f"\nCalling search_jobs tool")
        print(f"Searching jobs for: {skill} in {location}")

        url = "https://jsearch.p.rapidapi.com/search-v2"
        headers = {
            "x-rapidapi-key": rapid_api_key,
            "x-rapidapi-host": "jsearch.p.rapidapi.com",
            "Content-Type": "application/json",
        }
        querystring = {
            "query": f"{skill} in {location}",
            "page": "1",
            "country": "in",
            "employment_types": "INTERN,FULLTIME",
            "job_requirements": "no_experience,under_3_years_experience",
        }

        response = requests.get(url, headers=headers, params=querystring, timeout=30)
        response.raise_for_status()
        data = response.json()

        jobs = data.get("data", []) or []
        print(f"Found {len(jobs)} jobs\n")

        result = []
        for job in jobs:
            result.append(
                {
                    "title": job.get("job_title"),
                    "company": job.get("employer_name"),
                    "location": job.get("job_city"),
                    "apply_link": job.get("job_apply_link"),
                }
            )
        return result

    system_prompt = """You are a Skill-to-Career Mapping assistant that helps students understand skill demand and find matching job opportunities.

You have access to these tools:
- skill_demand_tool: Search for industry demand, salary insights, and career trends
- search_jobs: Find actual job listings requiring specific skills

Help the student by researching the skill they ask about and finding relevant opportunities.

Present results in a clean, readable format with clear sections and proper spacing. Include all job details with apply links. Don't use markdown format."""

    agent = create_agent(
        model=model,
        tools=[skill_demand_tool, search_jobs],
        system_prompt=system_prompt,
    )
    return agent


def main() -> None:
    user_query = (
        "What's the demand for generative ai in the industry and show me related job openings in India"
    )

    agent = build_agent()
    response = agent.invoke({"messages": [{"role": "user", "content": user_query}]})

    last_message = response["messages"][-1]
    content = last_message.content

    if isinstance(content, list):
        text = content[0].get("text", "") if content else ""
    else:
        text = str(content)

    print(text)


if __name__ == "__main__":
    main()
