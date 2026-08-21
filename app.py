import os

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Set it in Streamlit Cloud secrets or your local .env file."
        )
    return value


def build_agent():
    google_api_key = require_env("GEMINI_API")
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

    return create_agent(
        model=model,
        tools=[skill_demand_tool, search_jobs],
        system_prompt=system_prompt,
    )


def extract_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_value = item.get("text") or item.get("content")
                if text_value:
                    parts.append(str(text_value))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    if isinstance(content, dict):
        text_value = content.get("text") or content.get("content")
        if text_value:
            return str(text_value)
        return str(content)

    return str(content)


st.set_page_config(page_title="Skill-to-Career Mapping", page_icon="🎯")

st.title("Skill-to-Career Mapping")
st.write("Research skill demand and find relevant job opportunities in India.")

user_query = st.text_input(
    "Ask about a skill or role",
    value="What's the demand for generative ai in the industry and show me related job openings in India",
)

if st.button("Search"):
    try:
        agent = build_agent()
        with st.spinner("Researching skill demand and job openings..."):
            response = agent.invoke({"messages": [{"role": "user", "content": user_query}]})
            last_message = response["messages"][-1]
            text = extract_text(last_message.content)
        st.text_area("Result", value=text, height=500)
    except Exception as exc:
        st.error(f"Something went wrong: {exc}")
