import os
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Set it in Streamlit Cloud secrets or your local .env file."
        )
    return value


def build_agent(checkpointer: InMemorySaver):
    google_api_key = require_env("Gemini_API_Keyy")
    tavily_api_key = require_env("TAVILY_API_KEY")
    rapid_api_key = require_env("RAPID_API_KEY")
    job_results = []

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
        response = requests.get(
            "https://jsearch.p.rapidapi.com/search",
            headers={
                "x-rapidapi-key": rapid_api_key,
                "x-rapidapi-host": "jsearch.p.rapidapi.com",
            },
            params={
                "query": f"{skill} in {location}",
                "page": "1",
                "country": "in",
                "num_pages": "1",
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        jobs = data.get("data", []) if isinstance(data, dict) else []

        result = []
        for job in jobs:
            if not isinstance(job, dict):
                continue
            result.append(
                {
                    "title": job.get("job_title"),
                    "company": job.get("employer_name"),
                    "location": job.get("job_city"),
                    "apply_link": job.get("job_apply_link"),
                }
            )
        job_results.extend(result)
        return result

    system_prompt = """You are a Skill-to-Career Mapping assistant that helps students understand skills, companies, and career opportunities.

Use skill_demand_tool for current industry research and search_jobs when the user asks for job openings or application links. Explain answers clearly and include all useful details returned by the tools. Use each tool at most once per user request. Do not claim that no jobs were found if search_jobs returned results."""

    agent = create_agent(
        model=model,
        tools=[skill_demand_tool, search_jobs],
        system_prompt=system_prompt,
        checkpointer=checkpointer,
    )
    return agent, job_results


def extract_text(content) -> str:
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
        return "\n".join(parts)
    return str(content)


st.set_page_config(page_title="Skill-to-Career Mapping", page_icon="🎯")
st.title("Skill-to-Career Mapping")
st.write("Research skills, companies, and relevant job opportunities in India.")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = InMemorySaver()
if "agent" not in st.session_state:
    st.session_state.agent, st.session_state.job_results = build_agent(
        st.session_state.checkpointer
    )

user_query = st.text_input(
    "Ask about a skill, company, or role",
    value="Can you explain in detail about the Accenture company",
)

if st.button("Search"):
    try:
        st.session_state.job_results.clear()
        with st.spinner("Researching..."):
            response = st.session_state.agent.invoke(
                {"messages": [{"role": "user", "content": user_query}]},
                config={
                    "configurable": {"thread_id": st.session_state.thread_id},
                    "recursion_limit": 8,
                },
            )
            text = extract_text(response["messages"][-1].content)

        st.markdown(text)
        for job in st.session_state.job_results:
            apply_link = job.get("apply_link")
            if apply_link:
                title = job.get("title") or "Job listing"
                company = job.get("company") or "Company not listed"
                st.markdown(f"[{title} - {company}]({apply_link})")
    except Exception as exc:
        st.error(f"Something went wrong: {exc}")
