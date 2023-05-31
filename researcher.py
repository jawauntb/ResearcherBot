# I originally wrote this as a script in a Jupyter notebook. Haven't published a lot of that work so I wanted to start here.
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

key = "sk-"
skey = " "
# llm = OpenAI(temperature=0.9, openai_api_key=key)
# prompt = PromptTemplate(
#     input_variables=["company"],
#     template="give me a summary of what {company} is up to in 2023?",
# )

# chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
# print(chain.run("diageo"))


llm = OpenAI(temperature=0, openai_api_key=key)
tools = load_tools(["serpapi", "llm-math"], llm=llm, serpapi_api_key=skey)
agent = initialize_agent(
    tools, llm, agent="zero-shot-react-description", verbose=True)
agent.run(
    "The company is Google. How large is this company? What is it's market cap? What are their mission and what vision statements? Are they involved ESG, diversity, sustainability initiatives"
    )

from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, SerpAPIWrapper, LLMChain

search = SerpAPIWrapper(serpapi_api_key=skey)
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    )
]

prefix = """Answer the following questions as best you can. You have access to the following tools:"""
suffix = """Begin!"

Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "agent_scratchpad"]
)
print(prompt.template)

llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt, openai_api_key=key)
tool_names = [tool.name for tool in tools]
agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True)


from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, SerpAPIWrapper, LLMChain

key = " "
skey = " "

search = SerpAPIWrapper(serpapi_api_key=skey)
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    )
]

prefix = """Answer the following questions as best you can. You have access to the following tools:"""
suffix = """The company you must get this information for: {company}.

Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "company", "agent_scratchpad"]
)
llm_chain = LLMChain(llm=OpenAI(temperature=0, openai_api_key=key),
                     prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools)
inpt = """
what is this company's Market cap or valuation?
What key challenges does the company or its industry face in 2023?
According to earnings reports and press, what are their key initiatives in 2023?
what is the company's mission statement?
What values does the company hold or champion?
Does the company have corporate philanthropy initiatives--what are they?,
what are the company's funding priorities for 2023?
describe some examples of corporate social responsibility efforts by the company in 2022 or 2023.
How does the company give back within their community?
describe their Involvement in ESG related initiatives.
what does the company think or do about diversity?
Is the company involved in sustainability initiatives?
Are they involved in Brooklyn based initiatives, and if so which initiatives?
"""
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True)
companies_to_check = ["Diageo", "Meta", "Coinbase", "MetaMask", "Rainbow", "Zora", "Roblox", "Trust Wallet", "GQ", "Polygon", "SuperRare", "Gemini", "Argent", "Adobe", "Vogue", "NFTNYC",
                      "Snapchat", "ConsenSys", "OpenSea", "Rarible", "Nifty Gateway", "Sotheby's", "KnownOrigin", "Foundation", "Larva Labs", "Infura", "Alchemy", "Mintable", "Binance", "LooksRare"]
with open("/Users/jawaun/langchain_stuff/output2.txt", "w") as f:
    for comp in companies_to_check:
        output = agent_executor.run(
            input=inpt, company=comp)
        f.write(f"{comp}: {output}\n")

