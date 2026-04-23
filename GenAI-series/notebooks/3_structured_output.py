from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field
from typing import List

class PlayerStructure(BaseModel) :
    name : str = Field(description="Name of the player")
    age : int = Field(description="Age of the player")
    runs : int = Field(description="Total runs scored by the player")

class AllPlayersList(BaseModel) :
    players : List[PlayerStructure] = Field(description="List of all the players")

llm = ChatOpenAI(model="gpt-4o")
player_llm = llm.with_structured_output(PlayerStructure)
res = player_llm.invoke("give me details of the cricket player with highest runs in ipl all time")
print(res)
# players_llm = llm.with_structured_output(AllPlayersList)
# res = players_llm.invoke("give me 5 cricket players with highest runs in ipl all time")
# print(res)