import os
import langchain
import base64
from openai import OpenAI
from langchain import globals
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain
from langchain.chains import TransformChain
# from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferMemory
from langchain.chains import TransformChain
from pydantic import BaseModel, Field



os.environ["OPENAI_API_KEY"] = "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"

class GenerateQuestion(BaseModel):
    """Information about an image."""
    question: str = Field(description= "Respond to the user input and ask a follow back question, using conversation and photo provded as a guide.")

class StartingQuestion(BaseModel):
    """Information about an image."""
    question: str = Field(description= "A question to start converstion around the photograph")

class UserIntent(BaseModel):
    intent: str = Field(description= "Intent of the user")

class GenerateStory(BaseModel):
     story: str = Field(description= "Generate a story from a converation around a photo graph")



CONVERSATION_STARTER_PROMPT = """
  You are playing the role of a {character} who is interested in learning about the user by asking them questions about the photo they’ve uploaded.
  Provide:
  - A question to start the conversation around the photograph.
  Note:
  1. You first want to know about the photo. Focus on the people in the photo and the relationships.  For example, ask who is in it, where it was taken, or if it was a special occasion.
  2. Avoid questions about emotions or feelings. Keep questions simple and open ended.
  3. Don't ask about things in the photo.
  4. Ask them if there are topics they would like to talk about.
"""

CONVERSATION_STARTER2_PROMPT = """
  You are playing the role of a {character} who is interested in learning about the user by asking them questions about the photo they’ve uploaded.
  Here is the conversation history about the image between the user and you ({character}):
  {history}
  Provide:
  - A question about the contents of the photograph.
  Note:
  1. You first want to know about the contents of the photo. For example, ask who is in it, where it was taken, or if it was a special occasion.
  2. Do not repeat questions.  Don't ask a very personal question.
  3.  Focus on people and relationships. Keep questions simple and open ended.
  4. Use conversation history.
  5. Ask about people mentioned in user's responses not just in photo.
  6. Don't ask about things in the photo unless the person brings them up
  7. Avoid platitudes
"""


CONVERSATION_EXPANDING_PROMPT = """
  You are playing the role of a good friend who is interested in learning about the user by asking them questions about the photo they’ve uploaded.
  You are currently in the middle of a conversation with the user.
  Here is the conversation history about the image between the user and you (the good friend), reflecting the ongoing dialogue:
  {history}
  Provide:
  - A reply to the user's most recent input and a follow-up question that encourages them to expand on their answer about the people in the photograph or mentioned in their response
  Notes:
  1- Ask them about any stories they are reminded of and please ask only one question.
  2- Do not repeat questions or ask about information that has already been covered.
  3- Encourage full responses by asking open-ended questions that invite further elaboration.
  5- Use the conversation history to inform your question, while maintaining the flow of the ongoing conversation.
  6- Don't ask about things in the photo unless the person brings them up
  7- Avoid platitudes.
"""


CONVERSATION_ENDING_PROMPT = """
  You are playing the role of a {character} who is interested in learning about the user by asking them questions about the photo they’ve uploaded.
  Here is the conversation history about the image between the user and you ({character}): reflecting the ongoing dialogue:
  {history}
  Provide:
  - A reply to the user's most recent input and a follow-up question that encourages them to share more about the story depicted in the photograph,
    discuss anything that the photograph reminds them of, or move on to another photograph or stop reminiscing.
  Notes:
  1- Ask them if they want to keep talking about the photo or move onto another photo.
  2- please ask only one question.
  3- Ask them to summarize how they feel about the photo
  4- Do not repeat questions or ask about information already covered in the conversation.
  5- Encourage full responses by asking open-ended questions that invite further elaboration.
"""

user_intent_prompt = """
  The system and user are engaged in a conversation about a photo uploaded by the user.
  The system asks questions related to the photograph, and the user responds.
  Your task is to analyze the user's input to accurately determine their intent.
  The possible intents are:
  1. "change photo" - The user explicitly or implicitly indicates they want to move on to the next photo, do not wish to discuss the current photo, or directly state a desire to stop talking about the current photograph.
  2. "change topic" - The user expresses a desire to talk about something else within the context of the current photograph or shows disinterest in the current line of questioning but doesn't want to change the photograph itself.
  3. "continue" - The user is comfortable with the current conversation and wants to continue discussing the current photo.
  Here is the user's input:
  {input}
  Provide:
  1. The intent of the user.
"""

generate_story_prompt = """
  Here is a conversation b/w a good friend and a user around a photo graph uploaded by a user. 
  {conversation}
  Please convert this conversation into a story in 3 lines
  Provide:
  1. A Story in 3 lines.
"""






def load_image(inputs: dict) -> dict:
    """Load image from file and encode it as base64."""
    image_path = inputs["image_path"]
  
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    image_base64 = encode_image(image_path)
    return {"image": image_base64}


load_image_chain = TransformChain(
    
        input_variables=["image_path"],
        output_variables=["image"],
        transform= load_image

    )

@chain
def image_model(inputs: dict) -> str | list[str] | dict:
    """Invoke model with image and prompt."""
    model = ChatOpenAI(
        temperature=0.5, 
        model="gpt-4o", 
        max_tokens=1024,
        api_key= "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"
        )
    msg = model.invoke(
                [HumanMessage(
                content=[
                {"type": "text", "text": inputs["prompt"]},
                {"type": "text", "text": inputs["parser"].get_format_instructions()},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{inputs['image']}"}},
                ])]
                )
    return msg.content


@chain
def input_model(inputs: dict) -> str | list[str] | dict:
    """Invoke model with image and prompt."""
    model = ChatOpenAI(
        temperature=0.5, 
        model="gpt-4o", 
        max_tokens=1024,
        api_key= "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"
        )
    msg = model.invoke(
                [HumanMessage(
                content=[
                {"type": "text", "text": inputs["prompt"]},
                {"type": "text", "text": inputs["parser"].get_format_instructions()},
                ])]
                )
    return msg.content    


@chain
def story_model(inputs: dict) -> str | list[str] | dict:
    """Invoke model with image and prompt."""
    model = ChatOpenAI(
        temperature=0.5, 
        model="gpt-4o", 
        max_tokens=1024,
        api_key= "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"
        )
    msg = model.invoke(
                [HumanMessage(
                content=[
                {"type": "text", "text": inputs["prompt"]},
                {"type": "text", "text": inputs["parser"].get_format_instructions()},
                ])]
                )
    return msg.content    


class PromptGenerator:
    def __init__(self):
        self.question_parser = JsonOutputParser(pydantic_object=GenerateQuestion)
        self.starting_question_parser = JsonOutputParser(pydantic_object=StartingQuestion)
        self.intent_parser = JsonOutputParser(pydantic_object=UserIntent)
        self.story_parser = JsonOutputParser(pydantic_object=GenerateStory)
        self.ai_character = "Good Friend"

    def get_prompt(self, image_path: str, iter: int, memory: str) -> dict:
        if iter == 1:
            parser = self.starting_question_parser
            prompt = CONVERSATION_STARTER_PROMPT.format(character=self.ai_character)
        elif  iter >= 2 and iter <= 3:
            parser = self.starting_question_parser
            prompt =   CONVERSATION_STARTER2_PROMPT.format(history=memory, character=self.ai_character)
        elif  iter > 3 and iter <= 9:
            parser = self.question_parser
            prompt=   CONVERSATION_EXPANDING_PROMPT.format(history=memory, character=self.ai_character)
        else:
            parser = self.question_parser
            prompt=  CONVERSATION_ENDING_PROMPT.format(history=memory, character=self.ai_character)

        vision_chain = load_image_chain | image_model | parser
        return vision_chain.invoke({'image_path': f'{image_path}', 'prompt': prompt, 'parser':parser})

    def get_intent(self, user_input) -> dict:
        parser = self.intent_parser
        prompt = user_intent_prompt.format(input=user_input)
        intent_chain = input_model | parser
        return intent_chain.invoke({'prompt': prompt, 'parser':parser})    

    def get_story(self, conversation) -> dict:
        parser = self.story_parser
        prompt = generate_story_prompt.format(conversation=conversation)
        story_chain = story_model | parser
        return story_chain.invoke({'prompt': prompt, 'parser':parser})      