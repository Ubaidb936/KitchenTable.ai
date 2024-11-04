import os
import langchain
import base64
from openai import OpenAI
from langchain import globals
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain
from langchain.chains import TransformChain
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferMemory
from langchain.chains import TransformChain
from pydantic import BaseModel, Field



# os.environ["OPENAI_API_KEY"] = "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"

open_ai_api_key = ""


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
    You are an expert in identifying user intent.
    The system and user are engaged in a conversation about a photo uploaded by the user. The system asks questions related to the photo, and the user responds.
    Your task is to analyze the user's response to determine their intent accurately.

    Possible user intents include:
    1. "change photo" - The user indicates they want to move on to another photo, stop discussing the current one, or explicitly requests to end the conversation about this photograph.
    2. "continue" - The user is interested in continuing the current discussion and is engaged with the conversation about this photo.
    3. "fetch story" - The user asks to load a memory, story, etc or they want to talk about a particular memory, story, etc.
    4. "change topic" - The user indicates dissatisfaction with the current line of questioning and wishes to discuss a different aspect or topic.

    User input:
    {input}

    Provide:
    - The identified intent of the user.
    """

generate_summary_prompt = """
    Here is a conversation between a good friend and a user around a photograph uploaded by the user:
    {conversation}

    Summarize this conversation in a friendly, 3-line story using "you" to refer to the user. Then, ask a follow-up question to encourage them to continue sharing.

    This should be summarized for the user:
    1. A summary in 3 lines and a follow back question.
    """

generate_story_prompt = """
    Given the photograph upload by the user
    Here is a conversation between a good friend and a user around a photograph uploaded by the user:
    {conversation}

    Please generate a short story from this conversation about the photograph and a story name.
    Please build a realistic story, don't invent anything and use photo as a guide.  

    Provide:
    1. A short Story in 3 lines.
    2. Story name in 2 words.
    """


generate_story_name_prompt = """
    Here is a story:
    {story}

    Please generate a story name in 2 words.

    Provide:
    1. A Story name in two words.

"""    


change_photo_prompt = """
    A conversation is taking place between a user and a good friend around a photograph. Now, the user wants to move on to a new photograph.
    Here is the user's message: {message}

    Provide the following:

    1- Kindly acknowledge the user's message and, in one sentence, suggest they click the "Load New Photo" button to share a new story.

    Note:

    1- The user is an older person, so be gentle and polite.
    2- Adjust your response to match the tone of the user's message.
    3- The goal is to guide the user to seamlessly transition into sharing a new story by encouraging them to click the "Load New Photo" button.
    """



class GenerateQuestion(BaseModel):
    question: str = Field(description= "Respond to the user input and ask a follow back question, using conversation and photo provded as a guide.")

class StartingQuestion(BaseModel):
    question: str = Field(description= "A question to start converstion around the photograph")

class UserIntent(BaseModel):
    intent: str = Field(description= "Given a user input determine the intent of the user")

class GenerateSummary(BaseModel):
    summary: str = Field(description= " Summary and a follow back question")

class GenerateStory(BaseModel):
    story: str = Field(description= "A Story")
    story_name: str = Field(description= "A Story name")

class GenerateStoryName(BaseModel):
    story_name: str = Field(description = "A Story name")    


class ChangePhoto(BaseModel):
    message: str = Field(description= "A message to the user")


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
        api_key= open_ai_api_key
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
def text_model(inputs: dict) -> str | list[str] | dict:
    """Invoke model with image and prompt."""
    model = ChatOpenAI(
        temperature=0.5, 
        model="gpt-4o", 
        max_tokens=1024,
        api_key= open_ai_api_key
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
        self.change_photo_parser = JsonOutputParser(pydantic_object=ChangePhoto)
        self.summary_parser = JsonOutputParser(pydantic_object=GenerateSummary)
        self.story_name_parser = JsonOutputParser(pydantic_object=GenerateStoryName)
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
            prompt=   CONVERSATION_ENDING_PROMPT.format(history=memory, character=self.ai_character)

        vision_chain = load_image_chain | image_model | parser
        return vision_chain.invoke({'image_path': f'{image_path}', 'prompt': prompt, 'parser':parser})

    def get_intent(self, user_input) -> dict:
        parser = self.intent_parser
        prompt = user_intent_prompt.format(input=user_input)
        intent_chain = text_model | parser
        return intent_chain.invoke({'prompt': prompt, 'parser':parser})    

    def get_summary(self, conversation) -> dict:
        parser = self.summary_parser
        prompt = generate_summary_prompt.format(conversation=conversation)
        story_chain = text_model | parser
        return story_chain.invoke({'prompt': prompt, 'parser':parser})

    def get_story(self, image_path, conversation) -> dict:
        parser = self.story_parser
        prompt = generate_story_prompt.format(conversation=conversation)
        vision_chain = load_image_chain | image_model | parser
        return vision_chain.invoke({'image_path': f'{image_path}', 'prompt': prompt, 'parser':parser})   

    def change_photo_message(self, user_message)-> dict:
        parser = self.change_photo_parser
        prompt = change_photo_prompt.format(message = user_message)
        photo_chain = text_model | parser
        return photo_chain.invoke({'prompt':prompt, 'parser': parser})    

    def generate_story_name(self, story) -> dict:
        parser = self.story_name_parser
        prompt = generate_story_name_prompt.format(story=story)
        story_name_chain = text_model | parser
        return story_name_chain.invoke({"prompt": prompt, "parser": parser})    


# promptGen = PromptGenerator()

# res = promptGen.get_intent("It was my time with Ibrahim and my family. I took a trip recently.")

# print(res)
