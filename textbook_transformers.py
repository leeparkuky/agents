import json
import random
from groq import Groq


tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_prompt",
            "description": "Get prompt based on the prompt_type",
            "parameters": {
                "type": "object",
                "properties": {
                    "synthetic_textbook_generator": {
                        "type": "list",
                        "description": "select the list of type of synthetic text generator",
                        "enum":  [
                                "Argument Passage Generator",
                                "Debate Passage Generator",
                                "Conversation Passage Generator",
                                "Meeting Transcript Generator",
                                "Poem Generator",
                                "Satirical Passage Generator",
                                "Instructional Passage Generator",
                                "Long Text Generator",
                                "Identity Agent",
                                "Employee Training Material Generator",
                                "Customer Service Conversation Generator",
                                "Risk Assessment Discussion Generator",
                                "Casual Inquiry Generator"
                                ]
                    }
                },
                "required": ["synthetic_textbook_generator"],
            },
        },
    }]


def get_suggestions(document: str, client: Groq):
    prompt = f"""I have a list of text generators, each specialized in creating different types of passages. Given a passage or web data, I want to transform it using one of these generators. Please suggest the top three generators that would be most suitable for transforming the given passage. Here are the available generators:

            1. Argument Passage Generator: This agent is adept at creating passages that articulate arguments, which may occasionally contain logical inconsistencies.
            2. Debate Passage Generator: It specializes in crafting passages that mimic the structure and content of debate transcripts.
            3. Conversation Passage Generator: This agent generates passages that depict dialogues.
            4. Meeting Transcript Generator: It is designed to produce meeting transcripts.
            5. Poem Generator: This agent generates poems.
            6. Satirical Passage Generator: It creates texts infused with satirical wit.
            7. Instructional Passage Generator: This agent generates passages resembling instructional manuals.
            8. Long Text Generator: It extends the original text by incorporating additional information, thereby increasing its length.
            9. Identity Agent: A straightforward agent that replicates the input text verbatim.
            10. Employee Training Material Generator: It generates passages that can be used as training material for employees.
            11. Customer Service Conversation Generator: This agent generates passages that simulate customer service conversations.
            12. Risk Assessment Discussion Generator: It generates passages that discuss risk assessment.
            13. Casual Inquiry Generator: This agent generates casual inquiries.


            Based on the given passage, please suggest the top three generators that would be most suitable for transforming it.

            ### Passage:
            {document}"""
    messages= [{"role": "user", "content": prompt}]
    model = random.choice(["llama3-groq-70b-8192-tool-use-preview", "llama3-70b-8192", "llama3-8b-8192", "gemma2-9b-it"])
    chat_completion = client.chat.completions.create(
                        messages= messages, 
                        model=model,
                        top_p = .9,
                        temperature = .7,
                        # tools = tools
                        )
    return chat_completion.choices[0].message.content


def synthetic_textbook_function_calling(document, suggestion, client):
    prompt = f"""I have a list of text generators, each specialized in creating different types of passages. Given a passage or web data, I want to transform it using one of these generators. Please suggest the top three generators that would be most suitable for transforming the given passage. Here are the available generators:

1. Argument Passage Generator: This agent is adept at creating passages that articulate arguments, which may occasionally contain logical inconsistencies.
2. Debate Passage Generator: It specializes in crafting passages that mimic the structure and content of debate transcripts.
3. Conversation Passage Generator: This agent generates passages that depict dialogues.
...
11. Customer Service Conversation Generator: This agent generates passages that simulate customer service conversations.
12. Risk Assessment Discussion Generator: It generates passages that discuss risk assessment.
13. Casual Inquiry Generator: This agent generates casual inquiries.


Based on the given passage, please suggest the top three generators that would be most suitable for transforming it.

### Passage:
<document skipped>

### Suggestions:
{suggestion}"""
    # model = random.choice(["llama3-groq-8b-8192-tool-use-preview", "llama3-groq-70b-8192-tool-use-preview",  "gemma2-9b-it"])
    # if model in ['llama3-8b-8192', 'gemma2-9b-it']:
    #     model = random.choice(["llama3-groq-8b-8192-tool-use-preview", "llama3-8b-8192",  "gemma2-9b-it"])
    chat_completion = client.chat.completions.create(
    messages= [{"role": "user", "content": prompt}], 
    model= 'llama3-groq-8b-8192-tool-use-preview',
    tools = tools
)
    kwargs = json.loads(chat_completion.choices[0].message.tool_calls[0].function.arguments)
    return kwargs





generators = {
    "Argument Passage Generator": {
        "description": "This agent is adept at creating passages that articulate arguments, which may occasionally contain logical inconsistencies.",
        "system_message": """
        You are an Argument Passage Generator. Your task is to create passages that articulate arguments, which may occasionally contain logical inconsistencies.
        Ensure that the passage is comprehensive, logically structured, and may occasionally contain logical inconsistencies to reflect the nature of arguments.
        With your expertise, make the arguments more engaging, informative, and thought-provoking in the domain of the source text.
        """,
        "user_message": """
        Transform the following source text using the Argument Passage Generator. Here is the source text:

        {{document}}

        Your task is to create a passage that articulates arguments based on the source text. Ensure that the passage is comprehensive, logically structured, and may occasionally contain logical inconsistencies to reflect the nature of arguments. The transformed document should be in simple English paragraphs.
        """
    },
    "Debate Passage Generator": {
        "description": "It specializes in crafting passages that mimic the structure and content of debate transcripts.",
        "system_message": """
        You are a Debate Passage Generator. Your task is to craft passages that mimic the structure and content of debate transcripts.
        Ensure that the passage includes multiple perspectives, counterarguments, and rebuttals to reflect a debate format.
        With your expert knowledge, make the debate more engaging and informative.
        """,
        "user_message": """
        Transform the following source text using the Debate Passage Generator. Here is the source text:

        {{document}}

        Your task is to create a passage that mimics the structure and content of a debate transcript. Ensure that the passage includes multiple perspectives, counterarguments, and rebuttals to reflect a debate format. The transformed document should be in simple English paragraphs.
        """
    },
    "Conversation Passage Generator": {
        "description": "This agent generates passages that depict dialogues.",
        "system_message": """
        You are a Conversation Passage Generator. Your task is to generate passages that depict dialogues.
        Ensure that the dialogue is natural, engaging, and reflects the context of the source text.
        """,
        "user_message": """
        Transform the following source text using the Conversation Passage Generator. Here is the source text:

        {{document}}

        Your task is to create a passage that depicts a dialogue between two or more characters. Ensure that the dialogue is natural, engaging, and reflects the context of the source text. The transformed document should be in simple English paragraphs.
        """
    },
    "Meeting Transcript Generator": {
        "description": "It is designed to produce meeting transcripts.",
        "system_message": """
        You are a Meeting Transcript Generator. Your task is to produce meeting transcripts.
        Ensure that the transcript includes speaker labels, and captures the flow of the meeting accurately.
        Do not include timestamps in the transcript. Only include speaker labels.
        """,
        "user_message": """
        Transform the following source text using the Meeting Transcript Generator. Here is the source text:

        {{document}}

        Your task is to create a passage that resembles a meeting transcript. Ensure that the transcript includes speaker labels, timestamps, and captures the flow of the meeting accurately. The transformed document should be in simple English paragraphs.
        """
    },
    "Poem Generator": {
        "description": "This agent generates poems.",
        "system_message": """
        You are a Poem Generator. Your task is to generate poems.
        Ensure that the poem has a clear structure, rhythm, and conveys the emotions or themes present in the source text.
        """,
        "user_message": """
        Transform the following source text using the Poem Generator. Here is the source text:

        {{document}}

        Your task is to create a poem based on the source text. Ensure that the poem has a clear structure, rhythm, and conveys the emotions or themes present in the source text. The transformed document should be in simple English paragraphs.
        """
    },
    "Satirical Passage Generator": {
        "description": "It creates texts infused with satirical wit.",
        "system_message": """
        You are a Satirical Passage Generator. Your task is to create texts infused with satirical wit.
        Ensure that the passage is humorous, critical, and cleverly highlights the absurdities or contradictions in the source text.
        """,
        "user_message": """
        Transform the following source text using the Satirical Passage Generator. Here is the source text:

        {{document}}

        Your task is to create a passage infused with satirical wit based on the source text. Ensure that the passage is humorous, critical, and cleverly highlights the absurdities or contradictions in the source text. The transformed document should be in simple English paragraphs.
        """
    },
    "Instructional Passage Generator": {
        "description": "This agent generates passages resembling instructional manuals.",
        "system_message": """
        You are an Instructional Passage Generator. Your task is to generate passages resembling instructional manuals.
        Ensure that the passage is clear, step-by-step, and provides detailed instructions or guidelines.
        """,
        "user_message": """
        Transform the following source text using the Instructional Passage Generator. Here is the source text:

        {{document}}

        Your task is to create a passage that resembles an instructional manual based on the source text. Ensure that the passage is clear, step-by-step, and provides detailed instructions or guidelines. The transformed document should be in simple English paragraphs.
        """
    },
    "Long Text Generator": {
        "description": "It extends the original text by incorporating additional information, thereby increasing its length.",
        "system_message": """
        You are a Long Text Generator. Your task is to extend the original text by incorporating additional information, thereby increasing its length.
        Ensure that the extended passage is coherent, informative, and seamlessly integrates with the original content.
        """,
        "user_message": """
        Transform the following source text using the Long Text Generator. Here is the source text:

        {{document}}

        Your task is to extend the original text by incorporating additional information, thereby increasing its length. Ensure that the extended passage is coherent, informative, and seamlessly integrates with the original content. The transformed document should be in simple English paragraphs.
        """
    },
    "Identity Agent": {
        "description": "A straightforward agent that replicates the input text verbatim.",
        "system_message": """
        You are an Identity Agent. Your task is to replicate the input text verbatim.
        Ensure that the replicated text is accurate and maintains the original formatting and content.
        """,
        "user_message": """
        Transform the following source text using the Identity Agent. Here is the source text:

        {{document}}

        Your task is to replicate the input text verbatim. Ensure that the replicated text is accurate and maintains the original formatting and content. The transformed document should be in simple English paragraphs.
        """
    },
    "Employee Training Material Generator": {
        "description": "This agent generates comprehensive training materials for new employees, providing in-depth tutorials to help them understand and execute their duties effectively.",
        "system_message": """
        You are an Employee Training Material Generator. Your task is to create comprehensive training materials for new employees, providing in-depth tutorials to help them understand and execute their duties effectively.
        Ensure that the materials are detailed, easy to understand, and provide step-by-step instructions or guidelines.
        """,
        "user_message": """
        Transform the following source text using the Employee Training Material Generator. Here is the source text:

        {{document}}

        Your task is to create comprehensive training materials for new employees based on the source text. Ensure that the materials are detailed, easy to understand, and provide step-by-step instructions or guidelines to help new employees execute their duties effectively. The transformed document should be in simple English paragraphs.
        """
    },
    "Customer Service Conversation Generator": {
        "description": "This agent generates dialogues between a customer and a service representative. It includes sections such as greeting, inquiry, response, and resolution.",
        "system_message": """
        You are a Customer Service Conversation Generator. Your task is to generate dialogues between a customer and a service representative. It includes sections such as greeting, inquiry, response, and resolution.
        """,
        "user_message": """
        Transform the following source text using the Customer Service Conversation Generator. Here is the source text:

        {{document}}

        Your task is to create a dialogue between a customer and a service representative. Ensure that the dialogue includes sections such as greeting, inquiry, response, and resolution. The transformed document should be in simple English paragraphs.
        """
    },
    "Risk Assessment Discussion Generator": {
        "description": "This agent creates discussions around risk assessment scenarios. It includes sections such as risk identification, evaluation, and mitigation strategies, presented in a conversational format.",
        "system_message": """
        You are a Risk Assessment Discussion Generator. Your task is to create discussions around risk assessment scenarios. It includes sections such as risk identification, evaluation, and mitigation strategies, presented in a conversational format.
        """,
        "user_message": """
        Transform the following source text using the Risk Assessment Discussion Generator. Here is the source text:

        {{document}}

        Your task is to create a discussion around risk assessment scenarios. Ensure that the discussion includes sections such as risk identification, evaluation, and mitigation strategies, presented in a conversational format. The transformed document should be in simple English paragraphs.
        """
    },
    "Casual Inquiry Generator": {
        "description": "This agent generates casual conversations where customers inquire about various insurance-related topics. It includes sections such as question, detailed explanation, and follow-up questions.",
        "system_message": """
        You are a Casual Inquiry Generator. Your task is to generate casual conversations where customers inquire about various insurance-related topics. It includes sections such as question, detailed explanation, and follow-up questions.
        """,
        "user_message": """
        Transform the following source text using the Casual Inquiry Generator. Here is the source text:

        {{document}}

        Your task is to create a casual conversation where customers inquire about various insurance-related topics. Ensure that the conversation includes sections such as question, detailed explanation, and follow-up questions. The transformed document should be in simple English paragraphs.
        """
    }
}

system_prompt = """\
You are an AI text generator specialized in transforming texts according to specific styles and formats. Your role is to consistently apply the characteristics of the chosen generator to the given source text. Ensure that the transformed text aligns with the description and capabilities of the selected generator.
"""

def retrieve_prompt(document: str, synthetic_textbook_generator: list[str]):
    generator_name = random.choice(synthetic_textbook_generator)
    if generator_name in generators.keys():
        system_message = generators[generator_name]["system_message"]
        user_prompt = generators[generator_name]["user_message"]
    else:
        generator_name = random.choice(list(generators.keys()))
        system_message = generators[generator_name]["system_message"]
        user_prompt = generators[generator_name]["user_message"]
    user_prompt = user_prompt.replace("{{document}}", document)
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_prompt}]
    return messages, generator_name


