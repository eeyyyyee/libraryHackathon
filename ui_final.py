import chainlit as cl
from chainlit import AskUserMessage, Message
from pypdf import PdfReader
import openai
from io import BytesIO
import re


openai.api_key = "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"


@cl.on_chat_start
async def main():
    await Message(content="Welcome to the Law Library Chatbot").send()

    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a pdf file to begin!", accept=["application/pdf"],
            max_files=10,
            max_size_mb=1000,
            timeout=30,
        ).send()
    
    process_pdf_msg = Message(content="Processing PDF...")
    await process_pdf_msg.send()

    files_dict = {}
    for file in files:
        pdf_stream = BytesIO(file.content)
        pdf_data = PdfReader(pdf_stream)
        extracted_text = ""
        
        for page in pdf_data.pages:
            text = page.extract_text()
            extracted_text += text
        
        extracted_text = re.sub(r'\n+', '\n', extracted_text)
        extracted_text = re.sub(r'(\.\s)\n', r'\1', extracted_text)
        
        # chunking on extracted text
        extracted_text_split_by_space = extracted_text.split(" ")
        chunks = []
        start_index = 0
        number_of_words = len(extracted_text_split_by_space)
        while start_index < number_of_words:
            chunks.append(' '.join(extracted_text_split_by_space[start_index:start_index+200]))
            start_index += 200

        # chunks contain a list of strings, each string is a chunk of 200 words
        chunks_and_embeddings = []
        for chunk in chunks:
            chunks_and_embeddings.append((chunk, embed_text(chunk)))

        files_dict[file.name] = chunks_and_embeddings

    user_continue = True
    query = await AskUserMessage(content="What is your query?", timeout=1000).send()

    while user_continue:
        # Embed user query
        embedded_query = embed_text(query["content"])
    
        # {"name":[("score","embedding")]}
        # {"name": "extracted_text"}
        new_dict_with_scores = {}
        for title, chunks_and_embeddings_list in files_dict.items():
            temp_ls = []
            for each_tuple in chunks_and_embeddings_list:
                chunk_text = each_tuple[0]
                embedding = each_tuple[1]
                score = cosine_similarity(embedded_query, embedding)
                temp_ls.append((chunk_text, score))
            # sort the chunks according to the scores, ith the most relevant ones at the front
            temp_ls.sort(key=lambda x: x[1], reverse=True)
            new_dict_with_scores[title] = temp_ls
            
        new_dict_for_gpt_prompt = {}
        new_dict_for_extracts = {}
        for title, value in new_dict_with_scores.items():
            temp_string = ""
            i = 0
            extract_list = []
            while i < 3:
                temp_string += value[i][0]
                extract_list.append(value[i][0])
                i += 1
            new_dict_for_gpt_prompt[title] = temp_string
            new_dict_for_extracts[title] = extract_list
        gpt_summary = gpt_completion(query["content"], str(new_dict_for_gpt_prompt))
        finalMessage = await Message(content=gpt_summary).send()

        # asks the user if they would like to see the relevant extract from the case itself
        see_relevant_extracts = await Message(content="These are the relevant extracts that we have found:").send()
        output = ""
        for key, value in new_dict_for_extracts.items():
            output += key + "\n"
            for extract in value:
                output += extract + "\n\n"
            output += "\n"
        output_extract = await Message(content=output).send()

        # Queries that the user might ask
        see_relevant_queries = await Message(content="Based on your previous question, you might ask these questions next...").send()
        synthetic_query_prompt = "Generate a list of legal questions that the user, who is a law student, might ask based on the summary and his previous query provided below:\n\n" + "Summary:" + gpt_summary + f"\n\nPrevious query:{query['content']}" + "\n\nInstructions: Use the summary provided to generate a list of questions that the user might ask. Output each synthetic query on a new line/."
        gpt_synthetic_queries = gpt_completion(synthetic_query_prompt, gpt_summary)
        await Message(content="Your next questions might be:\n" + gpt_synthetic_queries).send()

        # Continue message
        query = await AskUserMessage(content="Please input another question, otherwise type 'exit.'", timeout=1000).send()
        if query["content"] == "exit":
            user_continue = False
            await Message(content="Thank you for using the Law Library Chatbot!").send()
        else:
            continue

# Helper functions
def dot(a, b):
    return (sum(a*b for a,b in zip(a,b)))

def cosine_similarity(a, b):
    return dot(a,b)/((dot(a,a)**.5)*(dot(b,b)**.5))

def embed_text(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    embedding_list = response["data"][0]["embedding"]
    return embedding_list

def gpt_completion(query, text):
    completion = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that will summarise the context given to answer the user's query."},
        {"role": "user", "content": f"Content provided: {text}" + "\n\n" + "This is the user's query:" + query + "Instructions: Use the content provided to answer the user's query"},
    ]
    )
    gpt_response = completion["choices"][0]["message"]["content"]
    return gpt_response