# libraryHackathon

## Overview
The user can upload a pdf (up to 10) to the application. The pdf is then processed, after which the user can then asks questions about the pdf

## How it works
The application is built upon chainlit. The pdf is then processed via pypdf's PDFReader and the python IO module to return the text from the pdf. 
The text is then split into smaller chunks of 200 words each. These chunks are then embedded by OpenAI's ada embedding model to generate embeddings, and these are stored locally.
When the user enters a query, the query is likewise embedded via the ada embedding model. The relevance scores of each chunk is then calculated via the cosine similarity between the chunk embedding and the query embedding.
For each pdf uploaded, the top 3 scoring chunks will be extracted. All the chunks and the queries would then be passed to GPT as context to answer the user's questions.
The user would also have the option of seeing what chunks were used to generate the response.
