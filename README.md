# Rag Pipeline Playground
Using RAG with LLMs to develop a simple pdpc chatbot. The objective is to understand the RAG pipeline where I used Pinecone, Ollama models and Streamlit.
I am using hansards pdpc documents as the knowledge base.


### Install libraries and pulling of ollama models

```Python
pip install requirements.txt
```
```Python
ollama pull llama3:8b  
```
#### Embeddings model

```Python
ollama pull mxbai-embed-larged
```

### Insights
Chunking is really important. I was learning from online on what are the important chunking strategies. Simple strategy include like not chunking a document that has one page because it may let the chatbot lose his context.




