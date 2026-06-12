import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import mlflow

load_dotenv(override=True)

# mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
# mlflow.set_experiment("LLMOps-RAG-Pinecone-Groq")
# mlflow.langchain.autolog()

# Get Groq API key
groq_api_key = os.environ['GROQ_API_KEY']
model = 'llama-3.3-70b-versatile'
# Initialize Groq Langchain chat object and conversation
groq_chat = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model
)

# Define the prompt template for generating AI responses
PROMPT_TEMPLATE = """
Human: You act as an HR expert, and provides answers to questions by using fact based and statistical information when possible.
Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
<context>
{context}
</context>

<question>
{question}
</question>

The response should be specific and use statistics or numbers when possible.
Please answer with the same language as the question.

If you don't know the answer, just say that you don't know, don't try to make up an answer. Don't hallucinate. 
Don't use any information that is not included in the context.

Assistant:"""

# Create a PromptTemplate instance with the defined template and input variables
prompt = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)

# prompt_chain = prompt | groq_chat | StrOutputParser()
# result = prompt_chain.invoke({"context": "", "question": "Siapa penulis dari module GenAI & Langchain Workshop?"})
# print(result)

embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
    )

from pinecone import Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

from langchain_pinecone import PineconeVectorStore
index_name = "karyawan-data-384"
# Instantiate the index
index = pc.Index(index_name)
vectorstore = PineconeVectorStore(index=index, embedding=embedding)

retriever = vectorstore.as_retriever()

# Define a function to format the retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the RAG (Retrieval-Augmented Generation) chain for AI response generation
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | groq_chat
    | StrOutputParser()
)

response = rag_chain.invoke("Siapa yang paling tua di antara karyawan XYZ? berikan data dan statistiknya jika memungkinkan.")
print(response)

# TRIAL Interactive loop for asking questions about employee data   

while True:
    question = input("\nTanya tentang data karyawan (ketik exit untuk keluar): ")

    if question.lower() == "exit":
        print("Program selesai")
        break

    response = rag_chain.invoke(question)
    print("\nJawaban AI:")
    print(response)
