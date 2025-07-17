import os
from typing import List
from dotenv import load_dotenv
import fitz
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY no encontrada en el archivo .env")


def load_documents_from_pdfs(folder_path: str) -> List[Document]:
    documents = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            doc = fitz.open(filepath)
            text = ""
            for page in doc:
                text += page.get_text()
            documents.append(Document(page_content=text, metadata={"source": filename}))
    return documents


def build_vectorstore_from_documents(documents: List[Document],
                                     chunk_size: int = 1000,
                                     chunk_overlap: int = 200) -> FAISS:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(documents)

    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vectorstore = FAISS.from_documents(documents=splits, embedding=embedding)
    return vectorstore


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)


def format_documents(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_template = '''
You are an expert assistant in analyzing scientific papers.
Answer questions only based on the provided information.
Do NOT invent answers beyond the given context.

Relevant information: {context}
Question: {question}
Language: {language}
Helpful answer:
'''


def build_rag_chain(llm, retriever, format_function, rag_template, language="english"):
    retriever_chain = retriever | format_function
    rag_prompt = PromptTemplate.from_template(rag_template)
    return (
        {
            "context": retriever_chain,
            "question": RunnablePassthrough(),
            "language": lambda _: language
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )


papers_folder = os.path.join("..", "Papers")


documents = load_documents_from_pdfs(papers_folder)
vectorstore = build_vectorstore_from_documents(documents)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
rag_chain = build_rag_chain(llm, retriever, format_documents, rag_template, language="english")


question = "What are the main catalogs of high-mass X-ray binaries mentioned in these papers?"
answer = rag_chain.invoke(question)

print("RAG Answer:\n", answer)
