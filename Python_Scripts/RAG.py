#%%
import os
import re
from typing import List
import pandas as pd
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
#%%

os.environ["GOOGLE_API_KEY"] = "AIzaSyCLusyUJPL_sJ78vDwzkpdTXWSmGCNFT4Q"

def load_documents_from_excel(excel_path: str) -> List[Document]:
    """
    Load all sheets from an Excel file and convert each sheet's content
    into documents (with metadata indicating source and sheet name).
    """
    xls = pd.ExcelFile(excel_path)
    documents = []

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        text = df.to_csv(index=False, sep='\t')  
        
        doc = Document(page_content=text, metadata={"source": excel_path, "sheet": sheet_name})
        documents.append(doc)
    return documents


def build_vectorstore_from_documents(documents: List[Document],
                                     chunk_size: int = 1000,
                                     chunk_overlap: int = 200) -> FAISS:
    """
    Build a FAISS vectorstore from a list of documents,
    splitting them into chunks and generating embeddings.
    """
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
    """Function to join retrieved chunks into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)


rag_template = '''
You are an expert assistant in analyzing scientific data and tables extracted from Excel sheets.
You must answer user questions based only on the relevant information provided.
Always answer fully and precisely using the data provided.
Do NOT invent answers beyond the given information.

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


excel_path = os.path.join("..", "Datasets", "All_four_catalogs.xlsx")

documents = load_documents_from_excel(excel_path)
vectorstore = build_vectorstore_from_documents(documents)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
rag_chain = build_rag_chain(llm, retriever, format_documents, rag_template, language="english")
#%%
question = "Can you give me python code in less than 10 lines, extracting lines 11 to 86 for column A from the first sheet of the xlsx dataset?"
answer = rag_chain.invoke(question)

print("RAG Answer:\n", answer)
