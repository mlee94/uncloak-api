from pathlib import Path
import os
import s3fs

import fitz
import asyncio
from typing import Optional
from hypercorn.config import Config
from hypercorn.asyncio import serve

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain import OpenAI, VectorDBQA, PromptTemplate, LLMChain

import faiss
import pickle

from fastapi import FastAPI, File, UploadFile, Body, Request, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import (
    BaseModel,
    Field,
    validator,
)
import fsspec
import json
from mangum import Mangum

class Query(BaseModel):
    url_endpoint: str
    query: str = Field(example="What is the maximum cover for a rug?")
    k: Optional[int] = Field(default=4, example=4)

    @validator('k')
    def set_name(cls, k):
        return k or 4


app = FastAPI()

# CORS Settings
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    # Convert pdf to text in memory
    if file is not None:
        stream_bytes = file.file.read()
        with fitz.open(stream=stream_bytes, filetype='pdf') as pdfreader:
            doc = {}
            for idx, page in enumerate(pdfreader):
                doc[f'{idx}'] = page.get_text() + '\n'

    txt_file_name = os.path.splitext(file.filename)[0]

    text_list = []
    meta_data = []
    for (page, text) in doc.items():
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        texts = text_splitter.split_text(text)
        for txt in texts:
            meta_data.append({'page': page})
        text_list += texts

    # Ensure ada for embedding API
    embeddings = OpenAIEmbeddings(
        document_model_name="text-embedding-ada-002",
        query_model_name="text-embedding-ada-002",
    )

    url = f's3://insurochat/'
    vectorstore = FAISS.from_texts(text_list, embeddings, metadatas=meta_data)
    faiss.write_index(vectorstore.index, f"{txt_file_name}_docs.index")
    vectorstore.index = None

    # Cache text file
    s3 = s3fs.S3FileSystem(anon=False)
    # Store vectorstore in S3 without index
    with s3.open(url + f"{txt_file_name}_faiss_store.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

    return {"filename": file.filename}


@app.get("/listfiles/")
async def list_files_by_kwarg():
    url = f's3://insurochat/'
    s3 = s3fs.S3FileSystem(anon=False)

    file_list = s3.glob(f'{url}*.pkl')

    stem_list = {}
    stem_list['Filelist'] = []
    for file in file_list:
        stem_name = Path(file).stem
        stem_list['Filelist'].append(stem_name)

    return stem_list


@app.post("/query/")
async def run_langchain_model(request: Query):
    url_endpoint = request.url_endpoint
    query = request.query
    k = request.k  # Euclidean nearest neighbours
    if k > 20:
        raise ValueError('k is too high, possible high expenditure')

    # Read json file
    url = 's3://insurochat/'

    s3 = s3fs.S3FileSystem(anon=False)
    fs = fsspec.filesystem('file')

    # Check if disk cache exists
    if s3.exists(url + f"{url_endpoint}_faiss_store.pkl") and fs.exists(f"{url_endpoint}_docs.index"):
        # Read vectorstore from S3 without index
        with s3.open(url + f"{url_endpoint}_faiss_store.pkl", "rb") as f:
            vectorstore = pickle.load(f)
        index = faiss.read_index(f"{url_endpoint}_docs.index")
        vectorstore.index = index

    text_list = ""
    doc_store = vectorstore.docstore._dict.values()
    for i in doc_store:
        text_list += i.page_content

    # davinci 3 for completions API
    llm = OpenAI(model_name='text-davinci-003', temperature=0)
    token_doc_estimate = llm.get_num_tokens(text_list)
    token_query_estimate = llm.get_num_tokens(query)
    doc_length = len(doc_store)

    total_tokens = (
        token_doc_estimate + (doc_length + 1) * 14
        + (doc_length + 1) * token_query_estimate
    )
    price_estimate = total_tokens / 1000 * 0.0004  # Approx only as token includes embeddings/completions

    template = """
    Answer the question below with reference to financial information if possible, stating any exclusions, or conditions.
    
    Question: {context}
    """

    prompt = PromptTemplate(input_variables=["context"], template=template)

    qa = VectorDBQA.from_llm(
        llm=llm,
        search_type='similarity',  # mmr
        prompt=prompt,
        vectorstore=vectorstore,
        k=k,  # Number of docs to query for
        return_source_documents=True
    )
    similarity_docs = vectorstore.similarity_search_with_score(query, k=qa.k)

    # Hack to append score to metadata:
    docs = []
    for (doc, score) in similarity_docs:
        doc.metadata['euclidean_distance'] = str(score)
        docs.append(doc)

    answer, _ = qa.combine_documents_chain.combine_docs(docs, question=query)
    result = {
        'query': query,
        'result': answer,
        'source_documents': docs,
        'total tokens': str(total_tokens),
        'price_estimate_dollars': str(price_estimate)
    }

    return result



config = Config()
config.bind = ["0.0.0.0"]

if __name__ == "__main__":
    asyncio.run(serve(app, config))
else:
    handler = Mangum(app)