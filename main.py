from pathlib import Path
import os
import s3fs

import fitz
import asyncio
from typing import Optional
from hypercorn.config import Config
from hypercorn.asyncio import serve

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain import OpenAI, VectorDBQA

from fastapi import FastAPI, File, UploadFile, Body, Request, Depends, HTTPException, status
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
    k: Optional[int] = Field(default=None, example=4)

    @validator('k')
    def set_name(cls, k):
        return k or 4


app = FastAPI(root_path='/prod/')

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    # Convert pdf to text in memory
    if file is not None:
        stream_bytes = file.file.read()
        with fitz.open(stream=stream_bytes, filetype='pdf') as pdfreader:
            text = ""
            for page in pdfreader:
                text += page.get_text() + '\n'

    txt_file_name = os.path.splitext(file.filename)[0] + '.txt'

    url = f's3://insurochat/'
    file_path = url + txt_file_name
    # Cache text file
    s3 = s3fs.S3FileSystem(anon=False)
    with s3.open(file_path, "w") as f:
        f.write(text)

    return {"filename": file.filename}


@app.get("/listfiles/")
async def list_files_by_kwarg():
    url = f's3://insurochat/'
    s3 = s3fs.S3FileSystem(anon=False)

    file_list = s3.glob(f'{url}*.txt')

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
    k = request.k


    # url_endpoint = 'suncorp-insurance-home-contents-insurance-product-disclosure-statement'
    # Read text file
    url = f's3://insurochat/{url_endpoint}.txt'

    s3 = s3fs.S3FileSystem(anon=False)
    with s3.open(url, 'rb') as f:
        chunk_size = 1024 * 1024  # 1 MB
        contents = b''
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            contents += chunk
        home_contents = contents.decode('utf-8')

    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(home_contents)

    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)

    qa = VectorDBQA.from_chain_type(
        llm=OpenAI(temperature=0), chain_type="stuff", vectorstore=docsearch,
        k=k,  # Number of docs to query for
        return_source_documents=True
    )

    result = qa({"query": query})
    answer = result["result"]
    source_docs = result["source_documents"]

    return answer



config = Config()
config.bind = ["0.0.0.0"]

if __name__ == "__main__":
    asyncio.run(serve(app, config))
else:
    handler = Mangum(app)