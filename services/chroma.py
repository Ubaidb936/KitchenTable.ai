import getpass
import os
from langchain_openai import OpenAIEmbeddings
import chromadb
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_core.documents import Document

api_key = ""

class VectorDB:
    def __init__(self, collection_name: str):
        self.persistent_client = chromadb.PersistentClient("data/chroma_langchain_db")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
        
        self.vector_store = Chroma(
            client=self.persistent_client,
            collection_name=collection_name,
            embedding_function=self.embeddings  # Path to persist data
        )    

    def add_document(self, text_converstion: str, image_path: str):
        uuid = str(uuid4())
        document = Document(
            page_content=text_converstion,
            metadata={"image_path": image_path},
            id=uuid,
        )
        self.vector_store.add_documents(documents=[document], ids=[uuid])
        return uuid

    def update_document(self, text_converstion: str, image_path: str, uuid: str):
        updated_document = Document(
            page_content=text_converstion,
            metadata={"image_path": image_path},
            id=uuid,
            )
        self.vector_store.update_document(document_id=uuid, document=updated_document)
        return uuid

    def remove_document(self, uuid: str):
        self.vector_store.delete(ids=[uuid])

    def search(self, query: str):
        results = self.vector_store.similarity_search(
            query,
            k=1)
        if len(results) > 0:
            res = results[0]
            return {
                "conversation": res.page_content,
                "image_path": res.metadata["image_path"]
            }
        else:
            return {
            "conversation": "",
            "image_path": ""
            }

    def strict_search(self, query: str):
        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": 0.1},
        )
        res = retriever.invoke(query)

        if len(res) > 0:
            return {
            "conversation": res[0].page_content,
            "image_path": res[0].metadata["image_path"]
            }
        else:
            return {
            "conversation": "",
            "image_path": ""
            }
        
