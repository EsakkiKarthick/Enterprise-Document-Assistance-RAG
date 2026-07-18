import chromadb
import os
from datetime import datetime
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import fitz
from chunking import chunk_sentence_statment_aware
from pathlib import Path

VECTOR_STORE_PATH=str(Path.home() / 'Vector_Store' )
class vector_store:

    def __init__(self, collection_name: str):
        self.collection_name=collection_name
        #self.model=SentenceTransformer(model)

        os.makedirs(VECTOR_STORE_PATH,exist_ok=True)
        self.client= chromadb.PersistentClient(path=VECTOR_STORE_PATH)

        self.collection = self.client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
        print(f'Connect to collection {collection_name}')
        #print(f'Created collection count {self.collection.count()}')
    
    def add_documents(self, records: list[dict], model: str="all-MiniLM-L6-v2"):
        self.model=SentenceTransformer(model)
        if isinstance(records, dict):
            records=[records]
        if not isinstance(records, list):
            raise ValueError("Input type mismatch")
        documents=[]
        ids=[]
        metadatas=[]
        for i , r in enumerate(records):

            if not isinstance(r, dict):
                raise ValueError("Input should be dict type. Got {type(r)}")
            text=r.get('text')
            if not text:
                continue
            id=r.get('id')
            metadata=r.get('metadata',{})
            
            documents.append(text)
            ids.append(id)
            metadatas.append(metadata)

        #data=[{'data':r} for r in records]
        #ids = [r['id'] for r in records]
        #text = [r['text'] for r in records]
        #metadatas=[{k : v for k, v in r.items() if k not in ('id','text')} for r in records]
        #print('Page:', documents)
        #print('Chunk Count:', len(documents))
        #print('Metadatas', metadatas)
        embeddings=self.model.encode(documents).tolist()
        self.collection.add(
            ids = ids,
            documents = documents,
            embeddings = embeddings,
            metadatas = metadatas
        )
        print('Documents Loaded count:', str(len(documents)))
             
    def search(self, query: str, top_k: int=3, model: str ="all-MiniLM-L6-v2") ->list[dict]:
        #initialize sentence transformers model
        self.model=SentenceTransformer(model)
        query_embedding=self.model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings= [query_embedding],
            n_results= top_k,
            include = ["documents", "metadatas", "distances"]
        )

        hits=[]

        for doc_id, doc, metadata, distance in zip(
            results['ids'][0],
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            hits.append({
                "id": doc_id,
                "text": doc,
                "metadata": metadata,
                "score" : round(1-distance,4)
            })

        return hits
    
    def add_data_to_vector_store(self, filePath: str , fileName: str, max_tokens: int=125):
        data=[]
        
    #   pdfData=fitz.open(pdf)[0].get_text
        pdf= fitz.open(filePath)
        for page in pdf:
            pdfData=page.get_text()  
            chunks=chunk_sentence_statment_aware(text=pdfData, max_tokens=max_tokens, sentence_overlap=1)
            for index, chunk in enumerate(chunks):
                print(f"id: {fileName}_pg{str(page.number)}_chunk{str(index)}|text:{chunk}|metadata:source:{fileName}.page{str(page.number)}.total_pages{str(pdf.page_count)}.extracted_date{str(datetime.now())}")
                data.append({"id": fileName + "_pg"+ str(page.number) + "_chunk" + str(index), "text":chunk, 
                    "metadata":{"source":fileName, "page": str(page.number), "total_pages":str(pdf.page_count), "extracted_date": str(datetime.now())}})
        
        #vs=vector_store(collection_name=collectionName)
        self.add_documents(data)    
        #vs.add_documents(data)
        #show data in vector store


        """vs=vector_store(collection_name='kinaxis_authoringguide')
        collection=vs.show_collections()
        vectorData=collection.get(limit=50)
        print(vectorData.keys())
        for i, (doc_id, doc_text, metadata) in enumerate(zip(vectorData["ids"], vectorData["documents"], vectorData["metadatas"])):
            print(f"[{i+1}] ID: {doc_id} | Preview: {doc_text[:50]}... | Metadata: {metadata}")
        """
    def list_collections(self):
        collections=self.client.list_collections()
        if not collections:
            print("No collections found")
            return
        else:
            for col in collections:
                print(f"Collection Name:{col.name}. Records Count:{str(col.count())}")
        return

    def show_collections(self, collection_name: str, records_count: int=5):
        collections=self.client.list_collections()
        if not collections:
            print("No collections found")
            return
        else:
            #print('Collection Count:',str(collections.count()))
            for col in collections:
                if col.name==collection_name:
                    print(f"Collection Name:{col.name}. Records Count:{str(col.count())}")
                    if col.count()==0:
                        print("Collection Empty. No Data to show")
                    else:    
                        print(f"Show data for Collection:{col.name}")
                        data=col.get(
                            limit=records_count,
                            offset=0,
                            include=['documents','metadatas']
                        )
                        ids=data.get('ids',[])
                        documents=data.get('documents',[])
                        metadatas = data.get('metadatas',[])
                        for i in range(len(ids)):
                            print(f'Id:{ids[i]}|Text:{documents[i]}|Metadata:{metadatas[i]}')
                
        return
    

if __name__ == "__main__":
    """
    RAG add vector store
    """
    vs= vector_store("Employee-Handbook")
    #vs.add_data_to_vector_store(filePath= r"F:\Esakki\Learning-AI\Repo\Enterprise Document Q&A\Input\Employee-Handbook.pdf", fileName= "Employee-Handbook.pdf")
    vs.list_collections()
    vs.show_collections(collection_name="Employee-Handbook")
    #Code To show the data in vector store
    """vs= vector_store(collection_name="JobPosting")
    vs.client.list_collections()
    vs.show_collections(collection_name="JobPosting")"""