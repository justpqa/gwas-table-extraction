import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz
import re
import warnings
warnings.filterwarnings("ignore")

CORPUS_PATH = "./test_papers"
CHROMA_DB_PATH = "./chroma_db"
CHROMA_DB_COLLECTION_NAME = "gwas_paper_collection"
EMBEDDING_MODEL_NAME = "NeuML/pubmedbert-base-embeddings"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def clean_text(text: str) -> str:
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove tabs and newlines
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # Remove special characters and punctuation (keep basic ones if needed)
    text = re.sub(r'[^a-z0-9\s\.\,\-]', '', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_document(document: Document) -> Document:
    cleaned_text = clean_text(document.page_content)
    cleaned_document = Document(page_content=cleaned_text, metadata=document.metadata)
    return cleaned_document

def ingest_corpus(corpus_path: str = CORPUS_PATH, chroma_db_path: str = CHROMA_DB_PATH,
                  chroma_db_collection_name: str = CHROMA_DB_COLLECTION_NAME, 
                  embedding_model_name: str = EMBEDDING_MODEL_NAME,
                  chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP, print_progress: bool = True) -> None:
    # load documents from corpus
    if print_progress:
        print("Loading documents from corpus...")    
    documents = []
    metadata = [] # store a list of metadata dictionaries, currenly only have filenames
    for filename in os.listdir(corpus_path):
        pmid_pmcid = filename.split(".")[0]
        pmid, pmcid = pmid_pmcid.split("_")
        if filename.endswith(".txt"):
            with open(os.path.join(corpus_path, filename), 'r', encoding='utf-8') as f:
                documents.append(f.read())
                metadata.append({"PMID": pmid, "PMCID": pmcid})
        elif filename.endswith(".pdf"):
            pdf_path = os.path.join(corpus_path, filename)
            with fitz.open(pdf_path) as doc:
                text = ""
                for page in doc:
                    text += page.get_text() + "\n\n" # special indicator of pages
                documents.append(text)
                metadata.append({"PMID": pmid, "PMCID": pmcid})
    if print_progress:
        print(f"Finished loading {len(documents)} documents.")
        print()

    # split documents into chunks
    if print_progress:
        print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    splitted_documents = text_splitter.create_documents(texts=documents, metadatas=metadata)
    splitted_documents = [clean_document(doc) for doc in splitted_documents]
    if print_progress:
        print(f"Finished splitting to make {len(splitted_documents)} chunks.")
        print()

    # create Chroma vector store
    if print_progress:
        print("Creating Chroma vector store...")
    chroma_db = Chroma(
        persist_directory=chroma_db_path,
        embedding_function=HuggingFaceEmbeddings(model_name=embedding_model_name),
        collection_name=chroma_db_collection_name,
        collection_metadata={"hnsw:space": "cosine"}
    )
    # delete current collection contents before adding new documents
    collection = chroma_db._collection
    all_docs = collection.get(include=[])
    all_ids = all_docs["ids"]
    if all_ids:
        collection.delete(ids=all_ids)
    # add documents to Chroma vector store
    chroma_db.add_documents(splitted_documents)
    if print_progress:
        print("Finished creating Chroma vector store.")
        print()

def verify_db_existence(chroma_db_path: str = CHROMA_DB_PATH,
                        chroma_db_collection_name: str = CHROMA_DB_COLLECTION_NAME) -> bool:
    chroma_db = Chroma(
        persist_directory=chroma_db_path,
        embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME),
        collection_name=chroma_db_collection_name,
        collection_metadata={"hnsw:space": "cosine"}
    )
    collection = chroma_db._collection
    all_docs = collection.get(include=[])
    all_ids = all_docs["ids"]
    return len(all_ids) > 0

if __name__ == "__main__":
    ingest_corpus()
    verify_db_existence()