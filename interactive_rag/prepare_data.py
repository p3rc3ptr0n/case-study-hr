# Functions to the index to be loaded at runtime using data from "/data" folder.
import os
import faiss
from llama_index.core import (
    load_index_from_storage,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

EMBEDDING_DIMENSIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "text-embedding-3-small": 1536,
}


def get_embedding_model(embedding_model_name: str):
    # Initialize embedding model
    if embedding_model_name.startswith("text-embedding-3"):
        embedding_model = OpenAIEmbedding(model=embedding_model_name)
    else:
        embedding_model = HuggingFaceEmbedding(model_name=embedding_model_name)
    return embedding_model


def get_faiss_vector_store(embedding_model_name: str):
    # Get dimensions (or throw a dict key error - TODO error handling)
    # Not using dict.get here since it is pointless to instantiate
    # a faiss index with unknown dimensions, so throwing an error is better.
    d = EMBEDDING_DIMENSIONS[embedding_model_name]
    # Flat index is okay (or overkill, depending on your point of view) here,
    # would not be okay for 1M documents though.
    faiss_index = faiss.IndexFlatL2(d)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    return vector_store


def create_persist_dir(
    persist_root: str = "data/index",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    # Purpose is to prepare and store different indices in different places.
    dir_path = os.path.join(persist_root, model_name)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def read_and_index(
    input_dir: str = "data/landing",
    persist_root: str = "data/index",
    chunk_size: int = 256,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):

    # File ingestion
    # Simple directory reader seems to be sufficient for this task.
    documents = SimpleDirectoryReader(input_dir=input_dir).load_data()
    # Split docs to nodes and create vector index
    # Just setting a sliding window chunk here, nothing fancy.
    node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size)
    nodes = node_parser.get_nodes_from_documents(documents=documents)

    # Indexing
    # Initialize embedding model
    embedding_model = get_embedding_model(model_name)
    # Initialize a faiss vector store
    vector_store = get_faiss_vector_store(model_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=nodes, storage_context=storage_context, embed_model=embedding_model
    )
    # Save index to disk
    persist_dir = create_persist_dir(persist_root=persist_root, model_name=model_name)
    index.storage_context.persist(persist_dir=persist_dir)


def load_index_local(persist_dir: str = "data/index"):
    # Load index from disk
    vector_store = FaissVectorStore.from_persist_dir(persist_dir)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=persist_dir
    )
    index = load_index_from_storage(storage_context=storage_context)
    return index


if __name__ == "__main__":
    import argparse

    # Parse args
    parser = argparse.ArgumentParser(
        description="Prepare and store a faiss index from (given) data folder."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Location of data to be indexed",
        default="./data/landing",
    )
    parser.add_argument(
        "--persist_root",
        type=str,
        help="Saving location for index data",
        default="./data/index",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        help="Size of document chunks",
        default=256,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the embedding model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )

    args = parser.parse_args()

    # Run read_and_index
    read_and_index(
        input_dir=args.input_dir,
        persist_root=args.persist_root,
        chunk_size=args.chunk_size,
        model_name=args.model_name,
    )
