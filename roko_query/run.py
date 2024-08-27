#!/usr/bin/env python
from roko_query.schemas import InputSchema
from naptha_sdk.utils import get_logger, load_yaml
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from ollama import Client

logger = get_logger(__name__)


def run(
    inputs: InputSchema,
    worker_nodes=None,
    orchestrator_node=None,
    flow_run=None,
    cfg=None,
):
    logger.info(f"Inputs: {inputs}")
    logger.debug(f"config = {cfg}")

    client = chromadb.PersistentClient(path=inputs.input_dir)
    collection_name = cfg["chroma"]["collection"]

    # Set the prompt
    messages = [{"role": "system", "content": cfg["inputs"]["system_message"]}]

    ef = OllamaEmbeddingFunction(
        model_name="Losspost/stella_en_1.5b_v5",
        url="http://localhost:11434/api/embeddings"
    )

    collections = client.list_collections()
    existing_collection_names = [x.name for x in collections]
    if collection_name in existing_collection_names:
        # collection = client.get_collection(name=collection_name, embedding_function=ef)
        collection = client.get_collection(name=collection_name)
        num = f"{collection_name} has {collection.count()} entries"
        logger.info(num)

        # put vector db results into query:
        results = collection.query(query_texts=inputs.question, n_results=10)
        for doc in results["documents"][0]:
            messages.append({"role": "assistant", "content": doc})

    else:
        logger.warning(f"Error: Collection {collection_name} not found.")

    messages.append({"role": "user", "content": inputs.question})

    logger.debug(messages)

    ollama_client = Client(host=cfg["models"]["ollama"]["api_base"])
    response = ollama_client.chat(
        model=cfg["models"]["ollama"]["model"], messages=messages
    )

    logger.debug(response)

    return response

