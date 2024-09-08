#!/usr/bin/env python
from roko_query.schemas import InputSchema
from naptha_sdk.utils import get_logger
import chromadb
from openai import OpenAI
from pathlib import Path

logger = get_logger("ROKO_QUERY")


def run(
    inputs: InputSchema,
    worker_nodes=None,
    orchestrator_node=None,
    flow_run=None,
    cfg=None,
):
    """Run a query using RAG against Roko's social media streams

    Args:
        inputs (InputSchema): input_dir is expected to contain "chroma.db" a vector
        database 

    Returns:
        str: Query response
    """

    logger.info(f"Inputs: {inputs}")
    logger.debug(f"config = {cfg}")

    path = Path(inputs.input_dir) / "chroma.db"
    client = chromadb.PersistentClient(path=str(path))
    collection_name = cfg["chroma"]["collection"]

    # Set the prompt
    messages = [{"role": "system", "content": cfg["inputs"]["system_message"]}]

    collections = client.list_collections()
    existing_collection_names = [x.name for x in collections]
    if collection_name in existing_collection_names:
        collection = client.get_collection(name=collection_name)
        num = f"{collection_name} has {collection.count()} entries"
        logger.info(num)

        # put vector db results into query:
        results = collection.query(query_texts=inputs.question, n_results=10)
        for doc in results["documents"][0]:
            messages.append({"role": "assistant", "content": doc})

    else:
        logger.warning(f"Error: Collection {collection_name} not found.")
        logger.warning(f"Collections = {existing_collection_names}")

    messages.append({"role": "user", "content": inputs.question})

    # lets make sure we can see this:
    logger.error(messages)

    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    response = completion.choices[0].message.content

    return response

