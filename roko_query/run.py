#!/usr/bin/env python
from roko_query.schemas import InputSchema
from naptha_sdk.utils import get_logger
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from pathlib import Path
import os
import json

logger = get_logger(__name__)


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

    path = Path(inputs.input_dir) / "chroma.db"
    client = chromadb.PersistentClient(path=str(path))

    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-3-small"
    )

    # Set the prompt
    messages = [{"role": "system", "content": cfg["inputs"]["system_message"]}]

    social_collection = client.get_collection(name=cfg["chroma"]["social_collection"], 
                                              embedding_function=ef)
    logger.info(f"Social collection has {social_collection.count()} entries.")
    results = social_collection.query(query_texts=inputs.question, n_results=10)
    for doc in results["documents"][0]:
        messages.append({"role": "assistant", "content": "social media: " + doc})

    doc_collection = client.get_collection(name=cfg["chroma"]["doc_collection"], 
                                           embedding_function=ef)
    logger.info(f"Doc collection has {doc_collection.count()} entries.")
    results = doc_collection.query(query_texts=inputs.question, n_results=5)
    for doc in results["documents"][0]:
        messages.append({"role": "assistant", "content": "information: " + doc})

    # if the message starts with "debug:" then also return
    # the message queue for analysis:
    debug = inputs.question[:6] == "debug:"
    if debug:
        inputs.question = inputs.question[6:]

    messages.append({"role": "user", "content": inputs.question})

    client = OpenAI()
    completion = client.chat.completions.create(model="gpt-4o-mini", messages=messages)

    response = completion.choices[0].message.content

    if debug:
        with open("messages.json", "w") as fp:
            json.dump(messages, fp, indent=2)


    return response
