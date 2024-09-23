from roko_query.schemas import InputSchema
from roko_query.run import run
from naptha_sdk.utils import get_logger
import yaml
from pathlib import Path
import argparse
from datetime import datetime
import logging


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Q&A on roko-query.")
    parser.add_argument("data_path", type=str, help="Path to the data directory")
    args = parser.parse_args()
    data_path = Path(args.data_path)

    # Suppress logging so we can capture the output markdown:
    logging.getLogger("roko_query.run").setLevel(logging.WARNING) 

    cfg_path = "../roko_query/component.yaml"
    with open(cfg_path, "r") as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    questions = [
        "What are the upcoming milestones and developments planned for the Roko Network?",
        "Can you give me an overview of the project's roadmap?",
        "Explain tokenonomics of Roko Network, including the token supply, distribution, and swaps/collabs.",
        "Can you provide a high-level explanation of the underlying technology and architecture of the Roko Network? How does it work with decentralized blockchain and AI?",
        "Provide the latest updates on the Roko Network's development progress?",
        "What new features or improvements have been recently implemented?",
        "Can you tell me what's the news, any events organized by the Roko Network team coming up? How can I participate in these things?"
    ]

    questions = questions[2:4]

    print("# Roko Query Napth Module\n")
    print(f"Date: {datetime.now()}\n")

    for i, q in enumerate(questions):

        print(f"## Q{i}: _{q}_\n")

        inputs = InputSchema(
            question=questions[0],
            input_dir=str(data_path)
        )

        response = run(inputs=inputs, cfg=cfg)
        print(response + "\n")