from pathlib import Path
import logging

from rag_service import write_outputs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    write_outputs(Path(__file__).resolve().parent)

