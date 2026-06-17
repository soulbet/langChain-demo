from langsmith import Client as LangSmith
import getpass
import os

if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Set LangSmith API key:\n\n")

os.environ["LANGSMITH_TRACING"] = "true"

ls_client = LangSmith()

dataset_name = "multiverse-math-few-shot-examples-v2"
dataset_public_url = (
    "https://smith.langchain.com/public/620596ee-570b-4d2b-8c8f-f828adbe5242/d"
)

ls_client.clone_public_dataset(dataset_public_url)

dataset_id = ls_client.read_dataset(dataset_name=dataset_name).id

ls_client.index_dataset(dataset_id=dataset_id)
examples = ls_client.similar_examples(
    {"question": "whats the negation of the negation of the negation of 3"},
    limit=3,
    dataset_id=dataset_id,
)
len(examples)
inputs= examples[0].inputs["question"]
outputs=examples[0].outputs["conversation"]