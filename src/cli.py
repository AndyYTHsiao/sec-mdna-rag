import os
import questionary
from openai import OpenAI
from dotenv import load_dotenv
from dataclasses import fields, MISSING, is_dataclass
from typing import get_origin, get_args, Literal, Type, Any
from utils import list_existing_databases
from query import run_query
from builder import Builder
from config import PipelineConfig, QueryConfig
from labels import get_class_label, get_field_label


def ask_dataclass(
    cls: Type[Any],
    *,
    show_header: bool = True,
) -> Any:
    """
    Prompt the user to fill in a dataclass interactively.

    Walks through each field in the dataclass and prompts the user for a
    value. Nested dataclasses are handled recursively. Literal fields are
    shown as a select list, while other values are entered as text and
    converted to the expected field type.

    Args:
        cls (Type[Any]): The dataclass type to prompt for.
        show_header (bool): Whether to print the class name header before prompting.

    Returns:
        An instance of the provided dataclass populated with the user's answers.
    """

    answers = {}

    if show_header:
        title = get_class_label(cls.__name__)
        questionary.print(f"\n{title}", style="bold")

    for f in fields(cls):
        field_name = f.name
        field_type = f.type
        has_default = f.default is not MISSING
        default_value = f.default if has_default else None

        # Nested dataclass → always show header
        if is_dataclass(field_type):
            answers[field_name] = ask_dataclass(
                field_type,
                show_header=True,
            )
            continue

        label = get_field_label(cls.__name__, field_name)
        message = f"{label}:"

        # Literal → select
        if get_origin(field_type) is Literal:
            value = questionary.select(
                message=message,
                choices=list(get_args(field_type)),
                default=default_value,
            ).ask()

            answers[field_name] = value
            continue

        # Text input
        prompt_kwargs = {"message": message}

        if has_default:
            prompt_kwargs["default"] = str(default_value)
        else:
            prompt_kwargs["validate"] = (
                lambda x: True if x else "This field is required."
            )

        value = questionary.text(**prompt_kwargs).ask()
        answers[field_name] = _cast_value(value, field_type)

    return cls(**answers)


def _cast_value(value: str, field_type: Any) -> Any:
    """
    Cast a string input into the given dataclass field type.

    Supports int, float, and bool conversions. For bool, the values
    "true", "1", "yes", and "y" (case-insensitive) are treated as True.
    Any other field type is returned unchanged as a string.

    Args:
        value (str): The raw string entered by the user.
        field_type (Any): The expected target type for the dataclass field.

    Returns:
        The value converted to the requested type, or the original string
        if the type is not handled explicitly.

    Raises:
        ValueError: If conversion to the requested type fails.
    """
    try:
        if field_type is int:
            return int(value)
        if field_type is float:
            return float(value)
        if field_type is bool:
            return value.lower() in ("true", "1", "yes", "y")
        return value
    except Exception:
        raise ValueError(f"Cannot cast '{value}' to {field_type}")


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    query_cfg = QueryConfig()
    registry_dir = "./artifacts/registry"

    while True:
        task = questionary.select(
            "What would you like to do?",
            choices=[
                "Build a RAG database",
                "Ask a question",
                "Change query settings",
                "Exit",
            ],
        ).ask()

        if task == "Build a RAG database":
            questionary.print("\nBuild a new RAG database", style="bold")
            questionary.print("Configure the pipeline below:\n", style="bold")

            db_name = questionary.text(
                "Database name:", validate=lambda x: True if x else "Name required"
            ).ask()

            cfg = ask_dataclass(PipelineConfig, show_header=False)

            builder = Builder(cfg, client)
            builder.build_database(db_name)

        elif task == "Ask a question":
            db_name = questionary.select(
                "Select database",
                choices=list_existing_databases("./artifacts/registry"),
            ).ask()

            query = questionary.text("What would you like to know?").ask()

            response = run_query(
                query_cfg, client, query, db_name, registry_dir=registry_dir
            )

            # Print answer and citations
            questionary.print(f"\n{response['answer']}\n", style="bold")

            retrieved_docs = response.get("retrieved_docs", [])

            if retrieved_docs:
                questionary.print("\nRetrieved Chunks:\n", style="bold")

                for i, doc in enumerate(retrieved_docs, start=1):
                    chunk_id = doc.get("chunk_id", "N/A")
                    text = doc.get("text", "")

                    # Badge for top match
                    badge = " 🥇 Top Match" if i == 1 else ""

                    # Title
                    title = f"Chunk {i}{badge} | ID: {chunk_id}"

                    # Divider
                    questionary.print("=" * 60, style="fg:#888888")

                    # Title line
                    questionary.print(title, style="bold")

                    # Content label
                    questionary.print("\nContent:", style="italic")

                    # Text body
                    questionary.print(text)

                    questionary.print("\n")  # spacing

        elif task == "Change query settings":
            query_cfg = ask_dataclass(QueryConfig)

        elif task == "Exit" or KeyboardInterrupt:
            break
