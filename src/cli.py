import os
import json
import questionary
from dataclasses import (
    MISSING,
    fields,
    is_dataclass,
)
from typing import Any, Literal, Type, TypeVar, get_args, get_origin
from dotenv import load_dotenv
from openai import OpenAI
from .builder import Builder
from .config import BuilderConfig, QueryConfig
from .labels import get_class_label, get_field_label
from .query import run_query
from .utils import list_existing_databases

T = TypeVar("T")


def ask_dataclass(
    cls: Type[T],
    *,
    current: T | None = None,
    show_header: bool = True,
) -> T:
    """
    Prompt the user to configure a dataclass interactively.

    Existing values can be provided through ``current``. These values are
    displayed as defaults, allowing the user to edit an existing
    configuration without resetting unchanged fields.

    Args:
        cls (Type[T]):
            The dataclass type to instantiate.
        current (T | None):
            An optional existing instance whose values are used as defaults.
        show_header (bool):
            Whether to display a heading for the dataclass.

    Returns:
        T: A configured instance of ``cls``.

    Raises:
        - TypeError: If ``cls`` is not a dataclass type or ``current`` is not
            an instance of ``cls``.

        - KeyboardInterrupt: If the user cancels a prompt.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass type")

    if current is not None and not isinstance(current, cls):
        raise TypeError(
            f"Expected current to be an instance of {cls.__name__}, "
            f"got {type(current).__name__}"
        )

    answers: dict[str, Any] = {}

    if show_header:
        title = get_class_label(cls.__name__)
        questionary.print(f"\n{title}", style="bold")

    for f in fields(cls):
        field_name = f.name
        field_type = f.type

        current_value = (
            getattr(current, field_name)
            if current is not None
            else _get_field_default(f)
        )

        # Nested dataclass
        if is_dataclass(field_type):
            answers[field_name] = ask_dataclass(
                field_type,
                current=current_value,
                show_header=True,
            )
            continue

        label = get_field_label(cls.__name__, field_name)
        message = f"{label}:"

        # Literal fields
        if get_origin(field_type) is Literal:
            value = questionary.select(
                message=message,
                choices=list(get_args(field_type)),
                default=current_value,
            ).ask()

            answers[field_name] = _require_answer(value)
            continue

        # Boolean fields are clearer as a select prompt
        if field_type is bool:
            value = questionary.select(
                message=message,
                choices=[
                    questionary.Choice("Yes", value=True),
                    questionary.Choice("No", value=False),
                ],
                default=current_value,
            ).ask()

            answers[field_name] = _require_answer(value)
            continue

        prompt_kwargs: dict[str, Any] = {"message": message}

        if current_value is not MISSING:
            prompt_kwargs["default"] = str(current_value)
        else:
            prompt_kwargs["validate"] = lambda value: (
                True if value.strip() else "This field is required."
            )

        value = questionary.text(**prompt_kwargs).ask()
        value = _require_answer(value)

        answers[field_name] = _cast_value(value, field_type)

    return cls(**answers)


def _get_field_default(dataclass_field: Any) -> Any:
    """
    Return a dataclass field's default value.

    Returns ``MISSING`` when the field has neither a default nor a
    default factory.

    Args:
        dataclass_field (Any): The dataclass field to inspect.

    Returns:
        default value (Any): The default value of the field, or ``MISSING`` if none exists.
    """
    if dataclass_field.default is not MISSING:
        return dataclass_field.default

    if dataclass_field.default_factory is not MISSING:
        return dataclass_field.default_factory()

    return MISSING


def _require_answer(value: T | None) -> T:
    """
    Convert a cancelled questionary prompt into KeyboardInterrupt.
    """
    if value is None:
        raise KeyboardInterrupt

    return value


def _cast_value(value: str, field_type: Any) -> Any:
    """
    Cast text input to the requested field type.

    Args:
        value (str): The text input to cast.
        field_type (Any): The target type to cast to.

    Returns:
        casted value (Any): The input value cast to the requested type.
    """
    try:
        if field_type is int:
            return int(value)

        if field_type is float:
            return float(value)

        if field_type is bool:
            normalized = value.strip().lower()

            if normalized in {"true", "1", "yes", "y"}:
                return True

            if normalized in {"false", "0", "no", "n"}:
                return False

            raise ValueError(
                "Boolean values must be one of: true, false, yes, no, 1, or 0"
            )

        return value

    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot cast {value!r} to {field_type}") from exc


def main() -> None:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment or .env file."
        )

    client = OpenAI(api_key=api_key)
    builder_cfg = BuilderConfig()
    paths_cfg = builder_cfg.paths
    query_cfg = QueryConfig()

    try:
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

            if task is None or task == "Exit":
                break

            if task == "Build a RAG database":
                questionary.print(
                    "\nBuild a new RAG database",
                    style="bold",
                )
                questionary.print(
                    "Configure the pipeline below:\n",
                    style="bold",
                )

                db_name = questionary.text(
                    "Database name:",
                    validate=lambda value: True if value.strip() else "Name required",
                ).ask()

                if db_name is None:
                    continue

                # Use current config values as defaults.
                builder_cfg = ask_dataclass(
                    BuilderConfig,
                    current=builder_cfg,
                    show_header=False,
                )

                builder = Builder(builder_cfg, client)
                builder.build_database(db_name)
                questionary.print(
                    f"\n[✓] Database '{db_name}' ready.\n", style="bold fg:green"
                )

            elif task == "Ask a question":
                databases = list_existing_databases(paths_cfg.registry_dir)

                if not databases:
                    questionary.print(
                        "\nNo databases found. Build one first.\n",
                        style="bold",
                    )
                    continue

                db_name = questionary.select(
                    "Select database",
                    choices=databases,
                ).ask()

                if db_name is None:
                    continue

                query = questionary.text("What would you like to know?").ask()

                if query is None or not query.strip():
                    continue

                company_info = None

                if query_cfg.filter_chunks:
                    with open(
                        query_cfg.company_info_path,
                        "r",
                        encoding="utf-8",
                    ) as file:
                        company_info = json.load(file)

                response = run_query(
                    query_cfg=query_cfg,
                    client=client,
                    query=query,
                    db_name=db_name,
                    company_info=company_info,
                )

                print_response(response)

            elif task == "Change query settings":
                query_cfg = ask_dataclass(
                    QueryConfig,
                    current=query_cfg,
                )

                questionary.print(
                    "\n[✓] Query settings updated.\n", style="bold fg:green"
                )

    except KeyboardInterrupt:
        questionary.print("\nExiting.", style="bold")


def print_response(response: dict[str, Any]) -> None:
    """
    Print a query response and its retrieved chunks.

    Args:
        response (dict[str, Any]): The LLM's response.
    """
    questionary.print(
        f"\n{response['answer']}\n",
        style="bold",
    )

    retrieved_docs = response.get("retrieved_docs", [])

    if not retrieved_docs:
        return

    questionary.print(
        "\nRetrieved Chunks:\n",
        style="bold",
    )

    for index, doc in enumerate(retrieved_docs, start=1):
        chunk_id = doc.get("chunk_id", "N/A")
        text = doc.get("text", "")
        badge = " 🥇 Top Match" if index == 1 else ""
        title = f"Chunk {index}{badge} | ID: {chunk_id}"

        questionary.print("=" * 60, style="fg:#888888")
        questionary.print(title, style="bold")
        questionary.print("\nContent:", style="italic")
        questionary.print(text)
        questionary.print("")


if __name__ == "__main__":
    main()
