import json
import os
import streamlit as st
from dataclasses import MISSING, fields, is_dataclass
from typing import Any, Literal, Type, TypeVar, get_args, get_origin
from dotenv import load_dotenv
from openai import OpenAI
from src.builder import Builder
from src.config import BuilderConfig, QueryConfig
from src.labels import get_class_label, get_field_help, get_field_label
from src.query import run_query
from src.utils import list_existing_databases

T = TypeVar("T")


def render_dataclass_form(
    cls: Type[T],
    *,
    current: T | None = None,
    prefix: str = "",
    show_header: bool = True,
) -> T:
    """Render a dataclass as Streamlit input widgets.

    Existing values are used as widget defaults so users can edit a
    configuration without resetting fields that they leave unchanged.

    Args:
        cls: Dataclass type to render.
        current: Optional existing instance whose values become defaults.
        prefix: Prefix added to Streamlit widget keys to keep them unique.
        show_header: Whether to show the human-readable dataclass heading.

    Returns:
        A new instance of ``cls`` populated with the current widget values.

    Raises:
        TypeError: If ``cls`` is not a dataclass type or ``current`` is not an
            instance of ``cls``.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass type")

    if current is not None and not isinstance(current, cls):
        raise TypeError(
            f"Expected current to be an instance of {cls.__name__}, "
            f"got {type(current).__name__}"
        )

    values: dict[str, Any] = {}

    if show_header:
        st.subheader(get_class_label(cls.__name__))

    for dataclass_field in fields(cls):
        field_name = dataclass_field.name
        field_type = dataclass_field.type
        key = f"{prefix}{field_name}"
        current_value = (
            getattr(current, field_name)
            if current is not None
            else _get_field_default(dataclass_field)
        )

        if is_dataclass(field_type):
            nested_current = current_value if current_value is not MISSING else None
            values[field_name] = render_dataclass_form(
                field_type,
                current=nested_current,
                prefix=f"{key}_",
                show_header=True,
            )
            continue

        label = get_field_label(cls.__name__, field_name)
        help_text = get_field_help(cls.__name__, field_name)

        if get_origin(field_type) is Literal:
            options = list(get_args(field_type))
            default_index = (
                options.index(current_value) if current_value in options else 0
            )
            values[field_name] = st.selectbox(
                label,
                options=options,
                index=default_index,
                key=key,
                help=help_text,
            )
            continue

        if field_type is bool:
            values[field_name] = st.checkbox(
                label,
                value=(bool(current_value) if current_value is not MISSING else False),
                key=key,
                help=help_text,
            )
            continue

        if field_type is int:
            values[field_name] = int(
                st.number_input(
                    label,
                    value=(int(current_value) if current_value is not MISSING else 0),
                    step=1,
                    key=key,
                    help=help_text,
                )
            )
            continue

        if field_type is float:
            values[field_name] = float(
                st.number_input(
                    label,
                    value=(
                        float(current_value) if current_value is not MISSING else 0.0
                    ),
                    key=key,
                    help=help_text,
                )
            )
            continue

        values[field_name] = st.text_input(
            label,
            value=(str(current_value) if current_value is not MISSING else ""),
            key=key,
            help=help_text,
        )

    return cls(**values)


def _get_field_default(dataclass_field: Any) -> Any:
    """Return a dataclass field's default value.

    Args:
        dataclass_field: Dataclass field to inspect.

    Returns:
        The declared default, the result of its default factory, or
        ``dataclasses.MISSING`` when neither exists.
    """
    if dataclass_field.default is not MISSING:
        return dataclass_field.default

    if dataclass_field.default_factory is not MISSING:
        return dataclass_field.default_factory()

    return MISSING


def initialize_session_state() -> None:
    """Initialize persistent configuration and query-result state."""
    if "builder_cfg" not in st.session_state:
        st.session_state.builder_cfg = BuilderConfig()

    if "query_cfg" not in st.session_state:
        st.session_state.query_cfg = QueryConfig()

    if "expand_all_chunks" not in st.session_state:
        st.session_state.expand_all_chunks = False

    if "last_response" not in st.session_state:
        st.session_state.last_response = None

    if "last_query" not in st.session_state:
        st.session_state.last_query = ""

    if "last_database" not in st.session_state:
        st.session_state.last_database = None


def render_build_page(client: OpenAI) -> None:
    """Render the database-building workflow.

    Args:
        client: Authenticated OpenAI client passed to ``Builder``.
    """
    st.title("Build a RAG Database")

    builder_cfg: BuilderConfig = st.session_state.builder_cfg
    db_name = st.text_input("Database name", key="build_database_name")

    updated_builder_cfg = render_dataclass_form(
        BuilderConfig,
        current=builder_cfg,
        prefix="build_builder_",
        show_header=False,
    )

    if not st.button(
        "Build Database",
        type="primary",
        key="build_database_button",
    ):
        return

    if not db_name.strip():
        st.error("Database name is required.")
        return

    st.session_state.builder_cfg = updated_builder_cfg

    try:
        with st.spinner("Building database..."):
            builder = Builder(updated_builder_cfg, client)
            builder.build_database(db_name.strip())
    except Exception as exc:
        st.error(f"Failed to build database: {exc}")
        return

    st.success(f"Database '{db_name.strip()}' built successfully.")


def render_query_page(client: OpenAI) -> None:
    """Render the question-answering workflow.

    Args:
        client: Authenticated OpenAI client passed to ``run_query``.
    """
    st.title("Ask a Question")

    builder_cfg: BuilderConfig = st.session_state.builder_cfg
    query_cfg: QueryConfig = st.session_state.query_cfg
    databases = list_existing_databases(builder_cfg.paths.registry_dir)

    if not databases:
        st.warning("No databases found. Build one first.")
        return

    db_name = st.selectbox(
        "Select database",
        options=databases,
        key="query_database",
    )
    query = st.text_input("Your question", key="query_text")

    if (
        query != st.session_state.last_query
        or db_name != st.session_state.last_database
    ):
        st.session_state.last_response = None

    if st.button("Ask", type="primary", key="ask_button"):
        if not query.strip():
            st.error("Please enter a question.")
            return

        try:
            with st.spinner("Running query..."):
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
                    query=query.strip(),
                    db_name=db_name,
                    company_info=company_info,
                )
        except FileNotFoundError as exc:
            st.error(f"Required file was not found: {exc}")
            return
        except json.JSONDecodeError as exc:
            st.error(f"The company metadata file contains invalid JSON: {exc}")
            return
        except Exception as exc:
            st.error(f"Query failed: {exc}")
            return

        st.session_state.last_response = response
        st.session_state.last_query = query
        st.session_state.last_database = db_name
        st.session_state.expand_all_chunks = False

    render_query_response()


def render_query_response() -> None:
    """Render the most recent answer and its retrieved chunks."""
    response = st.session_state.last_response

    if response is None:
        return

    st.markdown("### Answer")
    st.write(response.get("answer", ""))

    retrieved_docs = response.get("retrieved_docs", [])

    if not retrieved_docs:
        st.info("No chunks were retrieved.")
        return

    heading_col, button_col = st.columns([5, 1])

    with heading_col:
        st.markdown(f"### Retrieved Chunks ({len(retrieved_docs)})")

    with button_col:
        button_label = (
            "Collapse All" if st.session_state.expand_all_chunks else "Expand All"
        )
        if st.button(button_label, type="tertiary", key="toggle_all_chunks"):
            st.session_state.expand_all_chunks = not st.session_state.expand_all_chunks
            st.rerun()

    st.divider()

    for index, doc in enumerate(retrieved_docs, start=1):
        chunk_id = doc.get("chunk_id")
        text = doc.get("text", "")
        badge = " 🥇 Top Match" if index == 1 else ""
        title = f"Chunk {index}{badge}"

        if chunk_id:
            title += f" | ID: {chunk_id}"

        expanded = st.session_state.expand_all_chunks or index == 1

        with st.expander(title, expanded=expanded):
            st.markdown("**Content**")
            st.write(text)


def render_query_settings_page() -> None:
    """Render the form used to edit and save ``QueryConfig`` settings."""
    st.title("Query Configuration")

    query_cfg: QueryConfig = st.session_state.query_cfg
    updated_query_cfg = render_dataclass_form(
        QueryConfig,
        current=query_cfg,
        prefix="query_settings_",
        show_header=False,
    )

    if st.button(
        "Save Settings",
        type="primary",
        key="save_query_settings",
    ):
        st.session_state.query_cfg = updated_query_cfg
        st.session_state.last_response = None
        st.success("Query settings updated.")


def main() -> None:
    """Configure and run the Streamlit application."""
    load_dotenv()

    st.set_page_config(
        page_title="RAG System",
        page_icon="🔎",
        layout="wide",
    )

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("OPENAI_API_KEY is not set. Add it to your environment or .env file.")
        st.stop()

    client = OpenAI(api_key=api_key)
    initialize_session_state()

    st.sidebar.title("RAG System")
    task = st.sidebar.radio(
        "Select Task",
        [
            "Build Database",
            "Ask a Question",
            "Query Settings",
        ],
    )

    if task == "Build Database":
        render_build_page(client)
    elif task == "Ask a Question":
        render_query_page(client)
    elif task == "Query Settings":
        render_query_settings_page()


if __name__ == "__main__":
    main()
