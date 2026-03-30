import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from dataclasses import fields, MISSING, is_dataclass
from typing import get_origin, get_args, Literal, Type, Any
from utils import list_existing_databases
from query import run_query
from builder import Builder
from config import PipelineConfig, QueryConfig
from labels import get_class_label, get_field_label, get_field_help


def render_dataclass_form(
    cls: Type[Any],
    prefix: str = "",
    show_header: bool = True,
) -> Any:
    """Render a dataclass as a Streamlit input form and return the filled instance.

    Args:
        cls (Type[Any]): The dataclass type to render.
        prefix (str): Optional prefix to apply to Streamlit widget keys.
        show_header (bool): Whether to render a section header for this dataclass.

    Returns:
        An instance of `cls` populated with user input from the rendered form.
    """
    values = {}

    # Section Header
    if show_header:
        title = get_class_label(cls.__name__)
        st.subheader(title)

    # Fields
    for f in fields(cls):
        field_name = f.name
        field_type = f.type

        has_default = f.default is not MISSING
        default_value = f.default if has_default else None

        key = f"{prefix}{field_name}"

        # Nested Dataclass
        if is_dataclass(field_type):
            values[field_name] = render_dataclass_form(
                field_type,
                prefix=f"{key}_",
                show_header=True,
            )

            continue

        # Labels + Help
        label = get_field_label(
            cls.__name__,
            field_name,
        )

        help_text = get_field_help(
            cls.__name__,
            field_name,
        )

        # Literal → Dropdown
        if get_origin(field_type) is Literal:
            options = list(get_args(field_type))

            default_index = (
                options.index(default_value) if default_value in options else 0
            )

            values[field_name] = st.selectbox(
                label,
                options,
                index=default_index,
                key=key,
                help=help_text,
            )

            continue

        # Bool
        if field_type is bool:
            values[field_name] = st.checkbox(
                label,
                value=default_value if has_default else False,
                key=key,
                help=help_text,
            )

            continue

        # Integer
        if field_type is int:
            values[field_name] = st.number_input(
                label,
                value=int(default_value) if has_default else 0,
                step=1,
                key=key,
                help=help_text,
            )

            continue

        # Float
        if field_type is float:
            values[field_name] = st.number_input(
                label,
                value=float(default_value) if has_default else 0.0,
                key=key,
                help=help_text,
            )

            continue

        # Text
        values[field_name] = st.text_input(
            label,
            value=str(default_value) if has_default else "",
            key=key,
            help=help_text,
        )

    return cls(**values)


if __name__ == "__main__":
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    registry_dir = "./artifacts/registry"

    if "query_cfg" not in st.session_state:
        st.session_state.query_cfg = QueryConfig()

    # Sidebar navigation
    st.sidebar.title("RAG System")
    task = st.sidebar.radio(
        "Select Task",
        [
            "Build Database",
            "Ask Question",
            "Query Settings",
        ],
    )

    # Build database
    if task == "Build Database":
        st.title("Build a RAG Database")

        db_name = st.text_input("Database name")

        cfg = render_dataclass_form(
            PipelineConfig,
            prefix="pipeline_",
            show_header=False,
        )

        if st.button("Build Database"):
            if not db_name:
                st.error("Database name required")
            else:
                with st.spinner("Building database..."):
                    builder = Builder(cfg, client)
                    builder.build_database(db_name)

                st.success("Database built successfully!")

    # Ask question
    elif task == "Ask Question":
        st.title("Ask a Question")

        dbs = list_existing_databases(registry_dir)

        if not dbs:
            st.warning("No databases found.")
            st.stop()

        db_name = st.selectbox(
            "Select database",
            dbs,
        )

        query = st.text_input("Your question")

        # Session state initialization
        if "expand_all_chunks" not in st.session_state:
            st.session_state.expand_all_chunks = False

        if "last_response" not in st.session_state:
            st.session_state.last_response = None

        if "last_query" not in st.session_state:
            st.session_state.last_query = ""

        # Clear stale results if query changes
        if query != st.session_state.last_query:
            st.session_state.last_response = None

        # Ask button
        if st.button("Ask"):
            if not query:
                st.error("Please enter a question.")

            else:
                with st.spinner("Running query..."):
                    response = run_query(
                        st.session_state.query_cfg,
                        client,
                        query,
                        db_name,
                        registry_dir=registry_dir,
                    )

                st.session_state.last_response = response
                st.session_state.last_query = query

                # Reset expand state
                st.session_state.expand_all_chunks = False

        # Render result
        if st.session_state.last_response is not None:
            response = st.session_state.last_response

            st.markdown("### Answer")
            st.write(response["answer"])

            retrieved_docs = response["retrieved_docs"]

            if retrieved_docs:
                num_chunks = len(retrieved_docs)
                col1, col2 = st.columns([5, 1])

                with col1:
                    st.markdown(f"### Retrieved Chunks ({num_chunks})")

                with col2:
                    if st.button(
                        "Expand All"
                        if not st.session_state.expand_all_chunks
                        else "Collapse All",
                        type="tertiary",
                    ):
                        st.session_state.expand_all_chunks = (
                            not st.session_state.expand_all_chunks
                        )

                        st.rerun()

                st.divider()

                for i, doc in enumerate(retrieved_docs, start=1):
                    chunk_id = doc.get("chunk_id")
                    text = doc.get("text", "")

                    badge = " 🥇 Top Match" if i == 1 else ""

                    title = f"Chunk {i}{badge}"

                    if chunk_id:
                        title += f" | ID: {chunk_id}"

                    expanded_state = st.session_state.expand_all_chunks or i == 1

                    expander_key = f"chunk_{i}_{st.session_state.expand_all_chunks}"

                    with st.expander(
                        title,
                        expanded=expanded_state,
                        key=expander_key,
                    ):
                        st.markdown("**Content**")
                        st.write(text)

    # Query settings
    elif task == "Query Settings":
        st.title("Query Configuration")

        new_cfg = render_dataclass_form(QueryConfig, prefix="query_", show_header=False)

        if st.button("Save Settings"):
            st.session_state.query_cfg = new_cfg

            st.success("Settings updated!")
