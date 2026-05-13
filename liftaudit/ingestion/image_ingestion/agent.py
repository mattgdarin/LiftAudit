import base64
import mimetypes
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from liftaudit.ingestion.image_ingestion.schemas import ImageResponse
from liftaudit.ingestion.image_ingestion.state import ImageIngestionState


DEFAULT_IMAGE_MODEL = "gpt-5.4-mini"


def get_image_agent():
    print("test")
    graph = StateGraph(ImageIngestionState)
    graph.add_node("extract_image", extract_workout_csv_from_image)
    graph.add_node("human_feedback", add_human_feedback)

    graph.add_edge(START, "extract_image")
    graph.add_conditional_edges(
        "extract_image",
        route_after_extraction,
        {
            "human_feedback": "human_feedback",
            END: END,
        },
    )
    graph.add_edge("human_feedback", "extract_image")

    return graph.compile()




def extract_workout_csv_from_image(
    state: ImageIngestionState,
) -> ImageIngestionState:
    """Use a vision-capable LangChain model to extract generic CSV rows."""
    load_dotenv()
    image_data_url = _image_to_data_url(state.image_path)
    llm = ChatOpenAI(model=DEFAULT_IMAGE_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(ImageResponse)

    prompt_text = (
        "Convert this workout log image into rows for the "
        "generic LiftAudit CSV format: date, exercise, sets, "
        "reps, weight, unit, rir, notes."
    )
    if state.human_notes:
        prompt_text += (
            "\n\nA human reviewer provided this clarification. "
            "Use it to revise the extraction while still only returning rows "
            "supported by the image:\n"
            + "\n".join(f"- {note}" for note in state.human_notes)
        )
        if state.review_notes:
            prompt_text += (
                "\n\nPrevious issues of concern from the last extraction:\n"
                + "\n".join(f"- {note}" for note in state.review_notes)
            )
        if state.extraction:
            previous_rows = [
                row.model_dump() for row in state.extraction.rows
            ]
            prompt_text += (
                "\n\nPrevious extracted rows to reconsider with the clarification:\n"
                f"{previous_rows}"
            )

    extraction = structured_llm.invoke(
        [
            {
                "role": "system",
                "content": (
                    "Extract workout log entries from an image. "
                    "Return only data visible in the image. "
                    "Do not estimate, correct, or invent values. "
                    "Use blank strings for fields that are not visible."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                ],
            },
        ]
    )
    state.extraction = extraction

    
    state.needs_review = extraction.needs_review
    state.review_notes = extraction.review_notes
    return state


def route_after_extraction(state: ImageIngestionState) -> str:
    if state.needs_review and state.review_attempts < state.max_review_attempts:
        return "human_feedback"
    return END


def add_human_feedback(state: ImageIngestionState) -> ImageIngestionState:
    print(f"We flagged the response as needing review with the notes {state.review_notes}")
    resolution = input("Please give us additional context to resolve this: ").strip()
    state.review_attempts += 1
    if resolution:
        state.human_notes.append(resolution)
    else:
        state.errors.append("Human review requested, but no clarification was provided.")
    return state


def _image_to_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
