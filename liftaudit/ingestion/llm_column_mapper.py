from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


load_dotenv()

DEFAULT_MODEL = "gpt-5.4-mini"


class HeaderMappingSuggestion(BaseModel):
    mappings: List["HeaderMapping"] = Field(
        description="Mappings from original CSV headers to LiftAudit canonical columns."
    )
    unmapped_headers: List[str]
    missing_fields: List[str]
    warnings: List[str]


class HeaderMapping(BaseModel):
    original_header: str
    canonical_column: str


def suggest_header_mapping(
    headers: List[str],
    required_columns: List[str],
    optional_columns: Optional[List[str]] = None,
    model: str = DEFAULT_MODEL,
    client: Optional[OpenAI] = None,
) -> HeaderMappingSuggestion:
    """Ask the OpenAI API for a suggested CSV header mapping.

    This only suggests column names. The caller must validate and apply the
    mapping before parsing any workout rows.
    """
    optional_columns = optional_columns or []
    client = client or OpenAI()
    
    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You map workout CSV headers to LiftAudit columns. "
                    "Only use the allowed canonical column names. "
                    "Do not invent row data or transform values."
                    "Note that some of the CSV header may have severe typos, be in a different language, or otherwise not have a clear mapping"
                    "Only designate something as missing if there is not a logical match."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CSV headers: {headers}\n"
                    f"Required canonical columns: {required_columns}\n"
                    f"Optional canonical columns: {optional_columns}\n"
                    "Return conservative mapping entries from original_header to canonical_column."
                ),
            },
        ],
        text_format=HeaderMappingSuggestion,
    )
    print(response.output_parsed)

    return response.output_parsed
