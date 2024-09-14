import os
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from ai_service import (
    generate_answer_from_file,
    get_docs_from_json,
    get_docs_from_pdf,
)
from decouple import config


# Create uploads directory if not existing
UPLOAD_DIRECTORY = config("UPLOAD_DIRECTORY", default="uploads")
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


app = FastAPI()


# ==============================================
#               Pydantic Models
# ==============================================
class QuestionJson(BaseModel):
    """
    Pydantic modal schema to valudate question file input json format
    """

    questions: list[str] = Field(min_length=1)


# ==============================================
#               Exception Handlers
# ==============================================
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """
    Exception handler to pretty print Pydantic Validation Errors
    """

    return JSONResponse(
        status_code=422,
        content={"detail": "Validation Error", "errors": exc.errors()},
    )


# ==============================================
#               Route Handlers
# ==============================================
@app.post("/retrieve/")
def create_upload_file(
    questions_json_file: UploadFile, reference_file: UploadFile
):
    """
    Route handler for document submission API
    """

    json_raw = questions_json_file.file.read()

    # Convert bytes into utf-8 string
    try:
        json_raw = json_raw.decode("utf-8")
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="json file must contain valid utf-8 characters only",
        )

    # Validate Json file content format
    data = QuestionJson.model_validate_json(json_raw)

    # Generate unique upload name
    upload_file_path = (
        f"./{UPLOAD_DIRECTORY}/{uuid.uuid4()}_{reference_file.filename}"
    )

    # Save file at above path
    with open(upload_file_path, "wb+") as file_object:
        file_object.write(reference_file.file.read())

    # Extract langchain docs from uploaded file
    if reference_file.content_type == "application/json":
        documents = get_docs_from_json(upload_file_path)
    elif reference_file.content_type == "application/pdf":
        documents = get_docs_from_pdf(upload_file_path)
    else:
        raise HTTPException(
            status_code=422,
            detail="Unsupported file type for reference_file. ",
        )

    # Invoke RAG Inference and generate answers
    try:
        results = generate_answer_from_file(data.questions, documents)

        # Prepare response in required format
        response = {result["input"]: result["answer"] for result in results}

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Something went wrong",
        )

    return response
