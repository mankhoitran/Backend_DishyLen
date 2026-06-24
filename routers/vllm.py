import logging
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from agent.clients.vllm_client import VLLMClient
from agent.food_agent import VLLMFoodAgent
from agent.search import DuckDuckGoSearchService
from api.deps import get_optional_current_user
from configs.configs import get_settings
from db.database import get_db
from db.models import User
from schemas import request as request_schemas
from schemas import response as response_schemas
from services.ocr_service import (
    apply_ocr_prompt,
    corrected_items_from_text,
    extract_menu_items,
    ocr_menu_image,
    select_menu_item,
)
from services.summary_service import SummaryService
from services.translation_service import TranslationService
from utils.formatters import (
    build_sources,
    normalize_dish_detail,
    normalize_summary_fields,
    short_summary,
    sources_to_text,
    to_str_list,
)

logger = logging.getLogger(__name__)
settings = get_settings()

UPLOAD_DIR = Path(__file__).resolve().parent.parent / settings.uploads_dir

router = APIRouter()

async def save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "").suffix
    filename = f"{uuid4().hex}{suffix}"
    image_path = UPLOAD_DIR / filename
    try:
        image_path.write_bytes(await file.read())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc
    return image_path

def make_image_url(image_path: Path) -> str:
    return f"/uploads/{image_path.name}"

def get_safe_path(path_str: str) -> Path:
    p = Path(path_str)
    # Very basic safety check, actual implementation may vary
    return p

@router.post("/query", response_model=response_schemas.DishDetailResponse)
def query_dish(
    payload: request_schemas.QueryRequest,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_optional_current_user),
) -> response_schemas.DishDetailResponse:
    """Process food query through the vLLM-backed agent."""
    try:
        agent = VLLMFoodAgent(db=db)
        user_allergies = current_user.allergies if current_user else None
        result = agent.run(
            query=payload.query,
            target_language=payload.target_language,
            user_allergies=user_allergies,
        )
        ingredients: list[str] = []
        try:
            vllm_client = VLLMClient()
            search_service = DuckDuckGoSearchService(
                vllm_client,
                max_results=settings.duckduckgo_max_results,
            )
            ingredient_payload = search_service.get_dish_ingredients(
                result.get("dish", payload.query)
            )
            ingredients = to_str_list(ingredient_payload.get("ingredients"))
        except Exception:
            ingredients = []

        return normalize_dish_detail(
            result,
            fallback_name=payload.query,
            ingredients=ingredients,
            sources=[],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to process query: {exc}") from exc


@router.post("/summary", response_model=response_schemas.SummaryResponse)
def summarize(
    payload: request_schemas.SummaryRequest,
    current_user: User | None = Depends(get_optional_current_user),
) -> response_schemas.SummaryResponse:
    """Summarize either raw text or search-backed sources."""
    if not payload.text and not payload.query:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'query'.")

    try:
        vllm_client = VLLMClient()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    summary_fields = response_schemas.SummaryFields()
    sources: list[str] = []
    input_type = "text"
    user_allergies = current_user.allergies if current_user else None

    try:
        summary_service = SummaryService(vllm_client)

        if payload.text:
            raw_summary = summary_service.summarize_text(payload.text, payload.max_words, user_allergies)
            summary_fields = normalize_summary_fields(raw_summary)
        else:
            query = (payload.query or "").strip()
            if not query:
                raise HTTPException(
                    status_code=400,
                    detail="Query is required when text is empty.",
                )
            input_type = "search"
            search_service = DuckDuckGoSearchService(
                vllm_client,
                max_results=settings.duckduckgo_max_results,
            )
            raw_sources = search_service.search_sources(query)
            source_text = sources_to_text(raw_sources)
            if source_text:
                raw_summary = summary_service.summarize_text(source_text, payload.max_words, user_allergies)
                summary_fields = normalize_summary_fields(raw_summary)
            else:
                summary_fields = response_schemas.SummaryFields(
                    description="No sources found.",
                    summary="No sources found.",
                )
            if payload.include_sources:
                sources = build_sources(raw_sources)

        description_text = summary_fields.description or summary_fields.summary
        summary_text = summary_fields.summary or short_summary(description_text)
        if payload.target_language and description_text:
            translation_svc = TranslationService()
            translation_svc.vllm_client = vllm_client
            translated_payload = translation_svc.translate_text(description_text, payload.target_language)
            translated = translated_payload.get("translated_text", description_text)
            description_text = translated
            summary_text = short_summary(translated)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to summarize: {exc}") from exc

    return response_schemas.SummaryResponse(
        summary=summary_text,
        description=description_text,
        calories=summary_fields.calories,
        protein=summary_fields.protein,
        carbs=summary_fields.carbs,
        fats=summary_fields.fats,
        ingredients=summary_fields.ingredients,
        allergens=summary_fields.allergens,
        sources=sources,
        input_type=input_type,
        target_language=payload.target_language,
    )


@router.post("/ocr/upload", response_model=response_schemas.OCRUploadResponse)
async def upload_menu_image(file: UploadFile = File(...)) -> response_schemas.OCRUploadResponse:
    """Upload a menu image and return a path reference for OCR."""
    image_path = await save_upload(file)
    return response_schemas.OCRUploadResponse(
        image_path=str(image_path),
        image_url=make_image_url(image_path),
    )


@router.post("/ocr/items", response_model=response_schemas.OCRMenuResponse)
def ocr_menu_items(
    payload: request_schemas.OCRMenuRequest,
    current_user: User | None = Depends(get_optional_current_user),
) -> response_schemas.OCRMenuResponse:
    """Extract menu items from an OCR image."""
    try:
        ocr_result = ocr_menu_image(payload.image_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    user_allergies = current_user.allergies if current_user else None
    items = extract_menu_items(ocr_result.text, max_items=payload.max_items)
    ocr_payload = apply_ocr_prompt(
        ocr_result.text,
        fallback_items=items,
        prefer_backend=payload.ocr_backend,
        user_allergies=user_allergies,
    )
    corrected_items = corrected_items_from_text(
        ocr_payload.get("corrected_text", ""),
        max_items=payload.max_items,
    )
    if corrected_items:
        items = corrected_items

    return response_schemas.OCRMenuResponse(
        image_path=ocr_result.image_path,
        image_url=make_image_url(Path(ocr_result.image_path)),
        ocr_status=ocr_result.status,
        ocr_text=ocr_result.text,
        raw_text=ocr_payload.get("raw_text", ""),
        corrected_text=ocr_payload.get("corrected_text", ""),
        items=items,
    )


@router.post("/ocr/select", response_model=response_schemas.OCRSelectResponse)
def ocr_menu_select(
    payload: request_schemas.OCRSelectRequest,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_optional_current_user),
) -> response_schemas.OCRSelectResponse:
    """Process selection of an OCR menu item."""
    image_path = get_safe_path(payload.image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    item_name = payload.item_name
    if not item_name:
        raise HTTPException(status_code=400, detail="Must provide item_name or item_index")

    agent = VLLMFoodAgent(db=db)
    user_allergies = current_user.allergies if current_user else None
    result = agent.run(
        query=item_name,
        target_language=payload.target_language,
        user_allergies=user_allergies,
    )

    try:
        ocr_result = ocr_menu_image(payload.image_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    items = extract_menu_items(ocr_result.text, max_items=payload.max_items)
    ocr_payload = apply_ocr_prompt(
        ocr_result.text,
        fallback_items=items,
        prefer_backend=payload.ocr_backend,
        user_allergies=user_allergies,
    )
    corrected_items = corrected_items_from_text(
        ocr_payload.get("corrected_text", ""),
        max_items=payload.max_items,
    )
    if corrected_items:
        items = corrected_items

    try:
        selected_item = select_menu_item(items, payload.item_name, payload.item_index)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        vllm_client = VLLMClient()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        agent = VLLMFoodAgent(db=db)
        dish_payload = agent.run(query=selected_item, target_language=payload.target_language)

        ingredients: list[str] = []
        if payload.include_ingredients:
            search_service = DuckDuckGoSearchService(
                vllm_client,
                max_results=settings.duckduckgo_max_results,
            )
            ingredient_payload = search_service.get_dish_ingredients(selected_item)
            raw_ingredients = ingredient_payload.get("ingredients")
            ingredients = to_str_list(raw_ingredients)
        dish_info = normalize_dish_detail(
            dish_payload,
            fallback_name=selected_item,
            ingredients=ingredients,
            sources=[],
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dish info: {exc}") from exc

    return response_schemas.OCRSelectResponse(
        image_path=ocr_result.image_path,
        image_url=make_image_url(Path(ocr_result.image_path)),
        ocr_status=ocr_result.status,
        raw_text=ocr_payload.get("raw_text", ""),
        corrected_text=ocr_payload.get("corrected_text", ""),
        selected_item=selected_item,
        dish_info=dish_info,
        ingredients=ingredients,
        items=items,
    )


@router.post("/ocr/dish-info", response_model=response_schemas.DishDetailResponse)
def get_dish_info(
    payload: request_schemas.DishInfoRequest,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_optional_current_user),
) -> response_schemas.DishDetailResponse:
    """Fetch dish information for a named dish without re-running OCR."""
    try:
        agent = VLLMFoodAgent(db=db)
        user_allergies = current_user.allergies if current_user else None
        dish_payload = agent.run(
            query=payload.item_name,
            target_language=payload.target_language,
            user_allergies=user_allergies,
        )

        ingredients: list[str] = []
        if payload.include_ingredients:
            try:
                vllm_client = VLLMClient()
                search_service = DuckDuckGoSearchService(
                    vllm_client,
                    max_results=settings.duckduckgo_max_results,
                )
                ingredient_payload = search_service.get_dish_ingredients(payload.item_name)
                ingredients = to_str_list(ingredient_payload.get("ingredients"))
            except Exception:
                ingredients = []

        return normalize_dish_detail(
            dish_payload,
            fallback_name=payload.item_name,
            ingredients=ingredients,
            sources=[],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dish info: {exc}") from exc


@router.post("/translate", response_model=response_schemas.TranslationResponse)
def translate_text(payload: request_schemas.TranslateRequest) -> response_schemas.TranslationResponse:
    """Translate arbitrary text to English or Vietnamese."""
    try:
        service = TranslationService()
        lang = payload.target_language or payload.language or "en"
        result = service.translate_text(payload.text, lang)
        return response_schemas.TranslationResponse(
            original_text=payload.text,
            translated_text=result.get("translated_text", payload.text),
            target_language=lang,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Translation failed: {exc}") from exc
