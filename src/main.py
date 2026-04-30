# noinspection PyUnresolvedReferences
import faiss  # force single OpenMP init

import argparse
import json
import pathlib
import re
import sys
from typing import Dict, Optional, List, Tuple, Union, Any
import numpy as np

from rich.live import Live
from rich.console import Console
from rich.markdown import Markdown

from src.config import RAGConfig
from src.generator import answer, double_answer, dedupe_generated_text
from src.index_builder import build_index, preprocess_for_bm25
from src.instrumentation.logging import get_logger
from src.ranking.ranker import EnsembleRanker
from src.preprocessing.chunking import DocumentChunker
from src.query_enhancement import generate_hypothetical_document, contextualize_query
from src.retriever import (
    filter_retrieved_chunks, 
    BM25Retriever, 
    FAISSRetriever, 
    IndexKeywordRetriever, 
    get_page_numbers, 
    load_artifacts
)
from src.ranking.reranker import rerank

ANSWER_NOT_FOUND = "I'm sorry, but I don't have enough information to answer that question."
QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "compare",
    "define", "difference", "do", "does", "for", "how", "in", "into", "is",
    "it", "of", "on", "or", "the", "this", "to", "vs", "what", "when",
    "where", "which", "who", "why", "with"
}


def detect_query_intent(question: str) -> Dict[str, float]:
    """
    Heuristic query intent detection with confidence-like scores in [0, 1].
    """
    q = (question or "").strip().lower()
    starts = q.startswith

    definition_score = 0.0
    foundational_score = 0.0
    procedural_score = 0.0
    comparison_score = 0.0

    if starts("what is") or starts("what are") or starts("define") or "meaning of" in q:
        definition_score = 0.95
    elif re.search(r"\bdefinition\b", q):
        definition_score = 0.7

    if (
        re.search(r"\bbasics?\b", q)
        or re.search(r"\bintro(duction)?\b", q)
        or re.search(r"\bfundamentals?\b", q)
        or re.search(r"\bbeginner\b", q)
        or re.search(r"\boverview\b", q)
    ):
        foundational_score = 0.9

    if starts("how to") or starts("how do") or starts("how does") or re.search(r"\bsteps?\b", q):
        procedural_score = 0.9
    elif re.search(r"\bimplement\b|\bbuild\b|\bset up\b", q):
        procedural_score = 0.65

    if re.search(r"\bdifference between\b", q) or re.search(r"\bvs\.?\b", q) or "compare" in q:
        comparison_score = 0.9

    return {
        "definition_intent": definition_score,
        "foundational_intent": foundational_score,
        "procedural_intent": procedural_score,
        "comparison_intent": comparison_score,
    }


def extract_query_keywords(question: str) -> List[str]:
    tokens = preprocess_for_bm25(question or "")
    return [token for token in tokens if token not in QUERY_STOPWORDS]


def _normalize_scores_by_order(
    ordered: List[int],
    retrieval_scores: List[float],
) -> Dict[int, float]:
    if not ordered:
        return {}

    base_scores: Dict[int, float] = {}
    fallback_scores = [1.0 / (rank + 1) for rank in range(len(ordered))]
    raw = [
        float(retrieval_scores[rank]) if rank < len(retrieval_scores) else fallback_scores[rank]
        for rank in range(len(ordered))
    ]

    if max(raw) - min(raw) <= 1e-9:
        for rank, idx in enumerate(ordered):
            base_scores[idx] = fallback_scores[rank]
        return base_scores

    # Use softmax rather than min-max normalization so near-tied retrieval
    # scores stay near-tied and metadata can meaningfully break the tie.
    max_score = max(raw)
    exp_scores = [float(np.exp(score - max_score)) for score in raw]
    exp_total = sum(exp_scores)
    if exp_total <= 1e-12:
        for rank, idx in enumerate(ordered):
            base_scores[idx] = fallback_scores[rank]
        return base_scores

    for rank, idx in enumerate(ordered):
        base_scores[idx] = exp_scores[rank] / exp_total
    return base_scores


def _token_overlap_score(query_tokens: List[str], text: str) -> float:
    if not query_tokens or not text:
        return 0.0

    query_set = set(query_tokens)
    text_set = set(preprocess_for_bm25(text))
    if not text_set:
        return 0.0

    matched = len(query_set & text_set)
    if matched == 0:
        return 0.0

    precision = matched / len(text_set)
    recall = matched / len(query_set)
    return (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def apply_metadata_aware_scoring(
    ordered: List[int],
    retrieval_scores: List[float],
    meta: List[Dict[str, Any]],
    intent: Dict[str, float],
    query: str,
    cfg: RAGConfig,
) -> Tuple[List[int], List[float]]:
    """
    Apply post-retrieval score fusion:
      score_final = score_retrieval + alpha*typeBoost + beta*depthBoost + gamma*chapterPrior
    """
    if not ordered:
        return ordered, retrieval_scores

    enabled = getattr(cfg, "enable_metadata_scoring", True)
    if not enabled:
        return ordered, retrieval_scores

    alpha = float(getattr(cfg, "metadata_type_boost_alpha", 0.20))
    beta = float(getattr(cfg, "metadata_depth_boost_beta", 0.10))
    gamma = float(getattr(cfg, "metadata_chapter_boost_gamma", 0.15))
    delta = float(getattr(cfg, "metadata_heading_boost_delta", 0.24))
    base_weight = float(getattr(cfg, "metadata_base_score_weight", 0.65))

    query_tokens = extract_query_keywords(query)
    base_scores = _normalize_scores_by_order(ordered, retrieval_scores)

    def _type_boost(chunk_meta: Dict[str, Any]) -> float:
        chunk_type = chunk_meta.get("chunk_type", "unknown")
        boost = 0.0
        if intent["definition_intent"] > 0:
            if chunk_type == "definition":
                boost += 1.0 * intent["definition_intent"]
            elif chunk_type == "theorem":
                boost += 0.35 * intent["definition_intent"]
        if intent["procedural_intent"] > 0:
            if chunk_type == "example":
                boost += 0.7 * intent["procedural_intent"]
            elif chunk_type == "code":
                boost += 0.6 * intent["procedural_intent"]
        if intent["comparison_intent"] > 0 and chunk_type in {"definition", "theorem"}:
            boost += 0.4 * intent["comparison_intent"]
        return boost

    def _depth_boost(chunk_meta: Dict[str, Any]) -> float:
        depth = chunk_meta.get("section_depth", 2)
        try:
            depth = int(depth)
        except (TypeError, ValueError):
            depth = 2

        # Prefer moderate structure depth for explanatory responses.
        if depth in {1, 2}:
            return 1.0
        if depth == 0:
            return 0.65
        if depth == 3:
            return 0.75
        return 0.45

    def _chapter_prior(chunk_meta: Dict[str, Any]) -> float:
        # Prioritize earlier chapters for foundational queries.
        if intent["foundational_intent"] <= 0:
            return 0.0
        chapter_num = chunk_meta.get("chapter_num", 0)
        try:
            chapter_num = int(chapter_num)
        except (TypeError, ValueError):
            chapter_num = 0
        if chapter_num <= 0:
            return 0.0
        return (1.0 / chapter_num) * intent["foundational_intent"]

    def _heading_boost(chunk_meta: Dict[str, Any]) -> float:
        heading = chunk_meta.get("section", "")
        path = chunk_meta.get("section_path", "")
        hierarchy = " ".join(chunk_meta.get("section_hierarchy", []) or [])
        overlap = max(
            _token_overlap_score(query_tokens, heading),
            _token_overlap_score(query_tokens, path),
            _token_overlap_score(query_tokens, hierarchy),
        )

        # Favor heading/path matches even more for comparison and definition prompts.
        intent_scale = max(
            0.40,
            intent["definition_intent"],
            intent["comparison_intent"],
            intent["procedural_intent"] * 0.75,
        )
        return overlap * intent_scale

    rescored: List[Tuple[int, float]] = []
    for idx in ordered:
        chunk_meta = meta[idx] if idx < len(meta) else {}
        metadata_score = (
            alpha * _type_boost(chunk_meta)
            + beta * _depth_boost(chunk_meta)
            + gamma * _chapter_prior(chunk_meta)
            + delta * _heading_boost(chunk_meta)
        )
        final_score = (
            base_weight * base_scores.get(idx, 0.0)
            + (1.0 - base_weight) * metadata_score
        )
        rescored.append((idx, final_score))

    rescored.sort(key=lambda x: x[1], reverse=True)
    final_ordered = [idx for idx, _ in rescored]
    final_scores = [score for _, score in rescored]
    return final_ordered, final_scores

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Welcome to TokenSmith!")
    parser.add_argument("mode", choices=["index", "chat"], help="operation mode")
    parser.add_argument("--pdf_dir", default="data/chapters/", help="directory containing PDF files")
    parser.add_argument("--index_prefix", default="textbook_index", help="prefix for generated index files")
    parser.add_argument("--model_path", help="path to generation model")
    parser.add_argument("--system_prompt_mode", choices=["baseline", "tutor", "concise", "detailed"], default="baseline")
    
    indexing_group = parser.add_argument_group("indexing options")
    indexing_group.add_argument("--keep_tables", action="store_true")
    indexing_group.add_argument("--multiproc_indexing", action="store_true")
    indexing_group.add_argument("--embed_with_headings", action="store_true")
    parser.add_argument(
        "--double_prompt",
        action="store_true",
        help="enable double prompting for higher quality answers"
    )

    return parser.parse_args()

def run_index_mode(args: argparse.Namespace, cfg: RAGConfig):
    strategy = cfg.get_chunk_strategy()
    chunker = DocumentChunker(strategy=strategy, keep_tables=args.keep_tables)
    artifacts_dir = cfg.get_artifacts_directory()

    data_dir = pathlib.Path("data")
    print(f"Looking for markdown files in {data_dir.resolve()}...")
    md_files = sorted(data_dir.glob("*.md"))
    print(f"Found {len(md_files)} markdown files.")
    print(f"First 5 markdown files: {[str(f) for f in md_files[:5]]}")

    if not md_files:
        print("ERROR: No markdown files found in data/.", file=sys.stderr)
        sys.exit(1)

    build_index(
        markdown_file=str(md_files[0]),
        chunker=chunker,
        chunk_config=cfg.chunk_config,
        embedding_model_path=cfg.embed_model,
        artifacts_dir=artifacts_dir,
        index_prefix=args.index_prefix,
        use_multiprocessing=args.multiproc_indexing,
        use_headings=args.embed_with_headings,
    )

def use_indexed_chunks(question: str, chunks: list, cfg: RAGConfig) -> list:
    # Logic for keyword matching from textbook index
    try:
        with open(cfg.page_to_chunk_map_path, 'r') as f:
            page_to_chunk_map = json.load(f)
        with open(cfg.extracted_index_path, 'r') as f:
            extracted_index = json.load(f)
    except FileNotFoundError:
        return [], []

    keywords = get_keywords(question)
    chunk_ids = {
        chunk_id
        for word in keywords
        if word in extracted_index
        for page_no in extracted_index[word]
        for chunk_id in page_to_chunk_map.get(str(page_no), [])
    }
    ordered_chunk_ids = sorted(chunk_ids)
    return [chunks[cid] for cid in ordered_chunk_ids], ordered_chunk_ids

def get_answer(
    question: str,
    cfg: RAGConfig,
    args: argparse.Namespace,
    logger: Any,
    console: Optional["Console"],
    artifacts: Optional[Dict] = None,
    golden_chunks: Optional[list] = None,
    is_test_mode: bool = False,
    additional_log_info: Optional[Dict[str, Any]] = None
) -> Union[str, Tuple[str, List[Dict[str, Any]], Optional[str]]]:
    """
    Run a single query through the pipeline.
    """    
    chunks = artifacts["chunks"]
    sources = artifacts["sources"]
    retrievers = artifacts["retrievers"]
    ranker = artifacts["ranker"]
    meta = artifacts.get("meta", [])
    # Ensure these locals exist for all control flows to avoid UnboundLocalError
    ranked_chunks: List[str] = []
    topk_idxs: List[int] = []
    scores = []
    
    # Step 1: Get chunks (golden, retrieved, or none)
    chunks_info = None
    hyde_query = None
    if golden_chunks and cfg.use_golden_chunks:
        # Use provided golden chunks
        ranked_chunks = golden_chunks
    elif cfg.disable_chunks:
        # No chunks - baseline mode
        ranked_chunks = []
    elif cfg.use_indexed_chunks:
        ranked_chunks, topk_idxs = use_indexed_chunks(question, chunks, cfg)
    else:
        retrieval_query = question
        query_intent = detect_query_intent(question)
        if additional_log_info is not None:
            additional_log_info["query_intent"] = query_intent
        print(f"Retrieval query: {retrieval_query}")
        if cfg.use_hyde:
            retrieval_query = generate_hypothetical_document(question, cfg.gen_model, max_tokens=cfg.hyde_max_tokens)
        
        pool_n = max(cfg.num_candidates, cfg.top_k + 10)
        raw_scores: Dict[str, Dict[int, float]] = {}
        for retriever in retrievers:
            # print(f"Getting scores from retriever: {retriever.name}...")
            raw_scores[retriever.name] = retriever.get_scores(retrieval_query, pool_n, chunks)
        # TODO: Fix retrieval logging.

        # print("Raw scores from retrievers:")
        # for retriever_name, score_dict in raw_scores.items():
        #     print(f"  {retriever_name}: {list(score_dict.values())}")
        # Step 2: Ranking
        ordered, scores = ranker.rank(raw_scores=raw_scores)
        
        ordered, scores = apply_metadata_aware_scoring(
            ordered=ordered,
            retrieval_scores=scores,
            meta=meta,
            intent=query_intent,
            query=question,
            cfg=cfg,
        )


        # print(f"Ordered candidate indices after ranking: {ordered[:cfg.top_k]}")
        # print(f"Corresponding scores: {scores[:cfg.top_k]}")
        topk_idxs = filter_retrieved_chunks(cfg, chunks, ordered, meta=meta)
        ranked_chunks = [chunks[i] for i in topk_idxs]
        # print(f"Top-{cfg.top_k} chunk indices after filtering: {topk_idxs}")
        # print("Len Ranked chunks:", len(ranked_chunks))
        # print("Example ranked chunk content:", ranked_chunks[0] if ranked_chunks else "No chunks retrieved")
        
        
        # Capture chunk info if in test mode
        if is_test_mode:
            # Compute individual ranker ranks
            faiss_scores = raw_scores.get("faiss", {})
            bm25_scores = raw_scores.get("bm25", {})
            index_scores = raw_scores.get("index_keywords", {})
            
            faiss_ranked = sorted(faiss_scores.keys(), key=lambda i: faiss_scores[i], reverse=True)
            bm25_ranked = sorted(bm25_scores.keys(), key=lambda i: bm25_scores[i], reverse=True)
            index_ranked = sorted(index_scores.keys(), key=lambda i: index_scores[i], reverse=True)
            
            faiss_ranks = {idx: rank + 1 for rank, idx in enumerate(faiss_ranked)}
            bm25_ranks = {idx: rank + 1 for rank, idx in enumerate(bm25_ranked)}
            index_ranks = {idx: rank + 1 for rank, idx in enumerate(index_ranked)}
            
            chunks_info = []
            for rank, idx in enumerate(topk_idxs, 1):
                chunks_info.append({
                    "rank": rank,
                    "chunk_id": idx,
                    "content": chunks[idx],
                    "faiss_score": faiss_scores.get(idx, 0),
                    "faiss_rank": faiss_ranks.get(idx, 0),
                    "bm25_score": bm25_scores.get(idx, 0),
                    "bm25_rank": bm25_ranks.get(idx, 0),
                    "index_score": index_scores.get(idx, 0),
                    "index_rank": index_ranks.get(idx, 0),
                })

        # Step 3: Final re-ranking
        ranked_chunks = rerank(question, ranked_chunks, mode=cfg.rerank_mode, top_n=cfg.rerank_top_k)
        # print("Reranked Chunks", type(ranked_chunks), len(ranked_chunks), type(ranked_chunks[0]) if ranked_chunks else "No chunks")
        # print("Example reranked chunk content:", ranked_chunks[0] if ranked_chunks else "No chunks after reranking")

    if not ranked_chunks and not cfg.disable_chunks:
        if console:
            console.print(f"\n{ANSWER_NOT_FOUND}\n")
        return ANSWER_NOT_FOUND

    # Step 4: Generation
    model_path = cfg.gen_model
    system_prompt = args.system_prompt_mode or cfg.system_prompt_mode

    use_double = getattr(args, "double_prompt", False) or cfg.use_double_prompt

    if use_double:
        stream_iter = double_answer(
            question,
            ranked_chunks,
            model_path,
            max_tokens=cfg.max_gen_tokens,
            system_prompt_mode=system_prompt,
        )
    else:
        stream_iter = answer(
            question,
            ranked_chunks,
            model_path,
            max_tokens=cfg.max_gen_tokens,
            system_prompt_mode=system_prompt,
        )

    if is_test_mode:
        # We do not render MD in the test mode
        ans = ""
        for delta in stream_iter:
            ans += delta
        ans = dedupe_generated_text(ans)
        return ans, chunks_info, hyde_query
    else:
        # Accumulate the full text while rendering incremental Markdown chunks
        ans = render_streaming_ans(console, stream_iter)

        # Logging
        meta = artifacts.get("meta", [])
        page_nums = get_page_numbers(topk_idxs, meta)
        logger.save_chat_log(
            query=question,
            config_state=cfg.get_config_state(),
            ordered_scores=scores[:len(topk_idxs)] if 'scores' in locals() else [],
            chat_request_params={
                "system_prompt": system_prompt,
                "max_tokens": cfg.max_gen_tokens
            },
            top_idxs=topk_idxs,
            chunks=chunks,
            sources=sources,
            page_map=page_nums,
            full_response=ans,
            top_k=len(topk_idxs),
            additional_log_info=additional_log_info
        )
        return ans

def render_streaming_ans(console, stream_iter):
    ans = ""
    is_first = True
    with Live(console=console, refresh_per_second=8) as live:
        for delta in stream_iter:
            if is_first:
                console.print("\n[bold cyan]=== START OF ANSWER ===[/bold cyan]\n")
                is_first = False
            ans += delta
            live.update(Markdown(ans))
    ans = dedupe_generated_text(ans)
    live.update(Markdown(ans))
    console.print("\n[bold cyan]=== END OF ANSWER ===[/bold cyan]\n")
    return ans

def get_keywords(question: str) -> list:
    """
    Simple keyword extraction from the question.
    """
    stopwords = set([
        "the", "is", "at", "which", "on", "for", "a", "an", "and", "or", "in", 
        "to", "of", "by", "with", "that", "this", "it", "as", "are", "was", "what"
    ])
    words = question.lower().split()
    keywords = [word.strip('.,!?()[]') for word in words if word not in stopwords]
    return keywords

def run_chat_session(args: argparse.Namespace, cfg: RAGConfig):
    logger = get_logger()
    console = Console()

    print("Initializing TokenSmith Chat...")
    try:
        artifacts_dir = cfg.get_artifacts_directory()
        faiss_idx, bm25_idx, chunks, sources, meta = load_artifacts(artifacts_dir, args.index_prefix)
        print(f"Loaded {len(chunks)} chunks and {len(sources)} sources from artifacts.")
        retrievers = [FAISSRetriever(faiss_idx, cfg.embed_model), BM25Retriever(bm25_idx)]
        if cfg.ranker_weights.get("index_keywords", 0) > 0:
            retrievers.append(IndexKeywordRetriever(cfg.extracted_index_path, cfg.page_to_chunk_map_path))
        
        ranker = EnsembleRanker(ensemble_method=cfg.ensemble_method, weights=cfg.ranker_weights, rrf_k=int(cfg.rrf_k))
        print("Loaded retrievers and initialized ranker.")
        artifacts = {"chunks": chunks, "sources": sources, "retrievers": retrievers, "ranker": ranker, "meta": meta}
    except Exception as e:
        print(f"ERROR: {e}. Run 'index' mode first.")
        sys.exit(1)

    chat_history = []
    additional_log_info = {}
    print("Initialization complete. You can start asking questions!")
    print("Type 'exit' or 'quit' to end the session.")
    while True:
        print("CHAT HISTORY:", chat_history)  # Debug print to trace chat history
        try:
            q = input("\nAsk > ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            
            effective_q = q
            if cfg.enable_history and chat_history:
                try:
                    effective_q = contextualize_query(q, chat_history, cfg.gen_model)
                    additional_log_info["is_contextualizing_query"] = True
                    additional_log_info["contextualized_query"] = effective_q
                    additional_log_info["original_query"] = q
                    additional_log_info["chat_history"] = chat_history
                    print(f"Contextualized Query: {effective_q}")  # Debug print to trace contextualization
                except Exception as e:
                    print(f"Warning: Failed to contextualize query: {e}. Using original query.")
                    effective_q = q
            
            # Use the single query function. get_answer also renders the streaming markdown and takes care of logging, so we need not do anything else here.
            ans = get_answer(effective_q, cfg, args, logger, console, artifacts=artifacts, additional_log_info=additional_log_info)

            # Update Chat history (make it atomic for user + assistant turn)
            try:
                user_turn      = {"role": "user", "content": q}
                assistant_turn = {"role": "assistant", "content": ans}
                chat_history  += [user_turn, assistant_turn]
            except Exception as e:
                print(f"Warning: Failed to update chat history: {e}")
                # We can continue without chat history, so we do not break the loop here.

            # Trim chat history to avoid exceeding context window
            if len(chat_history) > cfg.max_history_turns * 2:
                chat_history = chat_history[-cfg.max_history_turns * 2:]

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
            break



def main():
    args = parse_args()
    config_path = pathlib.Path("config/config.yaml")
    if not config_path.exists(): raise FileNotFoundError("config/config.yaml not found.")
    cfg = RAGConfig.from_yaml(config_path)
    print(f"Loaded configuration from {config_path.resolve()}.")
    if args.mode == "index":
        run_index_mode(args, cfg)
    elif args.mode == "chat":
        run_chat_session(args, cfg)

if __name__ == "__main__":
    main()
