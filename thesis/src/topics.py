from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.preprocess import load_records_jsonl
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BERTopic narrative analysis on misinfo texts")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        default=[
            "data/processed/constraint_all.jsonl",
            "data/processed/mmcovar_all.jsonl",
        ],
        help="One or more JSONL files (will filter label=misinfo)",
    )
    parser.add_argument("--output", type=str, default="results/topics")
    parser.add_argument(
        "--source-filter",
        type=str,
        default=None,
        help="Optional: only keep records whose source contains this string",
    )
    parser.add_argument("--min-topic-size", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=800,
        help="Truncate long news bodies for embedding speed/quality",
    )
    return parser.parse_args()


def collect_misinfo_texts(
    paths: list[str],
    *,
    source_filter: str | None,
    max_chars: int,
) -> tuple[list[str], list[dict]]:
    texts: list[str] = []
    meta: list[dict] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"[topics] skip missing: {p}")
            continue
        records = load_records_jsonl(p)
        for r in records:
            if r.label != "misinfo":
                continue
            if source_filter and (not r.source or source_filter not in r.source):
                continue
            text = (r.text or "").strip()
            if len(text) < 20:
                continue
            if len(text) > max_chars:
                text = text[:max_chars]
            texts.append(text)
            meta.append(
                {
                    "id": r.id,
                    "source": r.source,
                    "platform": r.platform,
                    "split": r.split,
                }
            )
    return texts, meta


def suggest_theme_name(keywords: list[str]) -> str:
    """Lightweight heuristic labels for thesis taxonomy (edit manually if needed)."""
    joined = " ".join(keywords).lower()
    rules = [
        (("mrna", "dna", "genetic", "gene"), "mRNA / genetic alteration claims"),
        (("microchip", "chip", "tracking", "5g", "bill gates"), "Surveillance / microchip / 5G conspiracies"),
        (("side effect", "death", "died", "adverse", "injury", "blood clot"), "Side-effect exaggeration / harm narratives"),
        (("infertility", "fertility", "pregnant", "pregnancy"), "Fertility / pregnancy fear narratives"),
        (("mandate", "passport", "forced", "freedom"), "Mandate / coercion / freedom framing"),
        (("pharma", "profit", "pfizer", "moderna", "big pharma"), "Pharma distrust / profit motives"),
        (("natural", "immunity", "ivermectin", "hydroxychloroquine", "cure"), "Alternative cure / natural immunity claims"),
        (("hoax", "plandemic", "fake virus", "exaggerat"), "Pandemic / threat denial narratives"),
        (("child", "kids", "school"), "Children / school vaccination concerns"),
        (("efficacy", "ineffective", "does not work", "useless"), "Vaccine inefficacy claims"),
    ]
    for keys, name in rules:
        if any(k in joined for k in keys):
            return name
    if keywords:
        return f"Theme: {', '.join(keywords[:4])}"
    return "Unlabeled theme"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    topics_cfg = cfg.get("topics", {})
    min_topic_size = args.min_topic_size or int(topics_cfg.get("min_topic_size", 15))
    top_n = args.top_n or int(topics_cfg.get("top_n_topics", 10))

    texts, meta = collect_misinfo_texts(
        args.input,
        source_filter=args.source_filter,
        max_chars=args.max_chars,
    )
    print(f"[topics] misinfo docs={len(texts)} from {args.input}")
    if len(texts) < max(min_topic_size * 2, 30):
        raise RuntimeError(
            f"Too few misinfo docs ({len(texts)}) for BERTopic. "
            "Check input paths / labels."
        )

    # Import heavy deps only when running
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    print("[topics] loading embedding model all-MiniLM-L6-v2 ...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Adaptive min_topic_size for smaller corpora
    adaptive_min = min(min_topic_size, max(8, len(texts) // 40))
    print(f"[topics] min_topic_size={adaptive_min} top_n={top_n}")

    topic_model = BERTopic(
        embedding_model=embedder,
        min_topic_size=adaptive_min,
        nr_topics="auto",
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _probs = topic_model.fit_transform(texts)

    info = topic_model.get_topic_info()
    # Drop outlier topic -1 from "top" list but keep in full info
    non_outlier = info[info["Topic"] != -1].head(top_n)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Save full topic info
    info_path = out / "topic_info.csv"
    info.to_csv(info_path, index=False)

    # Per-doc assignments
    assignments = []
    for i, topic_id in enumerate(topics):
        assignments.append({**meta[i], "topic": int(topic_id), "text_preview": texts[i][:240]})
    assign_path = out / "doc_topics.jsonl"
    with assign_path.open("w", encoding="utf-8") as f:
        for row in assignments:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Taxonomy table for thesis
    taxonomy = []
    examples_by_topic: dict[int, list[str]] = {}
    for i, topic_id in enumerate(topics):
        examples_by_topic.setdefault(int(topic_id), []).append(texts[i])

    md_lines = [
        "# Post-pandemic vaccine hesitancy / misinfo narratives",
        "",
        f"- Documents analyzed: **{len(texts)}** (label=`misinfo`)",
        f"- Inputs: `{', '.join(args.input)}`",
        f"- Embedding model: `all-MiniLM-L6-v2`",
        f"- BERTopic `min_topic_size={adaptive_min}`",
        "",
        "| Topic | Count | Suggested theme | Top keywords | Example |",
        "|------:|------:|-----------------|--------------|---------|",
    ]

    for _, row in non_outlier.iterrows():
        tid = int(row["Topic"])
        count = int(row["Count"])
        words = topic_model.get_topic(tid) or []
        keywords = [w for w, _ in words[:8]]
        theme = suggest_theme_name(keywords)
        ex = examples_by_topic.get(tid, [""])[0].replace("\n", " ")
        if len(ex) > 160:
            ex = ex[:157] + "..."
        taxonomy.append(
            {
                "topic_id": tid,
                "count": count,
                "suggested_theme": theme,
                "keywords": keywords,
                "example": examples_by_topic.get(tid, [""])[0][:500],
            }
        )
        md_lines.append(
            f"| {tid} | {count} | {theme} | {', '.join(keywords[:6])} | {ex} |"
        )

    # Outlier summary
    outlier_n = sum(1 for t in topics if t == -1)
    md_lines += [
        "",
        f"Outlier topic (-1): **{outlier_n}** docs (too unique / noisy for a cluster).",
        "",
        "## Notes for thesis",
        "",
        "- Theme names are **heuristic**; rename after qualitative review.",
        "- CONSTRAINT posts are short social claims; MMCoVaR bodies are longer news text.",
        "- Use this table in Chapter 5 (Narrative Analysis).",
        "",
    ]

    tax_json = out / "taxonomy.json"
    tax_md = out / "taxonomy.md"
    tax_json.write_text(json.dumps(taxonomy, indent=2, ensure_ascii=False), encoding="utf-8")
    tax_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Persist model for later inspection
    model_dir = out / "bertopic_model"
    topic_model.save(model_dir, serialization="safetensors", save_ctfidf=True, save_embedding_model=False)

    summary = {
        "n_docs": len(texts),
        "n_topics_non_outlier": int((info["Topic"] != -1).sum()),
        "outlier_docs": outlier_n,
        "min_topic_size": adaptive_min,
        "top_n": top_n,
        "outputs": {
            "topic_info": str(info_path),
            "doc_topics": str(assign_path),
            "taxonomy_json": str(tax_json),
            "taxonomy_md": str(tax_md),
            "model_dir": str(model_dir),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"[topics] taxonomy -> {tax_md}")
    print("[topics] DONE")


if __name__ == "__main__":
    main()
