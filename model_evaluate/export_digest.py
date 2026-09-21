"""Prints a compact digest of everything in model_evaluate/outputs/.

Run this after the evaluation notebook and paste the output back into the chat — it is the whole
result set condensed to a few hundred lines (aggregates, not per-utterance rows), so the report
text can be written against the real numbers.

    python model_evaluate/export_digest.py                  # print
    python model_evaluate/export_digest.py --save           # also write outputs/99_digest.txt
"""

import argparse
import io
import json
import os
import sys

import pandas as pd

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(EVAL_DIR, "outputs")
KIT_DIR = os.path.join(OUT_DIR, "40_listening_test")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 60)

buf = io.StringIO()


def out(*args):
    print(*args)
    print(*args, file=buf)


def head(title):
    out("")
    out("=" * 78)
    out(title)
    out("=" * 78)


def load(name, kit=False):
    path = os.path.join(KIT_DIR if kit else OUT_DIR, name)
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
        return df if len(df) else None
    except Exception as exc:
        out(f"  [could not read {name}: {exc}]")
        return None


def show(df, label="", max_rows=25):
    if df is None:
        out(f"  {label}: NOT PRESENT")
        return
    if label:
        out(f"  -- {label} ({len(df)} rows)")
    text = df.head(max_rows).to_string(index=False)
    out("\n".join("  " + line for line in text.splitlines()))
    if len(df) > max_rows:
        out(f"  ... {len(df) - max_rows} more rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="also write outputs/99_digest.txt")
    args = parser.parse_args()

    if not os.path.isdir(OUT_DIR):
        sys.exit("outputs/ not found — run the notebook first")

    head("0. EVALUATION ENVIRONMENT")
    env_path = os.path.join(OUT_DIR, "00_eval_environment.json")
    if os.path.isfile(env_path):
        with open(env_path, encoding="utf-8") as fh:
            env = json.load(fh)
        for key in ("evaluated_at", "platform", "processor", "cpu_count", "device", "gpu_name",
                    "torch_threads", "python", "torch", "numpy", "pandas", "librosa"):
            out(f"  {key:16s} {env.get(key)}")
        out(f"  optional         {env.get('optional_packages')}")
        out(f"  config           {json.dumps(env.get('config', {}), ensure_ascii=False)}")
    else:
        out("  00_eval_environment.json NOT PRESENT (older notebook run)")

    head("1. MODEL REGISTRY")
    show(load("01_model_registry.csv"), max_rows=10)

    head("2. FOOTPRINT (all checkpoints)")
    fp = load("02_checkpoint_footprint.csv")
    if fp is not None:
        cols = [c for c in ("run", "checkpoint", "size_mb", "n_params", "params_fp32_mb", "step",
                            "epoch", "train_loss", "eval_loss", "is_primary", "md5") if c in fp]
        show(fp[cols], max_rows=40)

    head("3. TRAINING CURVES (summary per run)")
    found = False
    for name in sorted(os.listdir(OUT_DIR)):
        if name.startswith("03_training_scalars_"):
            found = True
            df = load(name)
            if df is None:
                continue
            out(f"  -- {name}: {len(df)} points, {df['tag'].nunique()} tags, "
                f"steps {int(df['step'].min())}-{int(df['step'].max())}")
            interesting = ["TrainEpochStats/avg_loss_mel", "EvalStats/avg_loss_mel",
                           "TrainEpochStats/avg_loss_duration", "EvalStats/avg_loss_duration",
                           "TrainEpochStats/avg_loss_1", "EvalStats/avg_loss_1"]
            sub = df[df["tag"].isin(interesting)]
            if len(sub):
                agg = sub.groupby("tag")["value"].agg(["first", "min", "last", "count"]).round(3)
                show(agg.reset_index(), max_rows=12)
    if not found:
        out("  no training scalar files present")

    head("4. CORPORA")
    corpus = load("04_eval_corpus.csv")
    if corpus is not None:
        out("  prompt categories: " + corpus["category"].value_counts().to_dict().__str__())
    refs = load("04_reference_pairs.csv")
    if refs is not None and "split" in refs:
        show(refs.groupby(["run", "split"]).size().reset_index(name="n"), "reference pairs")

    head("5. RTF / LATENCY")
    rtf = load("10_rtf.csv")
    if rtf is not None:
        agg = rtf.groupby("run").agg(
            rtf_model=("rtf_model", "mean"), rtf_total=("rtf_total", "mean"),
            p50_ms=("total_ms", lambda s: s.quantile(0.5)),
            p95_ms=("total_ms", lambda s: s.quantile(0.95)),
            frontend_ms=("frontend_ms", "mean"), n=("rtf_model", "size")).round(3)
        show(agg.reset_index())
        show(rtf.groupby(["run", "category"])["rtf_total"].mean().round(3).reset_index(),
             "RTF by category", max_rows=60)

    head("6. MCD")
    mcd = load("11_mcd.csv")
    if mcd is not None:
        keys = ["run", "split"] if "split" in mcd else ["run"]
        show(mcd.groupby(keys)["mcd_db"].agg(["mean", "std", "min", "max", "count"]).round(2).reset_index())

    head("7. CHECKPOINT SWEEP")
    show(load("11_checkpoint_sweep.csv"), max_rows=40)

    head("8. F0 / PITCH")
    f0 = load("12_f0.csv")
    if f0 is not None:
        keys = ["run", "split"] if "split" in f0 else ["run"]
        cols = [c for c in ("f0_rmse_hz", "f0_rmse_cents", "f0_corr", "voicing_agreement",
                            "f0_bias_hz") if c in f0]
        show(f0.groupby(keys)[cols].mean().round(3).reset_index())

    head("9. PESQ / STOI")
    intrusive = load("13_pesq_stoi.csv")
    if intrusive is not None:
        cols = [c for c in ("pesq_wb", "stoi", "estoi") if c in intrusive]
        show(intrusive.groupby("run")[cols].agg(["mean", "count"]).round(3).reset_index())
    else:
        out("  NOT PRESENT")

    head("10. ASR WER / CER")
    asr = load("15_asr_wer.csv")
    if asr is not None:
        out(f"  recogniser: {asr['asr_model'].iloc[0] if 'asr_model' in asr else 'unknown'}")
        held = asr[~asr["rel_path"].astype(str).str.startswith("prompt:")]
        prompts = asr[asr["rel_path"].astype(str).str.startswith("prompt:")]
        cols = [c for c in ("wer_ground_truth", "wer_synth_from_ipa", "wer_synth_from_text",
                            "cer_ground_truth", "cer_synth_from_ipa", "cer_synth_from_text")
                if c in asr]
        if len(held):
            show(held.groupby("run")[cols].mean().round(3).reset_index(), "held-out (with topline)")
        if len(prompts) and "category" in prompts:
            show(prompts.groupby(["run", "category"])[["wer_synth_from_text"]].mean().round(3).reset_index(),
                 "prompt set by category", max_rows=60)
    else:
        out("  NOT PRESENT")

    head("11. UTMOS")
    utmos = load("16_utmos.csv")
    if utmos is not None:
        show(utmos.groupby(["run", "source"])["utmos"].agg(["mean", "std", "count"]).round(3).reset_index())
    else:
        out("  NOT PRESENT")

    head("12. SPEAKER SIMILARITY / CONSISTENCY")
    spk = load("17_speaker_similarity.csv")
    if spk is not None:
        out(f"  embedding: {spk['embedding'].iloc[0] if 'embedding' in spk else 'unknown'}")
        show(spk.groupby("run")["similarity_to_own_recording"].agg(["mean", "std", "count"]).round(3).reset_index())
    show(load("17_speaker_consistency.csv"), "self-consistency")

    head("13. DURATION / PROSODY")
    dur = load("18_duration_prosody.csv")
    if dur is not None:
        cols = [c for c in ("dur_mean_frames", "dur_std_frames", "frames_at_minimum_pct",
                            "duration_ratio", "timing_drift_frames", "chars_per_second") if c in dur]
        keys = ["run", "source"] if "source" in dur else ["run"]
        show(dur.groupby(keys)[cols].mean().round(3).reset_index())

    head("14. G2P / LEXICON FIDELITY")
    g2p = load("20_g2p_lexicon_fidelity.csv")
    if g2p is not None:
        show(g2p.groupby("lexicon").agg(entries=("word", "size"),
                                        exact_pct=("exact_match", lambda s: 100 * s.mean()),
                                        mean_per=("per", "mean")).round(3).reset_index())
    show(load("20_g2p_worst_mismatches.csv"), "worst mismatches", max_rows=15)

    head("15. IPA -> VOCAB COVERAGE")
    show(load("20_ipa_vocab_coverage.csv"))
    dropped = load("20_ipa_dropped_chars.csv")
    if dropped is not None:
        show(dropped.sort_values("count", ascending=False), "dropped characters", max_rows=25)

    head("16. LANGUAGE ROUTING")
    routing = load("21_language_routing_tokens.csv")
    if routing is not None:
        out(f"  token routing accuracy: {100 * routing['correct'].mean():.1f}% over {len(routing)} tokens")
        errs = routing[~routing["correct"]]
        if len(errs):
            show(errs[["token", "expected", "routed"]], "misrouted")
    sentences = load("21_language_routing_sentences.csv")
    if sentences is not None and "unspoken_digit_tokens" in sentences:
        out(f"  digit tokens surviving normalisation: {int(sentences['unspoken_digit_tokens'].sum())}")

    head("17. LEXICON COVERAGE")
    show(load("22_lexicon_coverage.csv"), max_rows=12)
    oov = load("22_lexicon_oov_top300.csv")
    if oov is not None:
        show(oov.sort_values("count", ascending=False).head(20), "top out-of-lexicon tokens")

    head("18. LATENCY UNDER LOAD")
    show(load("30_load_latency.csv"), max_rows=30)

    head("19. ROBUSTNESS")
    robust = load("31_robustness.csv")
    if robust is not None:
        robust["flags"] = robust["flags"].fillna("")
        out("  clean rate per system:")
        show(robust.groupby("run")["flags"].apply(lambda s: f"{(s == '').sum()}/{len(s)}").reset_index())
        flagged = robust[robust["flags"] != ""]
        cols = [c for c in ("run", "case", "input_chars", "audio_s", "chars_per_s",
                            "dropped_chars", "flags") if c in robust]
        show(flagged[cols], "flagged cases", max_rows=40)

    head("20. LISTENING TEST MATERIAL (counts only)")
    for name in ("mos_rating_sheet.csv", "cmos_sheet.csv", "ab_preference_sheet.csv", "abx_sheet.csv",
                 "transcription_sheet.csv", "mushra_sheet.csv", "turing_sheet.csv",
                 "ict_comprehension_sheet.csv"):
        df = load(name, kit=True)
        out(f"  {name:34s} {0 if df is None else len(df)} trials")

    head("21. SCORED LISTENING RESULTS (if any)")
    any_scores = False
    for name in sorted(os.listdir(OUT_DIR)):
        if name.startswith("41_") and name.endswith(".csv"):
            any_scores = True
            show(load(name), name, max_rows=20)
    if not any_scores:
        out("  none yet — listening tests not administered")

    head("22. ERROR TAXONOMY COUNTS")
    show(load("50_error_taxonomy_counts.csv"), max_rows=20)

    head("23. SUMMARY METRICS (full table)")
    summary = load("90_summary_metrics.csv")
    if summary is not None:
        show(summary[["group", "metric", "model", "value", "unit", "n"]], max_rows=200)

    head("24. SKIPPED METRICS")
    skips = load("91_skipped_metrics.csv")
    if skips is not None:
        show(skips[["group", "metric", "reason"]], max_rows=40)

    head("END OF DIGEST")

    if args.save:
        path = os.path.join(OUT_DIR, "99_digest.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(buf.getvalue())
        print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
