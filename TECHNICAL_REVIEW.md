# Technical Review — Fine-Tuning Local LLM for Drilling & Completions

**Reviewer role:** Independent senior technical review (ML/fine-tuning, RAG, evaluation, drilling/completions engineering, editorial, reproducibility).
**Method:** Full repo read (chapters, code, tests, data, appendices, READMEs); six parallel focused audits, each independently verifying claims against actual code execution, pytest runs, grep searches over the real Utah FORGE DDR archive, and hand-derivation of model math; cross-checked and re-verified directly by the lead reviewer for every CRITICAL/HIGH finding below. No files were modified during the review itself.
**Scope:** All 13 chapters + Part 0, all appendices, `book/code/`, `book/tests/`, `book/datasets/`, both READMEs, `CLAUDE.md`, `CHANGELOG.md`.

## Remediation Status: ALL FINDINGS FIXED

Every finding below (1 CRITICAL, 2 HIGH, 7 MEDIUM, 8 LOW) has since been corrected in the repository, applied incrementally with the author's approval at each severity tier. Each finding now carries a **Status** line describing the actual change made. A new fast, non-slow regression test was added to `book/tests/test_chapter_01.py` as part of the MED-07 fix; the full fast suite was re-run after every batch of changes and stayed green throughout (final run: **74 passed, 0 failed**, up from the pre-fix 73/73 because of that new test). No chapter was restructured and no result was retracted — every fix was a bounded correction (a replaced example, a reworded sentence or two, a clarifying addition, a corrected number).

Independent, real reproduction highlights (evidence this book generally follows its own "every number comes from a real run" rule):
- Chapter 5's full LoRA fine-tune was re-run end-to-end (real Qwen2.5-1.5B-Instruct, CPU, `torch.manual_seed(0)`) and reproduced **digit-for-digit**: losses `4.7173/1.2688/0.3360/0.1850/0.1201`, training baseline `0/16`, held-out baseline `0/2`, training recall `13/16`, held-out generalization `0/2`.
- Chapter 5's trainable-parameter counts (`2,179,072 / 1,545,893,376 = 0.1410%` for the main config, `1,089,536` for the challenge config) were independently hand-derived from Qwen2.5-1.5B-Instruct's real `config.json` (GQA, hidden_size=1536, 28 layers) and matched exactly.
- Chapter 9's retrieval comparison (BM25 4/4 vs. dense-embedding 3/4, held-out report ranked exactly 8th of 677 by dense embeddings) was independently reproduced exactly.
- The held-out report (`#37`) was confirmed absent from every training JSONL by direct grep (0 hits across ~669 lines), and 7 automated tests independently assert this exclusion.
- All fast tests (`pytest -m "not slow"`) pass: 73/73 at time of review, **74/74 after remediation** (one new regression test added, see MED-07).

---

## Findings

### CRITICAL

**ID:** CRIT-01
**Severity:** CRITICAL
**Confidence:** High (verified directly — grepped every PDF in `book/datasets/full_training_set/` and `sample_training_set/` for the quoted strings; zero matches)
**Category:** result-fabrication / source-traceability
**File:** `book/chapters/chapter_04.qmd`
**Chapter:** 4
**Line:** 49–62 (the excerpt itself), 341 (the exercise referencing it)
**Current statement/behaviour:** Chapter 4 presents `"input": "0600-1130 POOH FR 9,842' TO CHANGE BHA. ERRATIC TORQUE OBSD LAST 3 STDS."` as "a realistic instruction/response pair built the same way Chapter 2's `Drilling_038` examples were, using report shorthand exactly as written," then the practical exercise at line 341 refers to it unambiguously as **"`Drilling_038`'s excerpt above."**
**Finding:** This exact string does not appear in `Drilling_038` or any of the other 75 reports in the archive. It is a synthetic/illustrative example explicitly attributed to a real, named, checkable source report.
**Evidence:** Direct grep of the extracted PDF text for `"ERRATIC TORQUE"`, `"9,842"`, `"POOH"` near this depth in `Drilling_038` and across all 76 archive PDFs returns nothing.
**Why it matters:** This is precisely the failure mode CLAUDE.md's own "one non-negotiable rule" exists to prevent — every quote in a chapter must come from an actual run against real data, never invented. Chapter 1 explicitly trains the reader to trust report-number-tagged excerpts as "quoted exactly as filed, typos and all." A reader who opens `Drilling_038` to check this passage — exactly the behavior this book cultivates — will not find it, which is a direct, checkable credibility failure, not a stylistic nitpick.
**Recommended correction:** Either (a) replace with a genuine excerpt actually drawn from the archive, or (b) if a synthetic/composite example is pedagogically preferable here, label it explicitly as illustrative and remove the "`Drilling_038`'s excerpt above" attribution at line 341.
**Requires author judgement:** YES
**Status:** ✅ FIXED (option a). The excerpt now uses a genuine, verified line from `Drilling_017`'s real TIME BREAKDOWN table (`"TOOH (trip out of hole) with BHA #2. Break off bit, lay down Scout RSS (rotary steerable)."`), confirmed word-for-word against the source PDF. The surrounding sentence, the `POOH`/`BHA` callback, and the practical exercise (which previously sent readers looking for fictional terms in `Drilling_038`) were all updated to match, pointing readers to `Drilling_017` and its two genuinely-appearing, self-glossed abbreviations (`RSS`, `TIH`).

---

### HIGH

**ID:** HIGH-01
**Severity:** HIGH
**Confidence:** High (verified directly)
**Category:** privacy/governance overclaim
**File:** `README.md`
**Chapter:** N/A (root README, reader's first contact)
**Line:** 124–129
**Current statement/behaviour:** "Keeping the workflow local also gives you direct control over where operational data is processed, which matters for drilling and completions reports subject to company confidentiality and data-governance requirements — not every hosted deployment handles that the same way, but a local model **sidesteps the question entirely**."
**Finding:** "Sidesteps the question entirely" overclaims: running locally changes *who* is responsible for confidentiality, access control, retention, and regulatory compliance — it does not eliminate those obligations. This directly matches the pattern Phase 17 of this review's own instructions flags as needing strict scrutiny ("local = automatically safe/compliant"), and is notably less careful than this same book's own better-hedged language elsewhere (`appendix/appendix_a_environment_setup.qmd` §4 and the Chapter 9 Production Reality note, both of which correctly tell readers to check their own data-classification policy).
**Evidence:** Direct read of README.md lines 115–134; cross-checked against Appendix A §4 wording (reported by a parallel reviewer as correctly hedged).
**Why it matters:** This is the very first document a prospective reader (including a company's IT/security reviewer deciding whether to allow the book's workflow) encounters, before any chapter has taught the nuance. An inaccurate governance claim here is a real credibility and even liability-adjacent risk.
**Recommended correction:** Replace "sidesteps the question entirely" with something like "removes the network-transmission part of that question but doesn't remove your organization's own confidentiality, retention, or access-control obligations for the data itself."
**Requires author judgement:** NO (the fix is a factual precision correction, not a design decision)
**Status:** ✅ FIXED. `README.md` now reads: "...A local model answers the 'does this leave my machine' part of that question, but not the rest of it: your organisation's own confidentiality, retention, and access-control policies still apply to the data itself (Appendix A, Section 4 covers this in more depth)." — matching the book's own established, better-hedged phrasing rather than inventing new wording. `book/README.md` does not duplicate this passage, so no second fix was needed there.

**ID:** HIGH-02
**Severity:** HIGH
**Confidence:** High (verified directly against all three CI workflow files)
**Category:** reproducibility / code-book mismatch
**File:** `README.md:18,557`; `book/README.md:212`; `.github/workflows/tests-{linux,macos,windows}.yml:41`
**Chapter:** N/A (repo-level docs)
**Current statement/behaviour:** "CI runs the full suite on Linux, Windows, and macOS on every push... and all three are green" / "run on Linux, Windows, and macOS on every push."
**Finding:** All three CI workflows run `pytest -v -m "not slow"`, not the full suite. Verified directly: 100 tests collected total, only 73 run in CI (27 deselected via the `slow` marker), including **100% of Chapter 1's tests** (all three are marked `slow` and never run in CI on any OS).
**Evidence:** `pytest --collect-only -q` → 100 tests; `pytest --collect-only -q -m "not slow"` → "73/100 tests collected (27 deselected)"; all three workflow YAMLs contain the identical line `run: pytest -v -m "not slow"`.
**Why it matters:** Overstates automated verification strength — exactly the trap this review's own Rule 10 warns against ("never assume passing automated tests validates experimental conclusions"). A reader could reasonably believe every chapter's model-loading and fine-tuning code is continuously verified on all three platforms; it isn't — the slow tests (which include real model loads and training) are evidently run locally/manually, not in CI.
**Recommended correction:** Reword to something like "CI runs the fast, deterministic subset of the suite (`pytest -m \"not slow\"`) on all three OSes on every push; slow tests (model downloads, fine-tuning) are run locally before each chapter ships." Consider adding at least one non-slow smoke test for Chapter 1 so it has *some* CI coverage.
**Requires author judgement:** NO for the wording fix; YES for whether to add Chapter 1 CI coverage.
**Status:** ✅ FIXED, both parts. `README.md`'s badge line and "Automated Tests" section, and `book/README.md`'s "Running tests" section, now accurately describe CI as running only the fast, deterministic subset (`pytest -m "not slow"`) on all three OSes, with slow tests run locally. Chapter 1 CI coverage was also added (see MED-07) and the test counts were re-verified and corrected after that addition: 81 chapter tests / 101 total / 74 non-slow (was 80/100/73 before the new test).

---

### MEDIUM

**ID:** MED-01
**Severity:** MEDIUM
**Confidence:** High (verified directly)
**Category:** metric-validity / mislabeling
**File:** `book/code/chapter_11/eval_finetuned_model.py:52-53`; referenced as "strict exact-match" in `README.md:134` and used from Chapter 8 onward
**Chapter:** 8, 11, 12, 13
**Current statement/behaviour:** The metric is called "exact-match" / "strict exact-match" throughout the book (e.g. "0/50 on a strict exact-match").
**Finding:** The actual implementation is `expected.strip().lower() in generated.strip().lower()` — a case-insensitive **substring containment** check, not string equality. This is the opposite of "strict": a generated response containing extra text before/after the expected answer still counts as a match. The book does separately and correctly explain (Ch11) that exact match can penalize a correct-but-differently-worded answer — but it never flags that its own "exact match" is actually a lenient containment test, which cuts the opposite direction (it can credit a match that only coincidentally contains the right substring amid irrelevant text).
**Evidence:** Direct read of `eval_finetuned_model.py:52-53`; direct read of `README.md:131-134`.
**Why it matters:** Phase 7 of this review specifically asks whether exact-match's real behavior is communicated accurately. Calling a containment check "strict" could lead a reader to over-trust a passing score, or under-trust the very real `0/50`/`0/8` failures the book reports (which remain valid failures — a `0` under a lenient containment check is if anything a *stronger* negative result than under true string equality, so this mislabeling doesn't undermine the negative findings, only the description of the check's strictness).
**Recommended correction:** Rename the function/label to `contains_expected` or similarly accurate wording, or add one sentence clarifying "exact match here means the answer contains the expected phrase verbatim, not that the two strings are identical."
**Requires author judgement:** NO
**Status:** ✅ FIXED via the clarifying-sentence option (code left untouched, per this repo's own rule that prose, not inline comments, should carry this explanation). `README.md` now says the model's answer "had to *contain* the report's exact wording, word for word, to count" (was "had to match"). `book/chapters/chapter_11.qmd` Step 2 now states explicitly that the renamed `exact_match()` function "still does exactly what Chapter 3 described: it checks whether the generated text *contains* the expected answer, word for word, not whether the two strings are identical."

**ID:** MED-02
**Severity:** MEDIUM
**Confidence:** Medium
**Category:** terminology-consistency / grounding-vs-verification conflation
**File:** `book/code/chapter_09/hybrid_rag_finetune.py:83`; `book/code/chapter_10/traceable_outputs.py:90,107,125`; `book/chapters/chapter_10.qmd:90-98`
**Chapter:** 9, 10
**Current statement/behaviour:** Chapter 9's prompt-builder is named `build_grounded_prompt()` and unconditionally labels any retrieval-augmented prompt "grounded," regardless of whether the model's answer actually uses the retrieved text. Chapter 10 then defines "grounded" as a verified outcome: `result["grounded"] = len(verified_sources) > 0`, and its own Theory callout explicitly distinguishes "grounded in general" from "faithful in this specific answer."
**Finding:** The same word, "grounded," is used with two different operational meanings in adjacent chapters — an unconditional label in Ch9's function name vs. a post-hoc-verified boolean in Ch10's output field. This is exactly the grounding/faithfulness conflation Phase 11 asks reviewers to catch, even though the surrounding prose is otherwise careful about the distinction.
**Evidence:** Direct code inspection by the RAG-focused reviewer; cross-referenced function/field names.
**Why it matters:** A reader following the code rather than only the prose could conclude a "grounded prompt" (Ch9) already means "verified" (Ch10's later, stricter sense).
**Recommended correction:** Rename Chapter 10's boolean field (e.g. `verified` or `faithfully_grounded`), or add one sentence in Ch10 Step 2 noting the term is intentionally reused with a stricter meaning now that verification exists.
**Requires author judgement:** YES
**Status:** ✅ FIXED via the clarifying-sentence option, not a rename (a rename would have rippled into `test_chapter_10.py` and the shipped companion app's two Streamlit pages for a naming-clarity issue, not a correctness bug). `book/chapters/chapter_10.qmd` Step 2 now says: "Note the naming shift: Chapter 9's `build_grounded_prompt` just means 'built from retrieved text,' true whether or not the model actually used it; this step's own `\"grounded\"` field is the stricter claim — true only once at least one source has passed the faithfulness check below."

**ID:** MED-03
**Severity:** MEDIUM
**Confidence:** Medium
**Category:** missing-concept / precision
**File:** `book/chapters/chapter_13.qmd`, Field Notes (~L285-315) and Key Takeaways (~L338-359)
**Chapter:** 13
**Current statement/behaviour:** Continuing training on only a new 17-report batch (no replay of the earlier 57-report data) causes `avg_overlap` to regress; the chapter attributes this to the model's learned template shifting toward the new batch's casing/cementing vocabulary.
**Finding:** This is a textbook instance of catastrophic forgetting / recency bias in continual fine-tuning, but the term "catastrophic forgetting" never appears anywhere in the book (confirmed via full-text grep), and the chapter doesn't mention the standard mitigation — replaying/mixing a sample of older examples with the new batch — even as a documented limitation.
**Evidence:** `grep -i "catastrophic forgetting"` across all chapters/appendices → no matches; Step 2 of Ch13 trains only on `new_examples`, never mixing in `current_examples`.
**Why it matters:** The book documents the actual failure honestly and attributes it correctly to distribution skew, but doesn't connect it to the general, well-known continual-learning concept — a real gap for the "ML correctness" audience, and a missed opportunity for readers who will meet "catastrophic forgetting" elsewhere and not recognize it as the same thing.
**Recommended correction:** Add a sentence in Field Notes or Production Reality naming this as catastrophic forgetting / recency bias, and noting replay of "current" examples alongside new ones as the standard (unattempted) mitigation.
**Requires author judgement:** YES
**Status:** ✅ FIXED. Chapter 13 Step 2's Production Reality now names it directly: "a real instance of what the field calls **catastrophic forgetting** (or, more precisely here, recency bias): training only on new data can pull a model's behavior away from what it learned earlier, even without erasing it outright," and states the standard, unattempted mitigation (replaying a sample of `current_examples` alongside each new batch).

**ID:** MED-04
**Severity:** MEDIUM
**Confidence:** High
**Category:** internal-consistency
**File:** `book/chapters/chapter_13.qmd`, lines 51-54 vs. line 416 (figure caption/alt-text) and `figures/app_screenshot_evaluation_readme.jpg`
**Chapter:** 13
**Current statement/behaviour:** The chapter's own inline experiment reports `Direction: exact_match=unchanged avg_overlap=regressed perplexity=improved` (line 54). The companion-app screenshot caption two pages later calls the app's Ch8→Ch13 checkpoint comparison (which shows perplexity **regressing**, `25.03→25.73`) "the exact... regression this chapter's own version-comparison run found above."
**Finding:** These are two distinct, both-genuine experiments — Ch13's inline toy "current vs. updated" split vs. the persisted Ch8→Ch13 checkpoint chain shown in the real app screenshot — and they disagree on perplexity's direction. Calling the screenshot "the exact" same regression overstates the connection between two different results.
**Evidence:** Direct comparison of the two passages; the screenshot itself was viewed directly and is genuine (not fabricated), it simply documents a different comparison than the one described immediately above it in prose.
**Why it matters:** A careful reader (or reviewer) checking the numbers finds two of the chapter's own regression claims pointing in different directions for perplexity, undermining confidence unnecessarily even though both underlying numbers are individually real.
**Recommended correction:** Reword the caption to "a similar regression, found separately when comparing the book's own persisted Ch8 and Ch13 checkpoints" and note explicitly that this is a different comparison (real checkpoint chain) from the chapter's own smaller illustrative split.
**Requires author judgement:** YES
**Status:** ✅ FIXED, essentially as recommended. The figure caption now reads "a similar 'latest doesn't mean best' regression, found separately by comparing the book's own persisted Chapter 5/8/13 checkpoints (this one regresses on both average overlap and perplexity, not the single metric this chapter's own version-comparison run above flagged — a different checkpoint pair, not the same result)." The `fig-alt` accessibility text, which was already accurate, was left unchanged.

**ID:** MED-05
**Severity:** MEDIUM
**Confidence:** High (found independently by three separate reviewers, converging on the same location and reasoning)
**Category:** terminology-precision
**File:** `book/appendix/appendix_b_glossary.qmd:88`; `book/chapters/chapter_12.qmd` (title and Theory callout, ~L65-72)
**Chapter:** 12
**Current statement/behaviour:** Glossary: "Drift — A model's behavior changing across versions or over time." Chapter 12 ("Detecting Drift Across Model Versions") implements this as a fixed comparison of two already-trained LoRA checkpoints against one static 8-example held-out set.
**Finding:** What's implemented is checkpoint/version regression testing, not the standard MLOps sense of drift (monitoring live, unlabeled production input/output distributions over time for data drift or concept drift). The chapter's own title already partially scopes this correctly ("Across Model Versions"), and nothing here is internally inconsistent or misleading to the target beginner reader — but an ML-ops-literate reader could reasonably expect this chapter to cover live-input monitoring, and the glossary's one-line definition doesn't disambiguate.
**Evidence:** Chapter 12's implementation (`summarize_version`/`compare_versions`) operates only on a fixed, already-collected eval set — no notion of incoming unlabeled traffic anywhere in the code.
**Why it matters:** Minor credibility risk with ML-ops-literate reviewers; low risk of misleading the target beginner audience since the term is self-defined and used consistently.
**Recommended correction:** Add one sentence to the Theory callout or glossary entry distinguishing "version/checkpoint drift" (what's implemented here) from "data drift / concept drift" (live-input monitoring, not implemented in this book).
**Requires author judgement:** YES
**Status:** ✅ FIXED in both locations. Chapter 12's "Engineering Translation: drift / regression" callout now adds: "This chapter's drift check means comparing two already-trained checkpoints against one fixed evaluation set. A production ML team also watches for a different kind of drift — a deployed model's live, unlabeled input traffic gradually looking different from what it was trained on — which is a separate, ongoing monitoring problem this book doesn't build." The glossary's "Drift" entry was extended with the same distinction.

**ID:** MED-06
**Severity:** MEDIUM
**Confidence:** Medium
**Category:** experimental-design / overclaiming from small n
**File:** `book/chapters/chapter_09.qmd`, Key Takeaways
**Chapter:** 9
**Current statement/behaviour:** BM25 retrieves the correct report for 4/4 test queries; dense embeddings retrieve it for 3/4 (missing the held-out report, ranked 8th of 677). The chapter's Key Takeaway states this as a general reason to prefer BM25 for this domain's jargon.
**Finding:** Both numbers are real and independently reproduced exactly, but n=4 queries is a very small sample to generalize a retrieval-method preference from — unlike Chapter 11, which explicitly caveats its own small-n (8 examples) results, Chapter 9's takeaway doesn't add the same explicit small-sample caveat here.
**Evidence:** Cross-reviewer comparison of Ch9's phrasing against Ch11's more careful small-n hedging on a structurally similar result.
**Why it matters:** Phase 6 of this review specifically asks whether conclusions are stronger than the sample size supports; this is a real, if minor, instance where the book is less careful in one chapter than it is in another about the exact same statistical issue.
**Recommended correction:** Add a one-sentence caveat matching Ch11's style, e.g. "with only 4 test queries this isn't a statistically powered comparison, but it's consistent with BM25's known strength on exact-jargon matching."
**Requires author judgement:** YES
**Status:** ✅ FIXED, essentially verbatim as recommended. Chapter 9's Key Takeaways now adds: "Four test queries is a real result, not a toy one, but it's still too small to call this a statistically powered comparison — read it as consistent with BM25's known strength on exact-jargon matching, not as proof dense embeddings lose on this archive in general."

**ID:** MED-07
**Severity:** MEDIUM
**Confidence:** Medium
**Category:** test-coverage
**File:** `book/tests/test_chapter_01.py`
**Chapter:** 1
**Current statement/behaviour:** All 3 of Chapter 1's tests are marked `@pytest.mark.slow`.
**Finding:** This means `load_local_model.py` — the reader's very first hands-on script — never executes in CI on any platform (see HIGH-02). This is the one chapter with zero automated cross-platform safety net, at exactly the point where a beginner is most likely to hit an environment problem.
**Evidence:** `grep -c "@pytest.mark.slow" test_chapter_01.py` = 3 of 3 tests.
**Why it matters:** Environment/first-run issues are the highest-risk failure point for this book's target reader (zero programming background); this is the chapter that most needs a fast smoke test.
**Recommended correction:** Add at least one fast test that doesn't require the real model download (e.g., mocking/stubbing the tokenizer load, or testing the `USE_TF=0`/`USE_FLAX=0` env-var-setting logic in isolation).
**Requires author judgement:** YES (may be intentional, given the file's own documented rationale)
**Status:** ✅ FIXED. Added `test_importing_load_local_model_disables_tf_and_flax_backends` to `test_chapter_01.py` — a real, fast, non-slow regression guard for the exact `USE_TF`/`USE_FLAX` crash CLAUDE.md documents, not a placeholder smoke test. Verified passing in isolation and as part of the full fast suite; test counts (README, book/README) were updated accordingly (81 chapter / 101 total / 74 non-slow).

---

### LOW

**ID:** LOW-01 — **File:** `book/chapters/chapter_04.qmd` (~L430-437) vs. `book/chapters/chapter_05.qmd` (L251)
Ch4 frames Ch5 as proof of "why Chapter 5 has to actually update the model's weights," but Ch5's LoRA config only targets attention projections (`q/k/v/o_proj`), never the embedding layer Ch4 actually measured. The raw input-embedding table Ch4 diagnosed as disconnected (e.g., `BHA` vs. `bottom hole assembly`) is still frozen and numerically unchanged after fine-tuning — the fix happens via how attention layers use those embeddings downstream, a real but different mechanism than what a reader might infer. No sentence is factually false, but there's a mental-model gap between what Ch4 measures and what Ch5 demonstrates. **Recommended correction:** one clarifying sentence noting LoRA here doesn't touch the embedding table itself. **Requires author judgement:** YES. **Status:** ✅ FIXED on both sides — Chapter 4's "WHAT YOU BUILT" box now adds "though, as you'll see there, Chapter 5's fine-tune leaves this exact raw embedding table frozen and untouched; the fix happens at the attention layers downstream of it instead," and Chapter 5's Step 2 now says "including the raw embedding table Chapter 4 measured; `BHA`'s own coordinate doesn't move."

**ID:** LOW-02 — **File:** `book/chapters/chapter_00.qmd:192-198` vs. `appendix/appendix_a_environment_setup.qmd:70-72`, `appendix/appendix_c_known_limitations.qmd:174-177`
Chapter 0's bitsandbytes/QLoRA callout doesn't include the explicit "no chapter in this book actually loads a quantized model" clause that both appendices use for the same caveat — minor inconsistency in how carefully the same caveat is worded across three locations. **Recommended correction:** align Ch0's wording with the appendices. **Requires author judgement:** NO. **Status:** ✅ FIXED — Chapter 0's callout now includes the same clause verbatim: "it doesn't matter for this book either way: no chapter script here actually loads a quantized model."

**ID:** LOW-03 — **File:** `book/chapters/chapter_01.qmd:58` (shorthand callback) vs. line 52 (verbatim quote)
The real DDR text (correctly quoted verbatim two lines earlier) is "lost tool face and became **assembly** became stuck"; the line-58 shorthand tag drops "assembly became," reading cleaner than the source. Used clearly as a memorable tag rather than a second citation, but sits right after the chapter's own "quoted exactly as filed" framing. **Recommended correction:** optional wording tweak ("that phrase, roughly:"). **Requires author judgement:** YES. **Status:** ✅ FIXED — the callback no longer uses quotation marks implying a second verbatim citation; it now reads "Hold that idea — the driller lost tool face and the assembly became stuck — in mind," clearly framed as paraphrase rather than quotation.

**ID:** LOW-04 — **File:** `book/datasets/README.md:6`
The well name appears in three different notations across the repo: `FORGE 16A(78)-32` (dataset README), `FORGE 16A [78]-32` (actual DDR field / training examples), `FORGE-16A-78-32` (filenames). **Recommended correction:** standardize prose on the bracket form, which matches the actual source field. **Requires author judgement:** NO. **Status:** ✅ FIXED — all three prose occurrences of the parenthesis form (`book/datasets/README.md`, `appendix_a_environment_setup.qmd`, `preface.qmd`) were changed to the bracket form matching the real DDR field. Filenames were left unchanged (a distinct, intentional convention).

**ID:** LOW-05 — **File:** `book/references.bib:17,33` (`wei2021flan`, `ji2023hallucination`)
Both entries exist in the bibliography but are never cited with `@key` anywhere in the `.qmd` sources, despite `chapter_01.qmd` (instruction tuning) and `chapter_10.qmd` (hallucination) discussing exactly those topics at length. **Recommended correction:** add `[@wei2021flan]` and `[@ji2023hallucination]` at the relevant points. **Requires author judgement:** NO. **Status:** ✅ FIXED — both are now cited, but in `preface.qmd` rather than the chapters, matching this book's actual established citation convention (all `[@key]` citations live in the preface, never inline in chapter prose): `[@wei2021flan]` where the preface notes the book fine-tunes an already instruction-tuned model, and `[@ji2023hallucination]` where hallucination is named as a risk distinct from data leakage.

**ID:** LOW-06 — **File:** chapter-status strips, `chapter_00.qmd`–`chapter_02.qmd` (en-dash) vs. `chapter_03.qmd`–`chapter_13.qmd` (hyphen)
Estimated-time ranges use an en-dash ("30–45 min") in Chapters 0–2 (matching the root README's table) and a plain hyphen ("20-30 min") in Chapters 3–13. Cosmetic only. **Recommended correction:** standardize on en-dash. **Requires author judgement:** NO. **Status:** ✅ FIXED — all "Estimated time" ranges in Chapters 3–13 now use en-dash, matching Chapters 0–2 and the README.

**ID:** LOW-07 — **File:** `book/chapters/chapter_01.qmd:198-200`
States "Chapter 5 introduces lower precision for faster fine-tuning on a GPU," but Chapter 5's GPU guidance never mentions dtype/precision (only `.to("cuda")`/`.to("mps")`); the only `float16`/`bfloat16`/`dtype` reference anywhere in the codebase is unrelated. A forward-reference that isn't fulfilled. **Recommended correction:** remove the forward-reference or add the missing content to Chapter 5. **Requires author judgement:** YES. **Status:** ✅ FIXED via the lower-scope option — the forward-reference no longer promises lower-precision content; it now accurately says Chapter 5 "shows how to move the same full-precision model onto a GPU (`model.to(\"cuda\")`) if you have one, without changing this precision."

**ID:** LOW-08 — **File:** `CLAUDE.md` ("Companion projects" section)
States `book/app/` is "not yet implemented," but the Streamlit companion app is in fact fully built (V1–V3, "frozen" per its own README) and is referenced as a real, working artifact by Chapter 13 (its Failure Analysis page reruns the book's own documented failure cases). This is a stale project-documentation note, not a book-content defect, but CLAUDE.md itself warns its own status notes "have gone stale before." **Recommended correction:** update CLAUDE.md's companion-projects note. **Requires author judgement:** NO. **Status:** ✅ FIXED — `CLAUDE.md` now states "V1-V3 are frozen and working" and names the three real pages, with a note to re-check `book/app/README.md` before assuming any specific page's status.

**ID:** INFO-01 — American English spelling (`color`, `license`, `optimize`, `behavior`, etc.) is used consistently throughout the entire book with zero UK-spelling instances found. This is internally 100% consistent; per this review's own instruction not to rewrite a consistent style choice, **no correction is recommended** — noted only so the review record shows this was checked.

---

## No-Leakage / Clean-Audit Confirmations

- **Data leakage (Phase 15): NO EVIDENCE OF LEAKAGE.** Held-out report `#37` was searched for by exact string (report number, date, distinctive phrases) across every training JSONL — zero matches. The `HELD_OUT_REPORT` constant is defined once (`code/chapter_02/build_training_examples.py`) and imported (never re-implemented) in Chapters 7, 8, 9, and 13, structurally preventing drift between chapters. Seven independent automated tests assert the exclusion at multiple pipeline stages. The fine-tuned model's own behavior is indirect corroborating evidence: it answers the held-out report's questions with a *wrong*, memorized answer borrowed from reports #36/#38 — the expected signature of genuine non-exposure, not of leakage. Near-duplicate/semantic leakage is explicitly disclosed by the book itself (Chapter 6, Production Reality) as an unchecked limitation rather than a silently hidden gap.
- **LoRA/PEFT implementation:** every checked hyperparameter, target-module list, and trainable-parameter count (including the challenge exercise's alternate configuration) was independently hand-derived from Qwen2.5-1.5B-Instruct's real architecture and matched the book exactly. No book-code mismatches found in this area.
- **Fine-tuning vs. knowledge/capability overclaiming:** no instance found anywhere in the book of fine-tuning being described as teaching facts, guaranteeing domain competence, preventing hallucination, or replacing retrieval — the book explicitly asserts the opposite multiple times (Chapters 1, 3, and the Chapter 9 transition).
- **Chapter 9→10 progression** (retrieval succeeds while generation ignores the retrieved text; Chapter 9 explicitly says it "has no way to catch it when they do"; Chapter 10 then builds exactly that missing check) is the cleanest real demonstration in the book of the retrieval/grounding/faithfulness distinction this review's Phase 10–11 asks reviewers to look for.
- **Faithfulness checker** (Chapter 10) is honestly scoped in both code and prose as lexical word-overlap, not semantic entailment, backed by concrete counter-examples (the wrong-source report #21/#27 case, and a same-source-flipped-direction case still scoring 0.80) — the "this is a teaching tool, not a safety check" framing is earned, not just a disclaimer.

---

## Summary Scorecard

The table below shows both the score **at time of review** and the score **after remediation** (all findings fixed, re-verified 2026-09-15). Pre-fix scores are preserved so the record shows what was actually found, not just the end state.

| Category | Score (at review) | Score (post-fix) | Justification |
|---|---|---|---|
| Fine-tuning / ML correctness | 9/10 | 10/10 | Explanations are technically accurate and appropriately hedged for a zero-programming audience; memorization-vs-generalization treatment is honest throughout (0/2, 0/50 preserved, not smoothed over). LOW-01's embedding-vs-attention-weight mental-model gap is now closed on both the Ch4 and Ch5 side. |
| LoRA / PEFT correctness | 9/10 | 10/10 | Every checked hyperparameter and trainable-parameter count independently reproduces exactly against the real model architecture; adapter save/load, masking, and frozen/trainable distinctions all match code. LOW-01/LOW-02's minor caveats are now resolved. |
| Experimental validity | 7/10 | 9/10 | Baseline/held-out separation is real and test-enforced; a flagship result (Ch5) reproduces bit-for-bit with a fixed seed. MED-04 (Ch13's internal inconsistency) and MED-06 (unhedged small-n conclusion) are both fixed; not a 10 only because these were prose fixes to an existing design, not new controlled re-experiments. |
| Evaluation methodology | 7/10 | 9/10 | Multiple complementary metrics used correctly in most places (exact match + overlap + perplexity), with real small-n caveats in Ch11. MED-01 ("strict exact-match" mislabeling) fixed at both definitional points (README, Ch11 Step 2). |
| RAG / retrieval correctness | 8/10 | 9/10 | Correct, standard, empirically-justified RAG pattern (BM25 chosen over dense embeddings based on a real, reproduced comparison, not assumption); retrieval/grounding/faithfulness are mostly kept conceptually distinct. MED-02 ("grounded" naming) and MED-06 (small-n caveat) both fixed. |
| Drilling & completions correctness | 8/10 | 9/10 | Engineering terminology (BHA, POOH/TOOH, packers, fishing, casing/cementing, NPT) is uniformly correct; DDR-derived claims across five chapters and seven cross-checked source PDFs matched exactly, including faithful preservation of the source's own internal inconsistencies. CRIT-01's fabricated example is fixed with a genuine, verified replacement excerpt. |
| Code-to-book consistency | 6/10 | 9/10 | Verified `.qmd` code blocks match their `.py` scripts verbatim almost everywhere checked, and printed transcripts match real script output. All three drivers of the original low score — CRIT-01 (fabricated example), HIGH-02 (CI-coverage overclaim), and MED-01 ("strict" mislabeling) — are fixed; not a 10 only because this reflects prose/example corrections re-verified by the same reviewer rather than a full independent re-audit. |
| Data provenance / leakage control | 9/10 | 9/10 | Unchanged — no leakage-control finding required a fix. Single source of truth for the held-out constant, imported everywhere it's needed, verified by 7 passing automated guard tests plus independent grep and behavioral corroboration. Not a 10 only because near-duplicate/semantic leakage is disclosed as unchecked rather than actually checked. |
| Reproducibility | 7/10 | 9/10 | Cross-platform instructions, the `USE_TF=0`/`USE_FLAX=0` import-order handling, and dependency/version guidance are genuinely careful and correct. HIGH-02 (CI-scope overclaim) fixed, test counts corrected, and Chapter 1 now has real fast CI coverage via the new regression test (MED-07). |
| Beginner accessibility | 9/10 | 9/10 | Unchanged — no accessibility finding required a fix. Sampled chapters consistently introduce every new concept via a physical/operational analogy before its technical name, matching CLAUDE.md's own stated bar. |
| Editorial quality | 8/10 | 9/10 | Prose is unusually clean for a repo this size; heading hierarchy, image references, and cross-references (`@sec-*`) are all consistent and unbroken. Stale test count, the unfulfilled Ch1→Ch5 forward-reference, dash-style inconsistency, and well-name notation drift are all fixed. |

---

## Top 10 Issues to Fix Before Publication (correctness/credibility, not grammar)

All 10 are now ✅ FIXED — see each finding's **Status** line above for the specific change.

1. ✅ **CRIT-01** — Chapter 4's fabricated example falsely attributed to a real, named report (`Drilling_038`).
2. ✅ **HIGH-01** — Root README's "a local model sidesteps the [data-governance] question entirely" overclaim.
3. ✅ **HIGH-02** — README/book-README claims of full-suite CI coverage on 3 OSes when CI only runs the fast subset (73/100 tests); Chapter 1 has zero CI coverage as a direct consequence.
4. ✅ **MED-01** — "Strict exact-match" is actually a lenient, case-insensitive substring-containment check; the label should match the behavior.
5. ✅ **MED-04** — Chapter 13's two regression claims (inline experiment vs. app screenshot) contradict each other on perplexity's direction while being described as "the exact" same result.
6. ✅ **MED-03** — Chapter 13's continual-fine-tuning regression is a real instance of catastrophic forgetting but is never named or connected to the standard concept/mitigation.
7. ✅ **MED-02** — "Grounded" is used with two different meanings (unconditional label in Ch9, verified boolean in Ch10).
8. ✅ **MED-05** — "Drift" (Ch12) is scoped to checkpoint/version comparison, narrower than the standard MLOps meaning, without disclaiming the difference.
9. ✅ **MED-06** — Chapter 9's BM25-vs-dense-embeddings takeaway draws a general conclusion from 4 test queries without the same small-n caveat Chapter 11 applies to a structurally similar result.
10. ✅ **MED-07** — Chapter 1 (the reader's first hands-on script) has no fast/CI-covered tests at all.

## Top 10 Minor Editorial Corrections

All are now ✅ FIXED — see each finding's **Status** line above.

1. ✅ **LOW-06 / stale count** — "74 automated tests" appears three times (`README.md:18,557`; `book/README.md:212`); actual count is now 81 (chapters) / 101 (including the app tests) after MED-07 added one new test.
2. ✅ **LOW-05** — Two bibliography entries (`wei2021flan`, `ji2023hallucination`) exist but are never cited in the text.
3. ✅ **LOW-06** — En-dash vs. plain-hyphen inconsistency in chapter-status time ranges (Ch0–2 vs. Ch3–13).
4. ✅ **LOW-04** — Well-name notation inconsistency: `FORGE 16A(78)-32` vs. `FORGE 16A [78]-32` vs. `FORGE-16A-78-32`.
5. ✅ **LOW-03** — Minor paraphrase drift in Chapter 1's callback quote ("lost tool face and became stuck" vs. the source's "...became assembly became stuck").
6. ✅ **LOW-07** — Chapter 1's forward-reference to Chapter 5 covering "lower precision" GPU training, which Chapter 5 doesn't actually cover.
7. ✅ **LOW-02** — Inconsistent QLoRA/bitsandbytes caveat wording between Chapter 0 and the two appendices that state the same caveat more completely.
8. ✅ **LOW-08** — `CLAUDE.md`'s "companion app not yet implemented" note is stale; the app is fully built.
9. ✅ **LOW-01** — Chapter 4→5 embedding-vs-attention-weights nuance could use one clarifying sentence.
10. *(No further genuine editorial defects surfaced in the sampled chapters — heading hierarchy, cross-references, and image links were all checked and found clean; padding this list further would manufacture issues rather than report them.)*

---

## What Should NOT Be Changed

- Chapter 5's LoRA explanation, diagram, and "two much shorter lists of numbers" analogy — technically correct, beginner-accessible, and matches the actual `LoraConfig`/`get_peft_model` code exactly.
- The memorization-vs-generalization treatment across Chapters 5, 8, and 11 — consistently honest, with negative results (0/2 held-out never improving, 0/50 at scale) preserved rather than smoothed over, exactly matching this book's stated editorial philosophy.
- Loss vs. exact-match vs. perplexity reasoning in Chapters 8/11 — a genuinely sophisticated point (a flat exact-match score can hide real progress that perplexity catches) argued correctly in both directions.
- Prompt-masking (`-100` labels) code and its accompanying prose explanation match exactly; the "batches with padding" simplification is flagged as a simplification, not hidden.
- The Chapter 8 checkpoint-resume caveat (fresh optimizer state, no restored Adam momentum) is stated as a plain, honest limitation.
- Chapter 9's BM25-vs-embeddings empirical comparison itself (separate from the small-n caveat issue above) is a genuinely rigorous, falsifiable claim, independently reproduced exactly in this review.
- Chapter 10's faithfulness checker scoping ("teaching tool, not a safety check") is earned by concrete counter-examples, not just asserted.
- Chapter 12's `compare_versions` design, which deliberately reports direction per-metric instead of collapsing to a single pass/fail — sound experimental-design reasoning, explicitly justified in-chapter.
- The held-out-report leakage-prevention architecture (single constant, imported everywhere, multiply tested) — a genuinely well-engineered safeguard, not just a documented promise.
- Beginner-accessibility voice throughout the sampled chapters — physical/operational analogies consistently precede technical terms, matching the bar CLAUDE.md itself sets and holds Chapter 5's LoRA callout to.
- Drilling/completions engineering terminology and the vast majority of DDR source-traceability (outside the one CRIT-01 instance) — accurate, and notably honest about preserving the source archive's own internal quirks (e.g., Report #38's two slightly different ROP figures, both kept rather than silently reconciled).

---

## Final Publication Assessment

**At time of review: READY AFTER TECHNICAL CORRECTIONS. After remediation (current state): READY.**

The book's core technical claims held up well under direct, independent reproduction even before any fixes were applied: this review re-ran Chapter 5's full fine-tuning pipeline and got digit-for-digit agreement, hand-verified its LoRA parameter math against the real model architecture, reproduced Chapter 9's retrieval comparison exactly, and found no evidence of held-out-data leakage after a dedicated search. The LoRA/PEFT implementation, the memorization-vs-generalization narrative, and the RAG/faithfulness pedagogy were all substantively sound and, in several places, unusually rigorous and self-critical for an educational text — none of that required any correction.

The one CRITICAL and two HIGH findings that did exist were real, checkable defects: a Chapter 4 example was fabricated and falsely attributed to a real, named source report (CRIT-01) — a direct violation of this project's own stated non-negotiable rule; the root README overclaimed that a local model "sidesteps" data-governance obligations "entirely" (HIGH-01); and the README/CI documentation overstated how much of the test suite actually ran continuously across platforms (HIGH-02). All three, along with every MEDIUM and LOW finding, have since been corrected — see each finding's **Status** line above for specifics — and the full fast test suite (74 tests, including one new regression test added as part of the fix) passes cleanly after every change. No chapter was restructured and no experimental conclusion was retracted or reversed in the process; every fix was a bounded, well-scoped correction consistent with the finding's own recommendation.

The repository, as it stands now, reflects a book whose numerical claims are independently reproducible, whose held-out data genuinely isn't leaked, whose LoRA/PEFT and RAG implementations match their own documentation exactly, and whose remaining terminology has been tightened for precision without softening any of its honest negative results (the `0/2` and `0/50` failures, the faithfulness checker's real false-positive case, the Chapter 13 regression) — all of which remain in the book exactly as documented, per this review's own instruction to preserve genuine negative findings.
