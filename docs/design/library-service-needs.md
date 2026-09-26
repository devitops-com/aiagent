# aiagent's requirements for the devai library service API

**Status:** agreed with devai's plan, pending the owner's review. Written on 2026-09-25 from the
owner's "digital library" decisions, as relayed by the devai session, and revised the same day
after an adversarial review (§12 lists the review points not taken). Revised on 2026-09-26 with the
owner's source decisions L12-L16 (§0.1) and devai's answers to O-D1..O-D13 (§11.2), both relayed by
devai. Revised again on 2026-09-26 with devai's further answers and requested adjustments (dated
notes in §0.1 and §11.2): HTML-primary text in v1, rate groups, `openalex:` keys, source stops,
English-only permanent records, versions and licences from arXivRaw, tombstones and enrichments in
v1, `egress_blocked`, and a revised acceptance test. Revised a third time on 2026-09-26 with
devai's last answers (§0.1 notes): the stop causes and how a partial stop shows, 429s as waits and
the two breaker kinds, the robots.txt reason codes, retries of the HTML request, the agreed field
names, licences of non-arXiv works, the HTML extras (`anchor.html_id`, `anchor.env`, `refs.cites`,
`refs.labels`) in v1, and `GET /v1/verdicts/{id}` and `GET /v1/library/topics` in v1. Revised a
fourth time on 2026-09-26 with the final agreement with devai (§0.1 notes): check 2, waits outside
the retry budget, the cause always at `error.details.cause`, the agreed shapes, `limited` sources,
what counts as "no HTML", `egress_blocked` and `egress_unavailable`, `ssrn:` fetch keys, the
HTML-only fields, "allowed host", and the later excerpt rule after a repo refresh; and with the
owner's answers to devai's own questions (D24-D28, relayed by devai), of which D28 answers O15.
The owner then added L5's "as fetched" clause for an arXiv HTML fetch error (§0.1).
Nothing is implemented. devai's plan (`docs/plans/library-service.md` in the devai repo) adopts
this document as its field-level contract, and matches it as revised.
**Scope:** what aiagent needs from devai's library service: identifiers, search, fetching and
staging, translation, verdicts, the research profile, library queries, jobs and progress, and
transport. How devai builds the service is devai's business (the store layout, the crawler, the
index engines, credentials), except where aiagent depends on something it can observe.
**Sources:**
- the owner's decisions (§0.1), including the source decisions made in the devai session after
  checking the sources' terms (L12-L16);
- devai's answers to O-D1..O-D13 (§11.2), its further answers and adjustments of 2026-09-26, and
  the owner's answers to devai's own questions (D24-D28), which its plan
  `docs/plans/library-service.md` records;
- aiagent at `1b4f161`: `ingest/`, `core/segment.py`, `distill/`, `system1/`, `llm/`, `skills/`,
  `cli/`;
- [laya-system1-distillation.md](laya-system1-distillation.md) §7-9 and §13;
- devai's [`docs/laya-trainer.md`](https://github.com/ksparavec/devai/blob/main/docs/laya-trainer.md).
  This document reuses its conventions on purpose: OpenAI-style objects and errors, a volume that
  is the source of truth, `<prefix>-<hex>` ids, and `inbox/` submissions named by their manifest
  hash;
- the review of this document (2026-09-25), which measured laya token costs with the real
  `laya-multilingual` tokenizer through aiagent's `SequenceTokenizer`.

**Wording.**
- **must**: aiagent cannot do the job without it.
- **should**: aiagent can work around its absence, at a cost this document names.
- **v1**: needed for the first acceptance test (next section). **later**: everything else. Later
  fields are specified now so that v1 does not rule them out; they stay nullable until then.
- All ids, names and numbers in the examples are illustrative, not real records.

---

## Minimal first version (for the acceptance test)

The acceptance test (§10) needs only what is listed here. Everything else in this document is
later.

- **Sources:** arXiv and OpenAlex, not arXiv only (L15, which answers O13).
  - **arXiv:** API search (`export.arxiv.org`). Then, in the fetch job: the PDF and, when arXiv has
    it, the arXiv HTML of the latest version (`arxiv.org`), and that version's `arXivRaw` record
    (`oaipmh.arxiv.org`) for the full version list, the withdrawal flag and the licence (§3.1).
    Never the e-print (L12).
  - **OpenAlex:** discovery beyond arXiv; open-access locations (on an allowed host a fetchable
    link record, otherwise a "known, not held" link record); `cited_by` counts; author h-index
    (advisory, with provenance); `openalex:W<id>` keys for works with neither an arXiv id nor a DOI
    (§1.1). **Unpaywall**, by DOI, adds an open-access location and the licence.
  - SSRN and the other restricted sources are link only (L14): an `ssrn:` fetch key is accepted,
    and its item ends `not_fetchable` with `terms_disallow_automation` (§3.1). Semantic Scholar
    comes later.
- **Text: HTML first** (L12, adjusted 2026-09-26).
  - When arXiv has the HTML, its LaTeXML markup is the primary text (`origin: html`,
    `role: primary`): sections, paragraphs numbered from LaTeXML's ids (each unit carries its own
    id as `anchor.html_id`), display equations as TeX with their printed numbers, inline math as
    `$TeX$` with `math_spans`, theorem-like environments (`anchor.env`), the bibliography entries
    each unit cites with their parsed identifiers (`refs.cites`), and cross-reference targets as
    LaTeXML ids (`refs.labels`). Each unit's PDF page comes from aligning its text with the PDF's
    text layer (`page_method: text_align`), and the PDF's text is staged as a second text
    (`origin: pdf`, `role: fallback`).
  - A definite "no HTML" (a 404 or 410, or a 200 without LaTeXML's generator marker) stages the
    item from its PDF alone at once: not a failure, with `quality.warnings` saying why. A 429, a
    5xx, a network failure, or a 200 with the marker but the wrong content type or version, is an
    error, retried within the item's retry budget like the PDF; once the budget is used up, the
    item is staged from its PDF alone, with the warning. Its HTML is fetched later only on request
    (§1.5, §3.1, §5.5).
  - Without HTML, the PDF is the primary text, structured by GROBID (sections, paragraphs, formulas
    as text, references with DOI and arXiv ids, page coordinates), with pypdfium2's text layer and
    pages as the fallback (L7: pages alone are acceptable). GROBID runs only for PDFs without HTML.
  - Units are capped at 1,300 characters; the service counts no tokens.
- **Judging:** by aiagent's LLM alone; students are optional (O14). No translation: the topic is
  English, and an accept of a non-English text is refused (412). Permanent records hold English
  only (L6): a verdict on a non-English item carries `metadata_en`, and `quote_en` in its evidence
  (§5.1).
- **Endpoints and the fields v1 needs:**

  | endpoint | v1 fields |
  |---|---|
  | `GET /health`, `GET /v1/info` | `api_version`; `/health`: `status`, `contact.configured`, `sources.{arxiv, openalex} {status, cause}`; `/v1/info`: `sources.{arxiv, openalex} {status, cause}`, `per_group {hosts, min_interval_s, max_connections}` (§9.1), `segmenting.unit_max_chars`, `limits` |
  | `PUT /v1/profiles/{name}/versions/{version}`, `GET …/versions[/{version\|latest}]` | as §6 |
  | `POST /v1/searches` (202 + job; 503 `unavailable` with `error.details.cause` when every named source is stopped, §9.1), `GET /v1/jobs/{id}/results?cursor=` | request: `query_id`, `topic_id`, `query {text, native.{arxiv, openalex}}`, `sources`, `filters {date_from, date_to, categories}`, `max_results`, `cursor`. Results: candidate records, plus per source `{upstream_total, returned, next_cursor, error}`, where `error.details.cause` names the stop for a stopped source |
  | `POST /v1/enrichments` (202 + job; devai Phase 2), `GET /v1/candidates/{item_id}` | `item_ids`, `enrich: ["author_metrics", "citations"]` (§2.3); results: candidate records |
  | `POST /v1/fetches` (202 + job; 503 as for searches) | `items [{item_id \| key \| link_id, want, version: "latest"}]`, `want` ⊆ `["pdf", "html"]`; per-item outcome (§3.1), with the primary text and `fallback_text` |
  | `GET /v1/jobs[/{id}]`, `GET /v1/jobs/{id}/events?after=`, `POST /v1/jobs/{id}/cancel` | `status`, `progress`, `waits`, `eta {at, seconds, basis, reason}`, `eta_inputs.per_group {requests_left, min_interval_s, next_slot_at}`, `upstream {requests, bytes}`; a heartbeat at least every 5 s |
  | `GET /v1/items/{id}`, `GET /v1/items/resolve?key=` | §7.4, with the tombstones of purged texts |
  | `GET /v1/items/{id}/texts/{text_id}/units?from=&to=` | units (§3.3); 410 `text_purged` with the tombstone for a purged text (§10 check 3) |
  | `POST /v1/verdicts`, `POST /v1/verdicts/batch` (inline, up to 500, one result per row), `GET /v1/verdicts/{id}` | below; the GET returns the stored record (§5.1) |
  | `POST /v1/items/{id}/topics` | as a verdict, with `topic_id` and `relevant` |
  | `GET /v1/library/topics`, `GET /v1/library/topics/{topic_id}/items` | the topic ids with counts; the listing (§7.3) |
  | optional: `POST /v1/library/search` | hybrid, fused score only (step 14) |

- **Record fields in v1:**
  - **Candidate:** `item_id`, `canonical_key` (namespaces `arxiv`, `doi`, `ssrn`, `openalex`),
    `kind`, `ids {arxiv, doi, openalex}`, `sources_seen`, `title`,
    `authors [{position, name, ids.openalex, metrics}]`, `abstract`, `language`, `categories`
    (arXiv scheme, for arXiv records), `dates`, `versions {latest, list}` (arXiv; `list` is null
    until a fetch has read `arXivRaw`), `journal_ref`, `comments`,
    `license {status, spdx, url, stated_by}` (`status` ∈ `stated`, `none_stated`; null until a
    fetch job has read it: for arXiv from `arXivRaw`, for another work from the open-access
    location its copy came from),
    `access {free, fetchable, formats, est_bytes, robots_allowed, reason}`, `links`,
    `links_found` (at least the open-access locations), `citations`, `library`, `retrieved_at`.
    `ids.openalex` and `citations` are filled where OpenAlex provides them (a search merge or an
    enrichment), `authors[].metrics` by an enrichment (§2.3); all are null otherwise.
  - **Staged text:** `text.json` as in §3.2 (`origin` `html` or `pdf`, `role` `primary` or
    `fallback`, `quality.warnings`); unit = `unit_id`, `kind`, `text`, `lang`,
    `anchor {section_path, paragraph, block, html_id, part, env, pdf_page, pdf_page_end,
    page_method, equation_number, equation_number_method}` (`section_path` and `paragraph` from
    LaTeXML's ids for HTML, and from GROBID for a PDF; `block` for PDF text only),
    `counts.chars`, `math_spans` (offsets include the `$` delimiters; empty for PDF text),
    `refs {cites, labels, links}`, `prev`, `next`. Exactly four fields are HTML-only, and null for
    PDF texts: `anchor.html_id`, `anchor.env`, `refs.cites` and `refs.labels` (agreed with devai);
    a PDF text structured by GROBID keeps its section and paragraph anchors.
  - **Job results:** per fetch item, the primary text and `fallback_text` (§8.1); in
    `item.staged`, `fallback_text_id` (§8.2). Staging-TTL purges run as jobs of kind
    `maintenance` (§3.8).
  - **Verdict:** `verdict_id`, `item_id`, `version`, `text_id`, `stage`, `decision`, `basis`,
    `reasons [{code, text, evidence}]` (`quote_en` in the evidence of a non-English item),
    `scores`, `topic`, `profile`, `deciders`, `duplicate_of`, `supersedes`, `override`,
    `metadata_en` (a non-English item), `client`, `decided_at`. The service adds `recorded_at` and
    the evidence match offsets.
- **Files on the volume** (a read-only mount; v1 needs no `inbox/`): the staged
  `text.json` + `units.jsonl` + `SHA256SUMS`, `jobs/<id>/{job.json, events.jsonl, results.jsonl}`,
  and the record files.
- **Rules that hold from day one:**
  - the plain files are the only source of truth, and every index is derived (principle 9);
  - permanent records hold English only; only staging, which is temporary, may hold the original
    language (L6, §5.1);
  - a rejected item keeps its verdict only: its texts are purged to tombstones, which
    `GET /v1/items/{id}` lists and the units endpoint answers with 410, and its evidence quotes live
    in the verdict (§1.3, §5.3); staging-TTL purges run as a maintenance job (§3.8);
  - the duplicate checks: DOI-to-arXiv mapping, a format check before hashing, and
    `duplicate_of_stored` / `duplicate_of_rejected` (§1.5);
  - a retry budget per item, against which only failed attempts count, never wait time (§3.1),
    and the licence recorded on accept, copied from what the fetch job read (`arXivRaw` for
    arXiv, the open-access location for another work), "none stated" included (§5.3);
  - the source rules: no e-print is ever fetched (L12); API terms govern API calls and robots.txt
    governs page and file fetches, with the reason codes `robots_disallowed`, `robots_denied` and
    `robots_unreachable` (L13, §2.2); link-only sources are never fetched (L14); no external fetch
    before the contact address is configured (L16); a single upstream 429 or 503 is a wait, never
    a breaker event (§9.1); a stopped source fails fast, with the cause at `error.details.cause`,
    and shows as `down` when all its traffic is stopped (`contact_not_configured`,
    `breaker_open`) or `limited` when part of it is (`arxiv_fulltext_cap`), while an overload
    pause or a used-up daily budget is a wait, not a stop (§9.1); an answer of the service's own
    egress proxy is never the source's answer: a DLP block is `egress_blocked`, and a failure of
    the proxy itself `egress_unavailable` (§3.1).

---

## 0. Scope and principles

### 0.1 Owner decisions (2026-09-25/26, relayed)

| # | Decision |
|---|---|
| L1 | **The split.** aiagent does everything that thinks: search planning, queries, classification (laya System-1 students), the multi-round encoder↔LLM loop, translation, verdicts, answers with citations, and the versioned research profile. devai runs a **library service** with a small API. It does all external network I/O (search, download, metadata), so credentials never enter the lab. It also owns robots.txt and per-site rate limits, the permanent store, the index, credentials (sops/age) and model serving. For now the only consumer is aiagent (an MCP server may come later). |
| L2 | **Research profile.** The outer bound is CS/AI, computational finance, analysis of PDEs, numerical analysis and probability. Each query is an inner bound. The profile is versioned. |
| L3 | **Sources.** Priority 1 is arXiv. Next come free independent (non-corporate, non-government) sources: SSRN (link only, L14), author pages, open-access journals. Next, GitHub repos linked from papers. Last, selected blogs, forums and Wikipedia. Kaggle, Colab, Reddit, Stack Exchange, Substack and Medium come later, not now. All but Medium are restricted: official APIs or links only (L14). Medium is not on the restricted list; its status is undecided until its terms are read (note below). Free sources only. A "well-known scientist" is defined by a metric, e.g. the h-index. |
| L4 | **Code.** Never clone a repo. Keep only short relevant excerpts, each with a permanent link to the exact commit (repo, file, lines) and the repo licence. No datasets. Verification level (a), metadata and content checks, applies now. Level (b), sandboxed execution as a devai job runner, comes later. |
| L5 | **Storage.** An accepted item keeps its full content (for arXiv: the PDF plus the arXiv HTML when arXiv has it, as fetched: an HTML error that outlasts its retries leaves the PDF and a `quality.warnings` reason (owner, 2026-09-26, note below); latest version only; as amended by L12). A rejected item keeps only a verdict record and no artifacts; it is never downloaded or judged again unless someone asks. Every verdict records the profile version and the model or student that decided. Nothing is re-judged automatically. Accepted items are never silently deleted. |
| L6 | **English only is stored.** A non-English item is machine-translated first, marked as translated with the model named. The original is kept as a link, and citations point to the original. Permanent records hold English only (note below). |
| L7 | **Citations** give the section, paragraph and equation, plus the PDF page. Pages alone are acceptable. (As amended by L12: the section, paragraph and equation anchors come from the LaTeXML markup of arXiv HTML, and for a PDF from GROBID.) |
| L8 | **Storage layout.** One configurable directory of plain files in a git-object-like hierarchy, fast with standard Linux tools, planned for millions of files. Search indexes are derived from it and can be rebuilt. Search is keyword plus vector, with an English embedding model on CPU. |
| L9 | **Acquisition runs as background jobs.** aiagent shows progress and an ETA; a prompt that stays silent for minutes is unacceptable. The service's job API exposes progress and rate-limit waits so an ETA can be computed. |
| L10 | **First acceptance test:** the topic "Automatic Adjoint Differentiation method in computational finance" (§10). |
| L11 | **No arbitrary URLs** (devai constraint, agreed 2026-09-25). The service accepts only structured requests: queries against vetted sources, and ids the service itself issued (item, version, link). It never takes a URL from aiagent, so the agent cannot use it to send data out. Links found in fetched metadata or text come back as **link records** with a service-issued `link_id`; the service alone decides whether a link is fetchable: only a link on an allowed host is. The service's outbound traffic goes through pipelock; pipelock's own answers are never the source's answer (note below). |
| L12 | **arXiv: the PDF and the arXiv HTML, never the e-print** (2026-09-25/26, relayed by devai). arxiv.org/robots.txt disallows `/e-print` and `/src` and allows `/pdf`, `/html` and `/abs`, with a Crawl-delay of 15. A fetch asks for `want: ["pdf", "html"]`, and the artifact kinds are `pdf` and `html`. The arXiv HTML is LaTeXML output, with broad coverage. It gives formulas as MathML with their TeX, and section numbers, equation numbers and LaTeXML's ids: LaTeX-grade anchors without compiling. Texts of `origin: html` replace the `origin: latex` planned earlier, and are the primary text in v1 (note below). For PDFs without HTML, the structure comes from **GROBID** (Apache-2.0, a CPU sidecar): sections, paragraphs, formulas as text, references with DOI and arXiv ids, and page coordinates; pypdfium2 is the fallback. |
| L13 | **robots.txt and APIs** (2026-09-25/26, relayed by devai). A documented API follows its API terms; robots.txt governs page and file fetches. (export.arxiv.org's robots.txt says `Disallow: /`, but arXiv's API terms sanction `/api/query`; Wikipedia's API is the same case.) |
| L14 | **Restricted sources: official APIs or link only** (2026-09-25/26, relayed by devai). SSRN is **link only**: its terms forbid automated queries, and its PDFs sit behind Cloudflare. An SSRN work is a candidate whose access is not fetchable, with reason `terms_disallow_automation` and its landing link. The service first tries OpenAlex and Unpaywall for another open location of the same work. The same link-only rule holds for Substack, Kaggle HTML, Stack Exchange, Reddit and Drive-hosted Colab notebooks. |
| L15 | **Not arXiv only** (2026-09-25/26, relayed by devai). For the AAD test the owner said "try to provide sources if feasible, do not limit them to arXiv only" (this answers O13). devai's first version includes **OpenAlex**: discovery beyond arXiv; open-access locations (on an allowed host a fetchable link record, otherwise a "known, not held" link record); `cited_by` counts; author h-index (advisory, with provenance). OpenAlex's budget: $0.10/day keyless, $1/day with a free key. **Unpaywall**, by DOI, gives an open-access location and the licence. Semantic Scholar comes later. Keys and enrichments: note below. |
| L16 | **Contact identity** (2026-09-25/26, relayed by devai). The owner creates a dedicated contact address. The service identifies itself as `User-Agent: devai-library/<ver> (+mailto:<address>)` and refuses external fetches until that address is configured. How that, and every other stop, shows: note below. |

**devai's answers and adjustments (2026-09-26, from devai).** They amend the rows named; devai's
plan records them.

- **L12, arXiv HTML (verified by devai).** Every formula carries its TeX
  (`annotation encoding="application/x-tex"`: 78 of 78 on `2609.27602`). The authors' `\label`
  keys are **not** in the page, only LaTeXML's ids (`S2.E1`, `S1.p1`) and the printed numbers
  (`"(2.1)"`). For an HTML text, `equation_labels` therefore stays empty, `equation_number_method`
  is `html`, `anchor.paragraph` comes from the LaTeXML id, which the unit carries as
  `anchor.html_id` (`S3.SS2.p2`), and `anchor.block` is null (§3.3). Coverage is broad, even for
  1999 papers (`hep-th/9901001`). Search cannot cheaply tell whether a paper has HTML; the fetch
  reports it. An HTML answer counts only if it is 200, `text/html`, carries LaTeXML's generator
  marker and names the resolved version `vN`. A definite "no HTML" is a 404 or 410, or a 200
  without the marker: the item is staged from its PDF alone at once, which is not a failure, and
  `quality.warnings` says why. A 429, a 5xx, a network failure, or a 200 with the marker but the
  wrong content type or version, is an error, retried within the item's retry budget like the
  PDF; when the budget is used up, the item is staged from its PDF alone, with the warning (final
  agreement; §1.5, §3.1). HTML for such an item, and HTML that arXiv backfills for an item
  accepted with its PDF alone, is fetched only on request (§5.5).
- **L12, contract adjustment: HTML-primary text in v1.** This replaces "the arXiv HTML is stored on
  accept but not converted". When arXiv has the HTML, it is the primary text (`origin: html`,
  `role: primary`) and the PDF's text is the fallback (`origin: pdf`, `role: fallback`); pages come
  from `page_method: text_align`; `math_spans` and printed equation numbers are v1. GROBID runs
  only for PDFs without HTML (§3.2, §3.3, §3.5).
- **L12, HTML extras in v1.** For HTML texts, v1 also gives `anchor.html_id` (the unit's own
  LaTeXML id, such as `S3.SS2.p2`), `anchor.env` (theorem-like environments), `refs.cites` (the
  bibliography entries the unit cites, with the identifiers parsed from them) and `refs.labels`
  (the targets of its cross-references, as LaTeXML ids, resolved through `anchor.html_id`). These
  four are exactly the HTML-only fields, null for PDF texts; a PDF text structured by GROBID keeps
  its section and paragraph anchors (final agreement; §3.3, §3.5). Their shapes are agreed
  (below).
- **L12, versions and licences.** A search candidate carries the latest version and the dates of
  v1 and of the latest version only (the limit of the API's Atom feed). The full version list, the
  withdrawal flag (best effort) and the licence come from `arXivRaw`, which the fetch job reads, so
  an accept makes no external request. The licence may be "none stated" (papers before 2007), and
  legacy licence URIs are accepted (§2.2, §3.1, §5.3).
- **L15, licences of non-arXiv works.** The fetch job records the licence from the open-access
  location the copy came from (OpenAlex's `locations[].license`, Unpaywall's
  `best_oa_location.license`), with `stated_by` naming that source (`stated_by` ∈ `arxiv`,
  `openalex`, `unpaywall`, agreed), and `license.status: none_stated` when it gives none. Accept
  never makes an external request (§2.2, §3.1, §5.3).
- **L6, English only in permanent records (devai's D16).** For a non-English item, aiagent's
  verdict carries `metadata_en`, and `quote_en` for its evidence. The service stores those and
  keeps the original only as a link with its sha256; staging, which is temporary, may hold the
  original language. This holds for the metadata snapshots of rejects too (§5.1, §5.3).
- **L15, keys and enrichments.** A work with neither an arXiv id nor a DOI gets the key
  `openalex:W<id>`; the minting order is `arxiv` > `doi` > `ssrn` > `openalex` > `gh` > `wiki` >
  `url`, and an item's OpenAlex id is always registered as an alias (§1.1). Enrichments and
  `GET /v1/candidates/{item_id}` are v1, delivered in devai's Phase 2 (§2.3).
- **L16, stops.** Until the contact address is configured, `/health` says `status: degraded` with
  `contact.configured: false`, `/v1/info` shows each source `down` with cause
  `contact_not_configured`, and a job-creating request for an external source answers 503
  `unavailable` with that cause, so §10 step 0 fails fast. Three causes stop a source until
  someone acts: `contact_not_configured` (every source), `breaker_open` (a denial breaker), and
  `arxiv_fulltext_cap` (the arXiv full-text cap, until the owner records having contacted arXiv;
  it stops arXiv's PDF and HTML downloads only, not its search or OAI-PMH). A stop shows the same
  way whatever its cause (§9.1):
  - `/health` and `/v1/info` show the source `down` with its cause when all its traffic is stopped
    (`contact_not_configured`, `breaker_open`), and `limited` with its cause when only part of it
    is (`arxiv_fulltext_cap`: downloads stop, search goes on); `/health`'s top-level `status` is
    `degraded` while any stop holds (final agreement);
  - a job-creating request whose every named source is stopped answers 503 `unavailable` with
    `error.details.cause`;
  - a request naming a stopped and a working source is accepted (202): in search results the
    stopped source gets a per-source `error`, and fetch items that need it end promptly as
    `failed` with `upstream_unavailable`, still fetchable. The cause is at `error.details.cause`
    there too (final agreement);
  - a source that stops while a job runs gives `eta: {basis: unknown, reason: source_down}`, and
    the job's remaining items for it end the same way;
  - `arxiv_fulltext_cap` fails promptly too, with no quota wait: only the owner can lift it.

  Not stops: an overload pause and a used-up daily budget (OpenAlex's) end by themselves, so jobs
  wait for them (`backoff`, or `quota` until the budget's reset time), outside the items' retry
  budgets.
- **Rate limits: 429s, overload and denial.** A single upstream 429 or 503 is a wait
  (`retry_after` when the answer has a `Retry-After` header, otherwise `backoff`) within the
  item's retry budget, and never opens a breaker: arXiv answers 429 even at compliant rates.
  **Overload** is sustained origin 429s or 5xx despite waiting (a configured count per window, for
  example 5 in 10 minutes): the provider is paused for at least 15 minutes, and the pause clears by
  itself. **Denial** is an origin 403: it opens a breaker that only an operator clears
  (`breaker_open`); for arXiv it stops all arXiv traffic (§3.1, §8.1, §9.1). **What the retry
  budget counts** (final agreement): only failed attempts count against it, and its 30 minutes
  exclude wait time, so an overload pause or a `quota` wait never uses it up and never fails an
  item (§3.1).
- **L13, robots.txt outcomes.** The reason codes are `robots_disallowed` (the rules forbid the
  path), `robots_denied` (robots.txt itself answered 401 or 403, or a challenge) and
  `robots_unreachable` (robots.txt answered 429 or 5xx, or failed on the network: treated as a
  disallow for now, with the cached copy kept and a retry later). A 404, a 410 or any other 4xx for
  robots.txt means "allow" (§2.2, §3.1, O-D12).
- **Names agreed.** The names this document proposed are agreed with devai: the per-source `error`
  in search results (§2.1), `license.status` (`stated` or `none_stated`, §2.2), `fallback_text`
  and `fallback_text_id` (§8.1, §8.2), the job kind `maintenance` (§3.8, §8), `math_spans` offsets
  that include the `$` delimiters (§3.3), and `error.details.cause` for the cause of a stop, on a
  503, in a per-source search `error` and in a per-item failure alike (§2.1, §3.1, §9.3; `cause`
  directly in `error` is gone, final agreement). **Shapes agreed** (final agreement): `refs.cites`
  entries `{html_id, printed, ids: {doi, arxiv, url}}`; `anchor.env` `{name, label, number}`; a
  `heading` unit carries its section's `anchor.html_id`; `license.stated_by` ∈ `arxiv`,
  `openalex`, `unpaywall` (§2.2, §3.3).
- **Endpoints.** `GET /v1/verdicts/{id}` and `GET /v1/library/topics` are v1 too (Minimal first
  version, §9.6).
- **L11, pipelock.** pipelock's own answers are never presented as the source's answer, and never
  open a breaker. Two codes (final agreement): a DLP block is `egress_blocked`, final for that
  request (a failure of that item or that search) and never resent unchanged; a pipelock 5xx or
  connect failure is `egress_unavailable`, a wait retried within the item's retry budget, and the
  item's failure code once the budget is used up (§2.1, §3.1, §9.3).
- **L3, Medium.** Medium is not on the restricted list: undecided until its terms are read. It is
  not a link-only source.
- **Tombstones and purges (v1).** `GET /v1/items/{id}` lists the tombstones of purged texts, and
  `GET /v1/items/{id}/texts/{text_id}/units` is v1: for a purged text it answers 410 `text_purged`
  with the tombstone, so §10 check 3 can pass. Staging-TTL purges run as a maintenance job that
  emits `item.purged` (§3.8).
- **Revoked items.** The 30-day trash (O-D10) is an exception to devai's D12, so devai moved it to
  the owner (O15), who has since approved it (D28, below; §11.1).
- **Acceptance test.** Check 2 is "the PDF, plus the arXiv HTML when the fetch staged it
  (otherwise a reason in `quality.warnings`), plus a `license` field, which may say none stated"
  (final agreement). devai's Phase 3 requires at least one non-arXiv work in the AAD results (an
  open-access copy fetched, or a "known, not held" link), so §10 plans the OpenAlex queries and the
  enrichment explicitly (steps 2 and 6) and adds check 7. devai's Phase 3 exit criteria cite checks
  1-7.
- **Rates.** The service schedules by **rate group**, and `/v1/info` and `eta_inputs` report
  `per_group`: `arxiv-legacy-api` (`export.arxiv.org` and `oaipmh.arxiv.org`: one connection, 3 s,
  shared), `arxiv-files` (`arxiv.org`: the 15 s crawl delay), and one group per other host (§8.3,
  §9.1). ETAs assume 15 s per `arxiv.org` file until an anonymous PDF mirror is cleared for use
  (D27, below; O-D7).

**The final agreement with devai (2026-09-26, from devai).** It settles the last open points; the
notes above are updated where it changed them (marked "final agreement"), and these points are
new:

- **L14, SSRN fetch keys.** An `ssrn:` key is accepted in a fetch request; its item ends
  `not_fetchable` with `terms_disallow_automation`. An open-access copy found elsewhere (OpenAlex,
  Unpaywall) is fetched by its `link_id` (§3.1).
- **L11, "allowed host".** A link record is fetchable only on an allowed host: a host in the
  service's own host list, with its robots.txt and limits (pipelock does not restrict
  destinations). This document says "allowed host" throughout (§2.2, §3.1, §10).
- **L4, code excerpts after a refresh (later; devai's D14, latest version only).** After a repo is
  refreshed to a newer commit, only the latest accepted commit's excerpts are kept (§3.6).

**The owner's answers to devai's own questions (D24-D28, 2026-09-26, relayed by devai).** devai's
plan records them as owner decisions.

- **D24, the embedding model's store.** The embedding model gets its own volume,
  `/var/cache/devai/embed`. It is internal to the service: aiagent reads nothing there (O-D5,
  §7.1).
- **D25, other paths to arXiv stay open.** The lab can still reach arXiv and other sources
  directly: through pipelock, the MCP gateway's servers (arXiv, Wikipedia, fetch) and trusted
  harnesses on devai-net. They share arXiv's per-operator budget with the library, which the owner
  accepts, and nothing is closed. aiagent's ad-hoc fetches through pipelock (`ingest/fetch.py`, for
  the `sentiment` skill) are fine; the library never uses them (principle 6).
- **D26, GROBID** structures PDFs without arXiv HTML, as L12 says (§3.2, §3.5).
- **D27, the anonymous arXiv PDF mirror** (Cornell's public GCS bucket) is checked in devai's
  Phase 1 and used only if its terms and robots.txt allow automated downloads. Until then ETAs stay
  at 15 s per `arxiv.org` file (§8.3, O-D7).
- **D28, revoked items** keep their artifacts 30 days in `trash/`, then lose them: an exception to
  devai's D12, as a safety net against a faulty client. This answers O15 (§5.5, §11.1).

**L5, the arXiv HTML "as fetched" (the owner, 2026-09-26).** When arXiv has the HTML but its fetch
ends in an error after its retries (an HTML error, not a definite "no HTML": §3.1), the accepted
item holds the PDF plus a `quality.warnings` reason, and still passes check 2 (§10). devai's plan
handles it the same way (its risk table: GROBID over the PDF, and a `quality.warnings` reason).

### 0.2 Principles

1. **The service does I/O, storage and indexing, and nothing more.**
   - It never judges relevance, quality or topic, calls no LLM and runs no judging classifier
     (GROBID's parsing models and the embedding model only extract and index).
   - It does only mechanical work:
     - fetch and parse;
     - convert arXiv HTML (v1, the primary text when it exists) and PDF (v1: GROBID for a PDF
       without HTML, pypdfium2 as the fallback) to text;
     - detect language (as a hint only);
     - cap unit size, in characters;
     - embed and index;
     - merge records that share an identifier;
     - check that quoted evidence occurs in a unit or in the metadata.
2. **aiagent decides; the service records.**
   - aiagent decides:
     - which queries to run and which candidates to fetch;
     - which units matter and what the translation says;
     - accept or reject, and which code lines to keep;
     - which topic an item belongs to.
   - The service stores these decisions as aiagent submits them. It refuses one only on integrity
     grounds: an unknown id, a missing precondition, or a conflict.
3. **Every call that creates something is idempotent.**
   - Repeating a request with the same key gives the same result. The same key with a different
     body gets 409.
   - aiagent journals every record it creates in a local outbox before sending it, and after a
     crash replays the exact bytes. A retry therefore always carries the same id and body.
   - An item that is already stored is never downloaded again.
   - A rejected item is never downloaded again without an explicit override.
4. **Ids are stable.**
   - An item id never changes and is never reused. A merged item's id stays as a redirect (§1.5).
   - A unit address (item, text version, unit) resolves forever for every text version of an
     accepted item. A unit of a purged text (a rejected item, or expired staging) resolves to a
     tombstone `{status: purged, reason}`; a reject's evidence keeps its quotes inline (§5.2).
5. **Everything is auditable and stamped.**
   - Each service action records what it did: the source URL, retrieval time, HTTP status, the
     sha256 of the bytes, and the extractor's name and version.
   - Each aiagent record carries the profile version, the query, the deciding model or student, and
     `client {name, version}`. The service adds `recorded_at` to every record.
   - Records are append-only. A correction is a new record that supersedes the old one.
6. **Credentials never enter the lab.**
   - aiagent never sees a source credential and never contacts an external source for library
     work.
   - `ingest/fetch.py`'s path through pipelock stays, for ad-hoc URLs given to the `sentiment`
     skill (D25, §0.1). The library does not use it.
7. **The volume comes first for bulk data and status**, as with the laya trainer.
   - Job status and staged text are files on a shared volume that aiagent reads directly.
   - HTTP carries control messages, result pages and small records.
8. **The store layout belongs to devai.** aiagent uses a path only when the API returns it
   (relative to the library root) and never builds one itself, so devai can change the hierarchy
   freely.
9. **The plain-file store is the only source of truth.**
   - Every item record, alias, verdict, topic assessment, translation record, excerpt and profile
     version is an append-only file in the library directory (L8).
   - The alias index, the topic listings and the keyword and vector indexes are derived, and can be
     rebuilt from those files. The candidate cache is derived or disposable (a search refills it).
   - A database the service keeps is a derived cache, never the store of record. §10 check 6
     proves it.
10. **The library is private.** It is served only on devai-net, to the lab. Any later consumer (an
    MCP server, say) needs a licence gate before content leaves devai-net (§5.3).

### 0.3 aiagent-side assumptions

| # | Assumption |
|---|---|
| A1 | aiagent reaches the service directly on devai-net: httpx with `trust_env=False`, never through pipelock, like `distill/client.py`. The settings are new: `library_api_base`, `library_api_key` (empty means no header) and `library_dir`. |
| A2 | The lab mounts the library root **read-only** at `/library` (O-D2). v1 needs nothing else. Later a **read-write `inbox/`** carries submissions over the 8 MiB inline limit, the same pattern as `/laya`. Secrets are never in the library directory. |
| A3 | **Students.** A student is a laya single-input, single-output, closed-label classifier on CPU, taking 50-110 ms per decision after a load of about 1 s, and run in batches through `run --jsonl`. For `laya-multilingual`, `max_len` is 1024 and `head_max_len` is 256. The worst-case room for the state is therefore **764 tokens**: 1024 minus (256 + 3) minus 1, as in `SequenceTokenizer.room` with a question that fills the head budget (this holds for questions of at most 62 options). The state is the JSON object `{field: text}`, and its escaping of backslashes, quotes and newlines adds 8-16 % (measured). TeX math is dense in backslashes. **Topic text never goes into a unit state**; the state holds the unit and at most a short breadcrumb. |
| A4 | aiagent counts tokens itself and makes the final fit check with its own tokenizer port (`system1/sequence.py`, `distill/segment.py`), splitting further when needed. The service caps units in characters only (§3.4). |
| A5 | aiagent's LLM is the local devai 27B model (4 requests in flight, context about 118k tokens), so a whole paper fits in one prompt. aiagent's own throughput for translation and judging has not been measured yet. |
| A6 | The skills that judge library items (the relevance and quality questions, their students and the verdict program) are aiagent's future work, distilled through the existing `distill` pipeline. They are not the service's concern. **Students are optional:** a verdict may carry an LLM decider alone (`system1_mode` off), and v1 runs that way. A student question is fixed per profile version and never names a topic, because the question text is bound into `question_set_sha256`, so a topic-specific question would need a new student per topic. The inner bound (the query's topic) is judged by the LLM. |
| A7 | aiagent keeps no library content in its home directory beyond caches and its outbox. The library directory is the store. |
| A8 | The research profile's source of truth is aiagent. The service stores registered versions as opaque copies (§6). |
| A9 | Every record aiagent submits carries `client: {"name": "aiagent", "version": "…"}`, so a later consumer names itself the same way. |

### 0.4 What the service does not do (from aiagent's side)

- It does not rank against the profile or a topic. It returns retrieval scores only.
- It does not deduplicate semantically, for example by similar titles. It matches mechanically:
  by identifier, by artifact hash, and (later) by exact normalised title (§1.5). aiagent marks
  duplicates itself (`duplicate_of`, §5).
- It does not translate, summarise or generate answers.
- It does not plan queries. For a plain-text query it only maps the text onto each source's
  all-fields search, mechanically (§2.1).

---

## 1. Identifiers

### 1.1 Items

An **item** is one work, whichever source it came from.

- **Canonical key** (`canonical_key`): a readable string of the form `<namespace>:<local id>`.
  - It is minted once, from the strongest identifier known when the item is first seen, in this
    order: `arxiv` > `doi` > `ssrn` > `openalex` > `gh` > `wiki` > `url` (`openalex` added
    2026-09-26, from devai: the key of a work with neither an arXiv id nor a DOI).
  - **Registry DOIs map to their namespace** before minting and before any alias lookup:
    `doi:10.48550/arxiv.<id>` (arXiv's DataCite DOI) becomes `arxiv:<id>`, and
    `doi:10.2139/ssrn.<id>` becomes `ssrn:<id>`.
  - It never changes afterwards, even if a stronger identifier turns up later. That identifier
    becomes an alias.
- **Item id** (`item_id`): `"itm-"` plus the first 24 hex characters of
  `sha256(canonical_key as UTF-8)`.
  - It is path-safe and fixed-length, and fits a hashed directory hierarchy.
  - It is deterministic **per canonical key**. Two jobs that find the same work under different
    keys can still mint two ids; §1.5 merges them.
  - 96 bits leaves no practical collision risk at millions of items.
- **Aliases** (`aliases`): every other identifier of the item, in the same `<namespace>:<id>` form,
  including `sha256:<hex>` for each artifact that passed the format check and extraction (§1.5).
  **Any alias resolves to the item** (`GET /v1/items/resolve?key=…`).
  - An item's OpenAlex id, whenever it is known, is registered as an `openalex:` alias, whatever
    the canonical key.
  - Aliases are registered by **a single writer** in the service, so two jobs cannot bind one
    alias to two items unnoticed.

| namespace | local id, normalised | version |
|---|---|---|
| `arxiv` | Without the version. New style as is (`2403.01234`). Old style `archive/NNNNNNN` with a lowercase archive and no subject class (`math/0501001`). | `v1`, `v2`, … |
| `doi` | Lowercase, with no `https://doi.org/` or `doi:` prefix. arXiv and SSRN DOIs map as above. | – |
| `ssrn` | The numeric abstract id. | the SSRN revision date |
| `openalex` | The OpenAlex work id, `W` plus digits (`W2741809807`), with no `https://openalex.org/` prefix. | – |
| `gh` | `<owner>/<repo>`, lowercase. | the commit sha (40 hex) |
| `wiki` | `<lang>:<Title_with_underscores>`. | the revision id (`oldid`) |
| `url` | The normalised URL: `https`, lowercase host, no fragment, no tracking parameters, no default port. | the sha256 of the content |

`kind` ∈ `paper`, `repo`, `web_page`, `wiki_article`.

### 1.2 Versions

- Every stored artifact names the item version it belongs to (`version`).
- The item record shows `versions.stored` (the version recorded at accept) and
  `versions.latest_seen`, with the time the latter was checked.
- A newer upstream version never replaces stored content by itself (L5). Replacing it is an
  explicit refresh (§5.5, and O2).
- **Withdrawn upstream.** When arXiv's latest version of an accepted item is a withdrawal notice
  (known from `arXivRaw`, best effort, §3.1), the item record gets `upstream_status: withdrawn`.
  Nothing is deleted automatically.

### 1.3 Text versions and units

- **Text version** (`text_id`): `"tx-"` plus 24 hex of `H(identity)`, where `identity` is
  `{item_id, version, origin, source, extractor, segmenting, units_sha256}`:
  - `source` is the source artifact's sha256, or for a translation its `translation_id`;
  - `extractor` is `{name, version, config_sha256}`, or for a translation the translator;
  - `units_sha256` is the sha256 of the `units.jsonl` bytes.

  The same extractor on the same input gives the same `text_id`, so re-extraction is idempotent. A
  new extractor version or a new segmenting setting gives a new `text_id`. A translation is a text
  version of its own (§4).
- **Unit** (`unit_id`): `"u"` plus a 6-digit ordinal within its text version (`u000123`), in
  reading order, with no gaps. The full address is `<item_id>/<text_id>/<unit_id>`.
- **Retention:**
  - The primary English text of an **accepted** item is never deleted.
  - Every text of a **rejected** item is deleted (L5). Its addresses resolve to a tombstone
    `{status: purged, reason}`, and the reject verdict keeps each evidence quote inline (§5.2).
  - On accept of a **translated** item, the source-language text version is deleted once its
    anchors are copied into the translation (§4, O6).
  - A staged text with no verdict expires with its staging (§3.8).

### 1.4 Other ids

All ids use the trainer's style: `<prefix>-<hex>`.

| id | form | minted by |
|---|---|---|
| `job_id` | `lbjob-` + 24 hex (random) | service |
| `query_id` | `q-` + 24 hex (random) | aiagent, one per planned query, passed through for correlation |
| `topic_id` | `[a-z0-9-]{1,64}`, e.g. `aad-comp-finance` | aiagent (it owns topic normalisation) |
| `verdict_id`, `assessment_id`, `translation_id` | `vd-` / `ta-` / `tr-` + 24 hex (random) | aiagent; this is also the idempotency key |
| `excerpt_id` (later) | `ex-` + 24 hex of `H([repo_key, commit, path, line_start, line_end])` | service |
| inbox submission (later) | `<kind>-` + 12 hex of `sha256(manifest.json bytes)` (`tr-…`) | aiagent, as for datasets `ds-…` |
| profile version | `{name, version: int, sha256}` | aiagent (§6) |

`H(x)` is aiagent's existing canonical hash: the sha256 of compact JSON with key order kept
(`system1/contract.py`).

### 1.5 Duplicates, and rejected items are never fetched again

- **Before any download**, the service maps the requested key (§1.1) and resolves it through the
  alias index:
  - a current `reject` verdict: the fetch item ends as `rejected_skipped`, with no network I/O,
    unless the request carries an override (§5.5);
  - an accepted item: `already_stored`.
- **Format check before hashing.** A downloaded body is checked for its format, its Content-Type
  and a minimum size. A PDF must start with the magic bytes `%PDF-`; a PDF body that fails (an
  HTML error page, a captcha, "PDF being generated") fails the item as `unsupported_format` or
  `upstream_error`. arXiv HTML counts only if the answer is 200, `text/html`, carries LaTeXML's
  generator marker and names the resolved version `vN` (2026-09-26, from devai). A definite "no
  HTML" (a 404 or 410, or a 200 without the marker) means the item is staged from its PDF alone,
  which is not a failure. A 429, a 5xx, a network failure, or a 200 with the marker but the wrong
  content type or version, is an error, retried within the item's retry budget; a used-up budget
  also stages the item from its PDF alone (§3.1; agreed with devai). A failed body is never stored
  and never becomes a `sha256:` alias, so it cannot poison the rejected-bytes check.
- **After a download** that passed the check, the service hashes the bytes:
  - an alias of a rejected item (the same PDF under another URL): it deletes the bytes and reports
    `duplicate_of_rejected` with the other item's id;
  - an alias of an accepted item: it adds the requested key as an alias, deletes the bytes, stages
    nothing, and reports `duplicate_of_stored`.
- **Search results** carry `library.status` for every candidate, so aiagent can skip known ones
  before it asks for anything.
- **Metadata-only rejections are real records.** A candidate that aiagent rejects from its metadata
  alone, never downloaded, still gets an item record: a metadata snapshot, in English (§5.1), plus
  the verdict (§5.3).
- **Merges.** When the single alias writer finds that two existing items share an alias:
  - the earlier item id survives, and the other becomes a redirect record (`redirect_to`) that
    `resolve` follows;
  - nothing is deleted: the survivor lists the other's aliases, artifacts and records;
  - the service emits `item.merged` (§8.2);
  - if the two current verdicts disagree (one accept, one reject), the survivor gets
    `verdict_conflict: true`, and any new fetch or verdict for it gets 409 `alias_conflict` until an
    owner-requested override supersedes both (§5.5). aiagent shows the conflict to the owner.
- **The same work with different bytes** (a journal PDF and an arXiv copy) cannot be recognised by
  identifier or hash. **Later:** `POST /v1/items/match {title, first_author, year}` returns items
  of any status whose normalised title (casefold, NFKC, alphanumerics only) matches exactly,
  narrowed by the first author and year when given. It is mechanical; aiagent decides whether it
  is a duplicate.

---

## 2. Search

### 2.1 Request

A search queries external sources, so it waits on rate limits. It is therefore a **job**:
`POST /v1/searches` returns 202 with the job at once, and aiagent polls it (§8).

```json
{"query_id": "q-3f9a0c1d2e4b5a6978877665",
 "topic_id": "aad-comp-finance",
 "query": {"text": "adjoint algorithmic differentiation Greeks Monte Carlo",
           "native": {"arxiv": "abs:\"adjoint\" AND abs:\"differentiation\" AND (cat:q-fin.CP OR cat:q-fin.PR OR cat:math.NA)",
                      "openalex": "filter=title_and_abstract.search:adjoint differentiation,open_access.is_oa:true"}},
 "sources": ["arxiv", "openalex"],
 "filters": {"date_from": "2000-01-01", "date_to": null,
             "categories": ["q-fin.CP", "q-fin.PR", "q-fin.RM", "math.NA", "cs.MS"],
             "authors": [], "languages": null, "kinds": ["paper"], "free_only": true},
 "max_results": 200,
 "sort": "relevance",
 "cursor": null,
 "enrich": [],
 "client": {"name": "aiagent", "version": "0.6.0"}}
```

| field | need | meaning |
|---|---|---|
| `query.text` | must | Plain text. The service maps it onto each source's default all-fields search, mechanically. |
| `query.native` | must | A source-specific query string per source (arXiv's query syntax, the OpenAlex `search`/`filter` string, …), passed through unchanged. aiagent plans these, and this is how it gets precision. |
| `sources` | must | Source names from `GET /v1/info`. v1: `arxiv`, `openalex` (L15). |
| `filters.*` | must: dates, categories, free_only. should: the rest | Applied natively where the source supports them, otherwise after retrieval (mechanically, on metadata). The response says which way each filter was applied (`applied: {"categories": "native", "languages": "post"}`). v1 categories are arXiv's and filter the arXiv source only; aiagent narrows OpenAlex through `query.native.openalex`. |
| `max_results` | must | The cap per source, not in total. |
| `cursor` | must | Continues an earlier search's upstream pagination: `next_cursor` from that job's results. |
| `enrich` | later | Any of `author_metrics`, `citations`, `code_links`. These cost extra upstream requests, so they are off by default; v1 enriches only the shortlist, through `POST /v1/enrichments` (§2.3). |

- **Results** are a stored snapshot: `GET /v1/jobs/{id}/results?cursor=…`. Paging through them
  never queries upstream again.
- **Per source**, the results report `{source, upstream_total, returned, next_cursor, error}`, so a
  short result set is visible. `error` is null, or `{code, message, details}` when that source's
  part of the search failed, with `code`:
  - `egress_blocked`: the service's own egress proxy (pipelock) blocked the request (DLP). That is
    final for this request, which is never resent unchanged, and never presented as the source's
    answer (L11 note);
  - `egress_unavailable`: the proxy itself failed (a 5xx or a connect failure) and still did after
    the retries (§3.1); not the source's answer either;
  - `upstream_unavailable`: the source did not answer, even after the retries, or it is stopped.

  `details` is null, except for a stopped source, where `details.cause` names the stop
  (`contact_not_configured` or `breaker_open`; `arxiv_fulltext_cap` never stops a search) (§9.1).
  The cause of a stop is always at `error.details.cause`, as in a 503 and a per-item failure
  (agreed with devai). export.arxiv.org sometimes returns an empty page: when a page comes back
  empty while `upstream_total > 0`, the service retries it once before reporting it.
- **Stopped sources.** A search whose every named source is stopped is refused with 503
  `unavailable` and `error.details.cause`; one that names a stopped and a working source is
  accepted (202), and the stopped source reports the per-source `error` above (§9.1). A single
  upstream 429 or 503 is a wait, not an error (§8.1).
- **Merging.** Records that share an identifier within one result set (for example an arXiv record
  and an OpenAlex record with the same DOI) are merged into one candidate. `sources_seen` keeps
  each source's rank. The merge is mechanical, by identifier only.

### 2.2 Candidate record

This is what aiagent needs to plan and to pre-judge **without downloading**. A student pre-judge
(later) uses a state built from title, authors, venue, categories and abstract (about 150-450
tokens). The LLM pre-judge uses the whole record.

| field | type | need | note |
|---|---|---|---|
| `item_id`, `canonical_key`, `kind` | str | must | §1.1. Minted for candidates too, even if never fetched. |
| `ids` | object | must | `arxiv`, `doi`, `ssrn`, `openalex`, `s2`, …; null when unknown. |
| `sources_seen` | list | must | `{source, rank, score, job_id}` per source. |
| `title` | str | must | Plain text; math kept as `$…$`. |
| `authors` | list of Author | must | In byline order (see below). |
| `abstract` | str or null | must, when the source has one | Plain text; math as `$…$`. |
| `language` | object | must | `{declared, detected, detector, confidence}`, for the title and abstract. |
| `categories` | list | must for arXiv | `{scheme, primary, all}`. v1 scheme: `arxiv`. Later: `msc`, `acm_ccs`, `jel`, `openalex_topic`. |
| `dates` | object | must | `submitted`, `updated`, `published` (ISO 8601 dates). For arXiv, `submitted` is the date of v1 and `updated` that of the latest version. |
| `versions` | object | must for arXiv | `{latest, list: [{version, date, withdrawn}]}`. A search fills only `latest`, with the dates of v1 and of the latest version in `dates`: that is all the API's Atom feed gives. `list` stays null until a fetch job has read the `arXivRaw` record (§3.1), which gives the full list and `withdrawn` (best effort; true for a withdrawal notice). (2026-09-26, from devai.) |
| `venue` | object or null | should | `{name, kind, issn, publisher, open_access}`. `kind` ∈ journal, conference, preprint_server, repository, blog, wiki, forum. |
| `journal_ref`, `comments` | str or null | should | arXiv's free-text fields. They often name the venue, the page count and a code URL. |
| `license` | object or null | should | `{status, spdx, url, stated_by}`: `status` ∈ `stated`, `none_stated` (agreed with devai). `stated_by` ∈ `arxiv`, `openalex`, `unpaywall` (agreed with devai) names the source that stated it: `arxiv` (its `arXivRaw` record) for an arXiv work; for another work, the source of the open-access location its copy came from, `openalex` (`locations[].license`) or `unpaywall` (`best_oa_location.license`). `url` is the licence URI as the source gives it, arXiv's legacy URIs included; `spdx` is null when the URI has no SPDX id. `none_stated` (arXiv papers before 2007, or an open-access location that gives no licence) has `spdx` and `url` null. The whole field is null until a fetch job has recorded it: for arXiv, when it has read `arXivRaw`; for another work, when it has fetched the copy (§3.1). The item record must carry it on accept (§5.3). |
| `access` | object | must | `{free, fetchable, formats, est_bytes, robots_allowed, reason}`. `fetchable` is false for a link-only source (L14, `reason: terms_disallow_automation`), or when robots.txt does not allow the fetch (robots.txt governs page and file fetches, L13; `robots_allowed: false`), with `reason` ∈ `robots_disallowed` (the rules forbid the path), `robots_denied` (robots.txt itself answered 401 or 403, or a challenge) or `robots_unreachable` (robots.txt answered 429 or 5xx, or failed on the network: a disallow for now, with the cached copy kept and a retry later, so the item may become fetchable). A 404, a 410 or any other 4xx for robots.txt means "allow" (2026-09-26, from devai). aiagent never fetches an item with `free: false` or `fetchable: false`, and records no verdict for it; the one way in is a fetchable open-access location of the same work in `links_found`, fetched by its `link_id` (L14, L15). `formats` for arXiv: `pdf`. A search cannot cheaply tell whether arXiv has the HTML, so aiagent always asks for `want: ["pdf", "html"]`, and the fetch reports whether it was there (2026-09-26, from devai). `est_bytes` feeds the ETA. |
| `last_fetch_error` | object or null | must | `{code, at}` of the latest failed fetch (§3.1). |
| `links` | object | must | `landing`, `pdf`, `html`, `doi`: whichever exist (for arXiv, `html` once a fetch has found it). A link-only work (L14) has at least `landing`. |
| `code_links` | list | should | Link records (below) whose `kind` is `repo`. At search time `found_in` is `abstract`, `comments` or `metadata`; links found in the full text come with staging (`refs.links`, §3.3). |
| `links_found` | list | must for open-access locations (v1); should for the rest | Every link found in the metadata, as **link records**: `{link_id, kind, url, resolved_key, fetchable, reason, found_in}`. `link_id` is `lnk-` + 24 hex issued by the service; `url` is for display and citation only (aiagent never sends it back); `kind` ∈ `repo`, `author_page`, `paper`, `doi`, `other`; `resolved_key` is the item key it maps to, if any; `fetchable` says whether the service may fetch it (never unless it is on an allowed host), with `reason` if not. An open-access location from OpenAlex or Unpaywall (`found_in: openalex` or `unpaywall`) is a `paper` link record: fetchable on an allowed host, otherwise "known, not held" (`fetchable: false`, with `reason`) (L15). |
| `citations` | object or null | v1 where OpenAlex provides it | `{cited_by, references, source, as_of}`. |
| `library` | object | must | `{status, verdict, stored_version, staged_text_id, upstream_status}`. `status` ∈ none, staged, accepted, rejected. `verdict` is `{verdict_id, decision, basis, scope_topic_id, decided_at, profile_version, topic_ids}` or null. |
| `retrieved_at` | datetime | must | When this record was fetched from upstream. |

**Author:**

```json
{"position": 1, "name": "Ada Example", "ids": {"orcid": null, "openalex": "A5000000001", "s2": null},
 "affiliations": [{"name": "Example University", "ror": "https://ror.org/00example", "type": "education", "country": "GB"}],
 "metrics": {"h_index": 31, "i10_index": 60, "works": 120, "cited_by": 5400,
             "source": "openalex", "as_of": "2026-09-01",
             "match": {"method": "source_id", "confidence": 0.9}}}
```

- arXiv fills `position` and `name`, and an affiliation name where arXiv gives one. In v1,
  OpenAlex (L15) adds `ids.openalex` (and `ids.orcid` where it has one), the ROR fields and
  `metrics` (the h-index, advisory, with provenance), where it has the author. `ids.s2` waits for
  Semantic Scholar (later).
- `affiliations[].type` is the ROR organisation type (education, company, government, nonprofit,
  …). It is **information only, never a gate**: "independent" describes who publishes the document
  fetched, not the authors' employers (O4).
- `metrics` is null until enriched. `match.method` ∈ `orcid`, `source_id`, `name_affiliation`,
  `name_only`. Author disambiguation is imperfect, and aiagent weighs a `name_only` h-index
  accordingly. **The metric's source, as-of date and match method are required whenever a metric
  is given.**
- **No author database.** Author metrics and affiliations live only inside candidate snapshots and
  verdicts.

### 2.3 Enrichment (v1: author metrics and citations; code links later)

Enrichments and `GET /v1/candidates/{item_id}` are v1; devai delivers them in its Phase 2, with
OpenAlex (2026-09-26, from devai).

- `POST /v1/enrichments` `{"item_ids": [...], "enrich": ["author_metrics", "citations",
  "code_links"]}` is a job like any other.
- v1 enriches from OpenAlex (L15): `author_metrics` (the h-index, advisory, with its source,
  as-of date and match method) and `citations` (`cited_by`). A candidate that search already
  merged with an OpenAlex record carries `citations` without enrichment. `code_links` comes later.
- aiagent's flow is:
  1. a wide, cheap search;
  2. a pre-judge on the metadata;
  3. enrichment of the shortlist only;
  4. an LLM pre-judge;
  5. a fetch.

  Enriching every candidate would spend the upstream rate budget, and OpenAlex's daily budget
  (L15), on items that are about to be dropped.
- **Enrichment results are cached** with their `as_of`, so a re-run makes no upstream requests.
- `GET /v1/candidates/{item_id}` (v1) returns the latest candidate record. The candidate cache
  **should** keep records for at least 30 days, so a fetch or a verdict by `item_id` never needs a
  re-search.

---

## 3. Fetching and staging

### 3.1 Fetch job

`POST /v1/fetches` returns 202 with the job at once:

```json
{"query_id": "q-3f9a0c1d2e4b5a6978877665",
 "items": [{"item_id": "itm-5d41402abc4b2a76b9719d91", "want": ["pdf", "html"], "version": "latest"},
           {"key": "doi:10.1234/example.5678", "want": ["pdf"]},
           {"link_id": "lnk-7e2c9a14b05d3f6a81c4e290", "want": ["pdf"]}],
 "client": {"name": "aiagent", "version": "0.6.0"}}
```

- Each item is given by exactly one of `item_id`, `key` (a source-native identifier the
  service knows how to resolve: `arxiv:`, `doi:`, `ssrn:`; an `ssrn:` key is accepted, but an
  SSRN work is link only, so its item ends `not_fetchable` with `terms_disallow_automation`, L14,
  agreed with devai) or `link_id` (a link record the service issued, below).
  **Never a URL (L11).**
  - An unknown `key`, or a fetchable `link_id` whose `resolved_key` names no known item, is minted
    as a new item (§1.1), after the robots.txt check. A `link_id` whose `resolved_key` names a
    known item (an open-access location of the same work) fetches for that item.
  - Author pages and repos that a paper links to, and the open-access locations OpenAlex and
    Unpaywall report (L15), are reached only through their `link_id`.
  - An item whose `access.fetchable` is false, or a `link_id` whose target is not on an allowed
    host, returns `not_fetchable` with a reason (`source_not_allowed` for a target not on an
    allowed host, `terms_disallow_automation`, `robots_disallowed`, `robots_denied`,
    `robots_unreachable`, `not_free`; the robots codes as in §2.2); aiagent records no verdict for
    it. For a link-only work (L14) the service has already tried OpenAlex and Unpaywall; another
    open location it found is a separate link record in the candidate, which aiagent fetches by
    its `link_id` for the same item.
- **Idempotency:**

  | item state | fetch result, with no upstream request |
  |---|---|
  | already accepted | `already_stored` |
  | already staged | `already_staged` |
  | rejected | `rejected_skipped` (per item; a batch fetch never fails for it) |

- The request carries an `Idempotency-Key` header. Resubmitting it returns the same job.
- The job reports `upstream {requests, bytes}`. That is how aiagent's acceptance test proves that a
  re-run downloads nothing (§10).
- **Per-item outcomes** (in the job results):
  - `staged`, `already_stored`, `already_staged`, `rejected_skipped`, `duplicate_of_rejected`,
    `duplicate_of_stored` (§1.5), `withdrawn_upstream` (the requested version is a withdrawal
    notice: nothing is stored), `not_fetchable` (with its `reason`, above);
  - or `failed`, with an `error {code, message, details}`; `code` ∈ `robots_disallowed`,
    `robots_denied`, `robots_unreachable`, `not_free`, `not_found_upstream`, `upstream_error`,
    `upstream_unavailable`, `egress_blocked`, `egress_unavailable`, `too_large`,
    `unsupported_format`, `no_text_layer` (a scanned PDF, O7), `dataset_refused`,
    `extraction_failed`.
  - The robots codes are those of §2.2, found at fetch time. `robots_unreachable` is transient: the
    service retries robots.txt later, and the item stays fetchable (2026-09-26, from devai).
  - The service's own egress proxy (pipelock) is never presented as the source's answer, and
    never opens a breaker (agreed with devai):
    - `egress_blocked`: pipelock blocked the request by DLP. That is final for the request: the
      item fails at once, and aiagent does not resend it unchanged;
    - `egress_unavailable`: pipelock answered 5xx or the connection to it failed. That is a wait,
      retried within the item's retry budget (below); the item fails with this code only once the
      budget is used up.
  - `details` is null, except for an item that needs a stopped source: `upstream_unavailable`
    with the stop in `details.cause` (`contact_not_configured`, `breaker_open` or
    `arxiv_fulltext_cap`, §9.1). The cause of a stop is always at `error.details.cause`, here, in
    a per-source search `error` (§2.1) and in a 503 (§9.3).
  - **A fetch failure is not a verdict.** The item stays fetchable, and the candidate record keeps
    `last_fetch_error`.
- **Retry budget.** Each item gets at most 5 failed attempts or 30 minutes, whichever ends first
  (a configured value, reported by `/v1/info`). Only failed attempts count, and the 30 minutes
  exclude wait time: waiting for a rate slot, an overload pause or a `quota` reset never uses up
  the budget and never fails an item (agreed with devai). Once the budget is used up, the item
  fails as `upstream_unavailable` (`egress_unavailable` when it was pipelock that kept failing),
  still fetchable. A source outage therefore ends a job instead of leaving it `waiting` in
  `backoff`.
  - A single upstream 429 or 503 is a failed attempt within this budget, followed by a wait:
    `retry_after` when the answer has a `Retry-After` header, otherwise `backoff`. It never opens a
    breaker (§9.1). A pipelock 5xx or connect failure (`egress_unavailable`) is retried the same
    way.
  - A stopped source (§9.1) fails its items promptly as `upstream_unavailable` with the stop in
    `error.details.cause`, still fetchable, without waiting out the budget. This includes
    `arxiv_fulltext_cap`, which only the owner can lift, so there is no quota wait for it.
  - An overload pause and a used-up daily budget are not stops, and they do not count against the
    budget: the job waits (`backoff` for the pause, `quota` until the budget's reset time) (§8.1).
- **What is fetched per source:**
  - **arXiv** (2026-09-26, from devai): the job resolves the latest version `vN` once. It then
    fetches `/pdf/<id>vN` and `/html/<id>vN` from `arxiv.org`, at its Crawl-delay of 15 s (rate
    group `arxiv-files`, O-D7), and that version's `arXivRaw` record, with the full version list,
    the withdrawal flag (best effort) and the licence, as one paced request in the
    `arxiv-legacy-api` group. The HTML's coverage is broad (even 1999 papers).
    - **Errors on the HTML request** (agreed with devai): a definite "no HTML", a 404 or 410 or a
      200 without LaTeXML's generator marker, stages the item with the PDF alone at once. A 429, a
      5xx, a network failure, or a 200 with the marker but the wrong content type or version (the
      check in §1.5), is an error, retried within the item's retry budget like the PDF; once the
      budget is used up, the item is staged with the PDF alone too. Neither is a failure:
      `text.json`'s `quality.warnings` says why (§3.2), and the HTML is fetched later only on
      request (§5.5).
    - With the HTML, the staged texts are the HTML (primary) and the PDF's text (fallback); without
      it, the PDF through GROBID (primary) (§3.2).
    - The e-print is never fetched: arxiv.org/robots.txt disallows `/e-print` and `/src` (L12). If
      the latest version is a withdrawal notice, the outcome is `withdrawn_upstream`.
    - Under the full-text cap (`arxiv_fulltext_cap`, §9.1) the item ends promptly as `failed`,
      `upstream_unavailable` with that cause in `error.details.cause`, still fetchable.
  - **Works found through OpenAlex (v1):** the PDF at an open-access location (from OpenAlex or
    Unpaywall) on an allowed host, through its link record (L15). The fetch job records the
    licence that location states (OpenAlex's `locations[].license`, Unpaywall's
    `best_oa_location.license`), with `stated_by` naming the source, or `status: none_stated` when
    it gives none (§2.2; 2026-09-26, from devai).
  - **Others (later):** whatever `want` names and the source offers, through official APIs; a
    link-only source (L14) is never fetched.

### 3.2 Staged text: files on the volume

After extraction the service writes one directory per text version. It is immutable once written,
and it is written atomically: into a temporary name, then renamed.

```
staging/<…item hierarchy…>/<text_id>/
  text.json      # manifest
  units.jsonl    # one unit per line, in reading order
  SHA256SUMS
```

- The directory holds exactly these three files: no symlinks and no subdirectories (the trainer's
  dataset rule). Artifacts (PDF, HTML) live elsewhere, at paths the API returns.
- The API returns the directory as `staging_path` (relative to the library root). aiagent reads
  from there, or through `GET /v1/items/{id}/texts/{text_id}/units?from=u000010&to=u000040`
  (v1). For a purged text that endpoint answers 410 `text_purged` with the tombstone (§1.3).
- `units.jsonl` has one JSON object per line, so `grep`, `jq` and `wc -l` work on it directly (L8).

**`text.json`** (a v1 example: an item without arXiv HTML, its PDF structured by GROBID; the
warning says why there is no HTML text):

```json
{"schema_version": 1,
 "item_id": "itm-5d41402abc4b2a76b9719d91", "version": "v3",
 "text_id": "tx-7c9e6679f1a2b3c4d5e6f708",
 "origin": "pdf", "role": "primary",
 "source_artifact": {"kind": "pdf", "sha256": "…"},
 "pdf": {"sha256": "…", "pages": 31},
 "lang": {"primary": "en", "detector": "…", "share_non_en_units": 0.0},
 "extractor": {"name": "…", "version": "…", "config_sha256": "…"},
 "segmenting": {"unit_max_chars": 1300},
 "units_sha256": "…",
 "counts": {"units": 312, "chars": 98231},
 "outline": [{"number": "3.2", "title": "Adjoint mode", "first_unit": "u000087"}],
 "quality": {"math": "pdf_text", "page_method": "pdf_native",
             "warnings": ["no arXiv HTML: arxiv.org answered 404 for /html/<id>v3"]},
 "files": {"units.jsonl": {"sha256": "…", "rows": 312}}}
```

The primary text of an item **with** arXiv HTML (another item; HTML-primary text is v1, L12 note):

```json
{"schema_version": 1,
 "item_id": "itm-0c8f3e2a9b7d6c5e4f3a2b1c", "version": "v2",
 "text_id": "tx-2b7f0c9d4e8a1f3b6c5d7e90",
 "origin": "html", "role": "primary",
 "source_artifact": {"kind": "html", "sha256": "…"},
 "pdf": {"sha256": "…", "pages": 28},
 "lang": {"primary": "en", "detector": "…", "share_non_en_units": 0.0},
 "extractor": {"name": "…", "version": "…", "config_sha256": "…"},
 "segmenting": {"unit_max_chars": 1300},
 "units_sha256": "…",
 "counts": {"units": 298, "chars": 101544},
 "outline": [{"number": "3.2", "title": "Adjoint mode", "first_unit": "u000084"}],
 "quality": {"math": "tex", "page_method": "text_align", "warnings": []},
 "files": {"units.jsonl": {"sha256": "…", "rows": 298}}}
```

- `origin` ∈ `html` (v1: arXiv HTML, L12), `pdf` (v1; `extractor` names GROBID, or pypdfium2),
  and later `markdown`, `code_excerpts`, `translation`.
- `role` (v1, adjusted 2026-09-26): when an item has the arXiv HTML, the HTML text is `primary`,
  and the PDF's text is staged too, as `fallback` (from its text layer; GROBID runs only for PDFs
  without HTML). The fallback is also what the HTML text's pages are aligned to. Without arXiv HTML
  (a definite "no HTML", a retry budget used up on the HTML request, both as in §1.5, or a failed
  conversion), the PDF text through GROBID is `primary`, with pypdfium2 as its fallback
  extractor.
- `quality.warnings` lists what the extraction could not do. For an arXiv item staged from its PDF
  alone it says why there is no HTML text: a definite "no HTML" (a 404 or 410, or a 200 without
  LaTeXML's generator marker), a used-up retry budget (§3.1), or a failed conversion. The wording
  is devai's; aiagent shows the warnings and does not parse them. §10 check 2 relies on it: an
  accepted arXiv item without the HTML artifact has such a reason.
- `quality.math` is `pdf_text` for the formulas as the PDF text holds them (GROBID gives them as
  text), and `tex` for arXiv HTML, where every formula carries its TeX
  (`annotation encoding="application/x-tex"`, verified by devai, L12 note).
- `quality.page_method` is `pdf_native` for PDF text and `text_align` for arXiv HTML (§3.3).
- `outline` holds the section headings the extractor found; it may be empty.

### 3.3 Unit record

A v1 unit, from a PDF structured by GROBID (the item without HTML): its section and paragraph
anchors come from GROBID; the four HTML-only fields (`html_id`, `env`, `refs.cites`,
`refs.labels`) are null, and `math_spans` is empty, since the formulas of PDF text are plain text:

```json
{"unit_id": "u000091", "kind": "paragraph",
 "text": "In adjoint mode the cost of all first-order sensitivities is a small multiple of the cost of the pricer, x̄ = ȳ ∂y/∂x, independent of the number of inputs [12].",
 "lang": "en",
 "anchor": {"section_path": [{"number": "3", "title": "Algorithmic differentiation", "label": null},
                             {"number": "3.2", "title": "Adjoint mode", "label": null}],
            "paragraph": 2, "block": 4, "html_id": null, "part": {"index": 0, "of": 1},
            "equation_labels": [], "equation_number": null, "equation_number_method": null,
            "env": null, "float": null, "footnote": null,
            "pdf_page": 7, "pdf_page_end": 7, "page_method": "pdf_native", "page_confidence": null},
 "counts": {"chars": 160},
 "math_spans": [],
 "refs": {"cites": null, "labels": null, "links": []},
 "prev": "u000090", "next": "u000092"}
```

A v1 unit from arXiv HTML (the item with HTML): the unit carries its LaTeXML id as `html_id`
(`S3.SS2.p2`), from which the paragraph number comes; `block` is null; inline math is `$TeX$` with
its offsets in `math_spans` (the `$` delimiters included); the citation `[12]` names its
bibliography entry, with the identifiers parsed from it, in `refs.cites`; and the page comes from
aligning the text with the PDF's text layer:

```json
{"unit_id": "u000088", "kind": "paragraph",
 "text": "In adjoint mode the cost of all first-order sensitivities is a small multiple of the cost of the pricer, $\\bar{x} = \\bar{y}\\,\\partial y/\\partial x$, independent of the number of inputs [12].",
 "lang": "en",
 "anchor": {"section_path": [{"number": "3", "title": "Algorithmic differentiation", "label": null},
                             {"number": "3.2", "title": "Adjoint mode", "label": null}],
            "paragraph": 2, "block": null, "html_id": "S3.SS2.p2", "part": {"index": 0, "of": 1},
            "equation_labels": [], "equation_number": null, "equation_number_method": null,
            "env": null, "float": null, "footnote": null,
            "pdf_page": 7, "pdf_page_end": 7, "page_method": "text_align", "page_confidence": null},
 "counts": {"chars": 190},
 "math_spans": [[105, 147]],
 "refs": {"cites": [{"html_id": "bib.bib12", "printed": "[12]",
                     "ids": {"doi": "10.1234/example.0012", "arxiv": null, "url": null}}],
          "labels": [], "links": []},
 "prev": "u000087", "next": "u000089"}
```

A display equation from arXiv HTML is its own `equation` unit: its `text` is the TeX, with
`html_id: "S2.E1"`, `equation_number: "(2.1)"`, `equation_number_method: "html"` and
`equation_labels: []`. A unit inside a theorem-like environment carries it in `anchor.env`, for
example `{"name": "theorem", "label": null, "number": "3.1"}`.

| field | need | meaning |
|---|---|---|
| `kind` | must | `title`, `abstract`, `heading`, `paragraph`, `equation`, `list_item`, `caption`, `table`, `algorithm`, `footnote`, `reference`, `code`. |
| `text` | must | The exact text that citations quote. No heading or context is prepended (see `section_path`). |
| `lang` | must | Detected per unit (a hint). |
| `anchor.section_path` | must | From the outermost to the innermost section: `{number, title, label}`. `number` is null for unnumbered sections. For arXiv HTML it comes from LaTeXML's section ids and printed numbers; for a PDF, from the sections GROBID found (empty with the pypdfium2 fallback). `label` is null in v1 (see below). |
| `anchor.block`, `anchor.paragraph` | must (v1: `paragraph` for arXiv HTML and from GROBID for a PDF; `block` for PDF text) | For PDF text, `block` is the block index on the page; for arXiv HTML it is null. `paragraph` is the 1-based paragraph within the innermost section: for arXiv HTML from LaTeXML's paragraph id (`S1.p1` is paragraph 1), for a PDF from GROBID (null with the pypdfium2 fallback). A display equation or list inside a paragraph shares that paragraph's number. |
| `anchor.html_id` | v1 for arXiv HTML; null for PDF text | The unit's own LaTeXML id, such as `S3.SS2.p2` (a paragraph), `S2.E1` (an equation) or `bib.bib12` (a bibliography entry); a `heading` unit carries its section's id (`S3.SS2`; agreed with devai). The parts of a split unit share it. `refs.labels` resolve through it (2026-09-26, from devai). |
| `anchor.part` | must | For a unit split to fit the cap (§3.4): its index and the number of parts. |
| `anchor.pdf_page`, `pdf_page_end`, `page_method`, `page_confidence` | must | `page_method` ∈ `pdf_native` (PDF text: GROBID's page coordinates, or the pypdfium2 page), `text_align` (arXiv HTML text aligned to the PDF's text layer), or null when no page is known. Both are v1. |
| `counts.chars` | must | Characters of the raw `text`. The service counts no tokens (A4). |
| `refs.links` | must | Link records (§2.2 `links_found`) for links found in the unit, for code links, author pages and cited papers. aiagent follows one only by its `link_id` (L11). |
| `prev`, `next` | must | Neighbours in reading order, so aiagent can build context windows. |
| `anchor.equation_number`, `equation_number_method` | v1 for arXiv HTML | The printed number, such as `"(2.1)"`, with `equation_number_method` ∈ `html` (LaTeXML's number, from arXiv HTML), `pdf_text` (read from the PDF's text, best effort; later), null (O-D11). |
| `anchor.equation_labels` | always present; empty in v1 | The authors' LaTeX `\label` keys of the unit. arXiv's HTML does not keep them (only LaTeXML's ids, such as `S2.E1`), so for an HTML text this stays empty (2026-09-26, from devai); GROBID gives none either. |
| `anchor.env` | v1 for arXiv HTML; null for PDF text | `{name, label, number}` for a unit inside a theorem-like environment (theorem, lemma, definition, proof, …), from arXiv HTML: `name` the environment, `number` its printed number (null when unnumbered), `label` null (see below); null outside one (shape agreed with devai). |
| `anchor.float` | later (must then for captions and tables) | `{kind: figure, table or algorithm; label; number}`. |
| `math_spans` | v1, must for arXiv HTML; empty for PDF text | `[start, end)` character offsets of each inline math span in `text`, `$` delimiters included (agreed with devai), so aiagent can mask math for a student, or keep it for the LLM, without parsing. For PDF text it is `[]`: its formulas are plain text, with no `$` spans. |
| `refs.cites` | v1 for arXiv HTML; null for PDF text | The bibliography entries the unit cites, in order: `{html_id, printed, ids: {doi, arxiv, url}}`, where `html_id` is the entry's LaTeXML id (the `anchor.html_id` of its `reference` unit), `printed` the citation as printed (`[12]`), and `ids` the identifiers parsed from the entry (null where not found) (shape agreed with devai). |
| `refs.labels` | v1 for arXiv HTML; null for PDF text | The targets the unit's cross-references point to, as LaTeXML ids such as `S2.E1`, since the authors' labels are not in the page. Each resolves to the unit whose `anchor.html_id` it is (2026-09-26, from devai). |

**HTML-only fields** (agreed with devai): exactly four, null for every PDF text:
`anchor.html_id`, `anchor.env`, `refs.cites` and `refs.labels`. A PDF text structured by GROBID
keeps its `section_path` and `paragraph` anchors (only the pypdfium2 fallback leaves `paragraph`
null and `section_path` empty), and its `math_spans` is `[]`. `anchor.block` is the reverse: PDF
text only, null for HTML.

Every `label` field (`section_path[].label`, `equation_labels`, `env.label`, `float.label`) means an
author's `\label` key. arXiv's HTML does not carry them and GROBID does not give them, so they are
null or empty in v1; anchors rest on printed numbers and paragraph ordinals instead.

For a **`reference` unit**, the fields include `bib_key` (later) and the identifiers parsed from the
entry (`doi`, `arxiv`, `url`), where found (in v1 from the bibliography of arXiv HTML, or by GROBID
for a PDF without HTML). From arXiv HTML it also carries the entry's LaTeXML id in
`anchor.html_id`, which `refs.cites` names. aiagent uses these to follow references into new
candidates.

### 3.4 Segmentation rules, sized for laya

| quantity | value | why |
|---|---|---|
| student `max_len` / `head_max_len` | 1024 / 256 | `laya-multilingual`, from the dataset manifest (A3) |
| worst-case state room | 764 tokens | `SequenceTokenizer.room`, for questions of at most 62 options |
| **the service's unit cap** | **1,300 characters** of raw `text` (`unit_max_chars`) | dense TeX math runs about 2.1 characters per token with this tokenizer (measured), so a capped unit stays under about 640 tokens; prose units are much smaller |
| JSON escaping in the state | +8-16 % (measured) | `{"text": …}` escapes `\`, `"` and newlines: a 581-token equation unit becomes 630 in the state, a 610-token code unit 646 |
| left for aiagent | a short breadcrumb at most | topic text never goes into a unit state (A3, A6); aiagent drops the breadcrumb or splits further when its own count says so (A4) |

1. **Units follow the structure.** A unit is one heading, one paragraph or PDF text block (or part
   of one), one display equation, one caption, one list item, one footnote, one bibliography entry,
   or one code range. Units never cross a known section boundary.
2. **Only oversized units are split.** A unit over the cap is split at sentence boundaries into
   parts that share its anchor. A split never falls inside inline math (`$…$`, or a span in
   `math_spans`) or a citation (`[12]`): the plain sentence rule `(?<=[.!?])\s+` would split
   inside `$…$`. A sentence over the cap is split at words, and a word over the cap at characters,
   the same order `distill/segment.py` uses. **Nothing is ever dropped.**
3. **No overlap.** Every character belongs to exactly one unit, so citations, counts, search hits
   and translations are unambiguous. aiagent builds overlapping context windows itself, from
   `prev`/`next`.
4. **Headings are kept twice:** as `heading` units, and as `section_path` on every unit. aiagent
   decides whether a breadcrumb fits into a state.
5. **The service counts characters only.** `unit_max_chars` is **one setting for the whole
   library**, not a per-request parameter, since a different cap gives a different `text_id`.
   `GET /v1/info` reports it. With no tokenizer in the service, a new student base model never
   changes a `text_id`.
6. **Tables** keep their cells tab-separated and their rows on separate lines, capped like
   paragraphs (split by rows).
7. **Code** (later) is split at blank-line or top-level-definition boundaries into line ranges
   under the cap (§3.6).

### 3.5 Text normalisation

- UTF-8, NFC. Inside `paragraph`, `list_item`, `caption` and `footnote` units, a run of whitespace
  becomes one space. `equation`, `table`, `algorithm` and `code` units keep their line breaks.
- **PDF (v1):**
  - the structure (sections, paragraphs, formulas as text, references with their DOI and arXiv
    ids) comes from GROBID; the pypdfium2 fallback gives the text and pages only (L12);
  - hyphenation at line ends is repaired and ligatures are expanded;
  - running headers, footers and page numbers are removed;
  - reading order is resolved for two-column layouts.
- **arXiv HTML (v1, L12):**
  - sections and paragraphs follow LaTeXML's ids (`S3.SS2`, `S3.SS2.p2`), which give
    `section_path` and `anchor.paragraph`; each unit carries its own id as `anchor.html_id`;
  - theorem-like environments (theorem, lemma, definition, proof, …) give `anchor.env` to the units
    inside them;
  - display math becomes its own `equation` unit holding the formula's TeX, with LaTeXML's printed
    equation number in the anchor (`equation_number_method: html`); the authors' labels are not in
    the page;
  - inline math is written as `$…$` holding the formula's TeX, which every formula carries
    (`annotation encoding="application/x-tex"`, verified by devai, L12 note); its offsets go in
    `math_spans`;
  - citations keep their printed label (`[12]`), with the entries cited, and the identifiers parsed
    from them, in `refs.cites` (v1);
  - cross-references keep their printed number, with their targets' LaTeXML ids in `refs.labels`
    (v1), resolvable through `anchor.html_id`.
- **Figures** are not extracted as images; the stored PDF keeps them. Captions become units.

### 3.6 Code: repos, files, excerpts (later)

v1 keeps `code_links` in candidate metadata and `refs.links` in units only. A repo is its own item
(`kind: repo`, key `gh:<owner>/<repo>`) with its own verdict. It is never cloned (L4). **For
`kind: repo`, L4 overrides L5:** an accepted repo keeps only its `repo_meta` and its excerpts, never
"full content".

1. **Repo metadata and tree.** A fetch with `want: ["repo_meta", "tree", "readme"]`:
   - pins the default branch's head to a **commit sha at fetch time**, and returns:
     - `repo_meta`: `{commit, default_branch, license: {spdx, license_file_sha256}, archived,
       fork, stars, pushed_at, languages}`;
     - `tree`: `[{path, size, blob_sha, lang, binary, dataset_like}]`, capped at 10,000 entries,
       with a `truncated` flag;
   - stages the README as text units.
2. **Files.** A fetch with `{"item_id": "itm-…", "commit": "<sha>", "files": ["src/aad/tape.h"]}`:
   - stages text files only, at most 1 MiB each and 50 per request;
   - refuses binary files and anything `dataset_like` (by extension and size) with
     `dataset_refused`;
   - stages `code` units whose anchor is `{repo_key, commit, path, line_start, line_end,
     lang_code}`, plus `permalink`, e.g.
     `https://github.com/<owner>/<repo>/blob/<sha>/src/aad/tape.h#L40-L88`.
3. **Excerpts.** aiagent chooses the line ranges and submits them in the repo's accept verdict
   (`keep.code_excerpts`, §5.1). The service then:
   - checks that the lines exist at that commit and that the text matches the file;
   - requires a non-empty `linked_from` that resolves to a unit of an accepted paper;
   - stores the **excerpt record**: `{excerpt_id, repo_key, commit, path, line_start, line_end,
     license, license_file_sha256, copyright_line, permalink, text, linked_from: [{item_id,
     text_id, unit_id}], stamp}`;
   - writes a text version with `origin: code_excerpts` and one `code` unit per excerpt (anchor,
     permalink, licence). That text is what the verdict cites and what `kinds: code` searches
     (§7.1);
   - discards everything else it staged from the repo;
   - enforces the "short" limit as configured caps (at most 80 lines per excerpt and 10 excerpts
     per repo, O3);
   - after the repo is refreshed to a newer commit (only on request, like any refresh, §5.5), keeps
     only the latest accepted commit's excerpts (devai's D14, latest version only; agreed
     2026-09-26).
4. **Level (a) checks, as data.** The service provides:
   - whether the commit exists and is reachable from a branch or tag;
   - the licence, and whether the repo is archived or a fork;
   - its last push;
   - for each paper, the unit that links to the repo.

   aiagent evaluates these.
5. **Level (b)**, sandboxed execution, is a later job kind (`execute`). The job model in §8 must not
   rule it out.

### 3.7 Non-English items

- The service stages the original-language text exactly as it would English text, with `lang` on
  every unit and `text.json.lang`. aiagent decides whether and what to translate (§4).
- Language detection is only a hint. aiagent can override it per unit in its translation
  submission.
- v1 has no translation: an accept of a text whose primary language is not English is refused (412
  `translation_incomplete`).
- **Permanent records hold English only** (L6 note, devai's D16). Staging, which is temporary, may
  hold the original language. What stays after a verdict is English: a verdict on a non-English
  item carries `metadata_en`, and `quote_en` for each evidence quote, and the service keeps the
  original only as a link with its sha256 (§5.1, §5.3). In v1, with no accepts of non-English
  texts, this concerns rejects.

### 3.8 Staging lifetime

- Staged artifacts and texts stay until a verdict, or until `staging_ttl` expires (30 days, O11).
- On expiry a maintenance job (job kind `maintenance`, agreed with devai, §8) purges them and
  emits `item.purged`, and the item goes back to `library.status: none` (2026-09-26, from devai).
  Their addresses resolve to tombstones (§1.3).
- A purge is not a rejection: the item can be fetched again.

---

## 4. Translation round trip (later)

Not needed for the acceptance test (the topic is English). aiagent can judge a non-English item
from its own translation without submitting it; only an accept needs a stored English text (a
reject needs only `metadata_en` and `quote_en`, §5.1).

1. **aiagent reads the original text's units** from staging and translates them with its LLM.
   - Unit boundaries are kept **one-to-one**, so every anchor carries over.
   - `equation`, `code` and `reference` units, and the math spans inside other units, are copied
     unchanged and marked `"method": "copied"`.
2. **aiagent submits one complete English text**, inline, as
   `POST /v1/items/{item_id}/translations` (8 MiB is enough for a paper). There are no partial
   translations, no merging, and no inbox submission.

   ```json
   {"translation_id": "tr-0a1b2c3d4e5f60718293a4b5",
    "source_text_id": "tx-1f2e3d4c5b6a79880a1b2c3d",
    "target_lang": "en",
    "translator": {"role": "translate", "kind": "llm",
                   "model": "openai/Qwen3.8-27B-MTP-devai-NVFP4::nothink@118784",
                   "program_sha256": "…", "temperature": 0.0},
    "client": {"name": "aiagent", "version": "0.6.0"},
    "metadata_en": {"title": "Adjoint methods for Bermudan options", "abstract": "…"},
    "units": [{"source_unit_id": "u000001", "text": "Adjoint methods for Bermudan options", "method": "llm"},
              {"source_unit_id": "u000044", "text": "V(t,S) = \\max(\\ldots)", "method": "copied"}],
    "section_titles": [{"label": null, "number": "3.2", "title": "Adjoint mode"}]}
   ```
3. **The service validates the submission:**
   - every source unit id exists, and each appears exactly once;
   - the output is non-empty wherever the source unit was non-empty.
4. **The service writes a new text version** with `origin: translation` and `lang: en`:
   - the unit numbering is the same (unit *n* ↔ source unit *n*), and each unit also carries
     `source_unit_id`;
   - anchors are copied from the source: section numbers, paragraph, equation numbers, page;
   - `section_path[].title` holds the English title and `title_original` the original;
   - a translated unit over the cap is **not** split, since that would break the one-to-one
     mapping. It is flagged `over_cap: true`, and aiagent's packer splits it;
   - `text.json.translation` = `{from_text_id, from_lang, translator, translated_at,
     machine: true}`;
   - `metadata_en` is stored with the item, and listings and citations use it.
5. **On accept** (L6):
   - The English translation is the stored primary text, and what gets indexed.
   - The original is kept as **a link**: `{url, sha256, lang, retrieved_at}` plus the anchors
     inside every unit. The original's bytes and its source-language text version are deleted once
     the anchors are copied (§1.3, O6).
   - Every citation and search hit built from a translation says `machine_translated: true`, names
     the translator model, and **points to the original**: its URL, page and section numbering.
   - An accept of a non-English item without a complete English text is refused (412
     `translation_incomplete`).
6. **A reject** of a non-English item cites the original's units, in staging. Its verdict carries
   `metadata_en`, and each evidence entry a `quote_en`: the service checks `quote` against the
   original unit, then keeps `quote_en` in its place, since permanent records hold English only
   (§5.1). This applies in v1 already.

---

## 5. Verdicts

### 5.1 Request

`POST /v1/verdicts` sends one verdict. `POST /v1/verdicts/batch` sends up to 500 inline and returns
one result per row; each row is idempotent on its own, and a bad row does not fail the others.

A v1 verdict (the LLM alone decides):

```json
{"verdict_id": "vd-9b8a7c6d5e4f30211203f4e5",
 "item_id": "itm-5d41402abc4b2a76b9719d91", "version": "v3",
 "text_id": "tx-7c9e6679f1a2b3c4d5e6f708",
 "stage": "fulltext",
 "decision": "accept",
 "basis": "in_scope",
 "reasons": [{"code": "applies_aad_to_greeks",
              "text": "Derives adjoint pathwise Greeks for a Libor market model Monte Carlo and reports cost ratios.",
              "evidence": [{"text_id": "tx-7c9e6679f1a2b3c4d5e6f708", "unit_id": "u000091",
                            "quote": "independent of the number of inputs"}]}],
 "scores": {"relevance": 0.91, "quality": 0.74},
 "topic": {"topic_id": "aad-comp-finance",
           "query_id": "q-3f9a0c1d2e4b5a6978877665",
           "query_text": "Automatic Adjoint Differentiation method in computational finance"},
 "profile": {"name": "research", "version": 1, "sha256": "…"},
 "deciders": [{"role": "final", "kind": "llm",
               "model": "openai/Qwen3.8-27B-MTP-devai-NVFP4::nothink@118784",
               "served": {"model": "…", "context": 118784, "mtp": true, "revision": null},
               "program_sha256": "…", "temperature": 0.0}],
 "duplicate_of": null,
 "supersedes": null, "override": null,
 "metadata_en": null,
 "client": {"name": "aiagent", "version": "0.6.0"},
 "decided_at": "2026-09-25T14:05:12Z"}
```

The stored record adds `"object": "library.verdict"`, the service's `recorded_at`, and a `match:
[start, end)` offset on each evidence entry. `GET /v1/verdicts/{verdict_id}` (v1) returns it, or
404.

| field | meaning |
|---|---|
| `stage` | `metadata` (judged from the candidate record only) or `fulltext` (judged on a staged text). |
| `decision` | `accept` or `reject`. |
| `basis` | `in_scope` (accept), or for a reject: `out_of_bounds`, `off_topic`, `low_quality`, `duplicate`. An `off_topic` reject is scoped to `topic.topic_id`, shown as `scope_topic_id` in `library.verdict` (O1). Conditions of the world or the tooling (not free, unreadable, licence) are never a basis: they are fetch or candidate status (`access.free`, `last_fetch_error`, `license`) and carry no verdict. |
| `reasons` | Codes from aiagent's controlled vocabulary, English text (at most 1,000 characters), and evidence. A `fulltext` evidence entry is `{text_id, unit_id, quote}`. A `metadata` evidence entry is `{field, quote}`, with `field` ∈ `title`, `abstract`, `categories`, `comments`. For a non-English item every entry also carries `quote_en`, aiagent's English rendering of `quote` (see `metadata_en`). |
| `metadata_en` | Null for an English item. For a non-English item (aiagent decides; language detection is a hint), aiagent's English rendering of the metadata's free-text fields (`title`, `abstract`, …). **Permanent records hold English only** (L6 note, devai's D16, 2026-09-26): the stored metadata snapshot holds `metadata_en`, and each evidence entry `quote_en`, while the original is kept only as a link with its sha256. This holds for rejects, metadata-only ones included, and in v1. |
| `scores` | An open map of name → float in [0, 1]. Stored opaquely and sortable in listings (§7.3). |
| `topic` | The query that brought the item in. It creates the first topic assessment (§5.4). |
| `profile` | Must name a registered profile version (§6), else 422 `profile_not_registered`. |
| `deciders` | Every model or student that contributed, at least one. `role` ∈ `prejudge`, `unit_filter`, `final`, `translate`, `excerpt_select`, `topic`; `kind` ∈ `student`, `llm`, `human`, `rule`. Exactly one has `role: final`; otherwise 422 `deciders_invalid`. An LLM decider carries `served`: the served model as the router's `/health` reports it (model, context, MTP), with `revision` when the router reports one. A student decider (later) carries `skill`, `predictor`, `artifact_id`, `fine_tuned_model`, `question_set_sha256`, `threshold` and `n_decisions`. |
| `keep` | Later, for a repo only: `{code_excerpts}` (§3.6). A paper has no `keep`: an accept keeps every fetched artifact of the version (L5). |
| `client` | `{name, version}` of the submitting client (A9). |

### 5.2 Rules

| decision | stage | preconditions |
|---|---|---|
| reject | metadata | the item is known (candidate cache or a resolvable key), and it has no current verdict |
| reject | fulltext | the item is staged, and it has no current verdict |
| accept | fulltext | the item is staged; its primary text is English (§4); every evidence anchor resolves; the profile is registered |
| accept | metadata | **refused**: an accepted item keeps its full content (L5) |

- **Evidence is checked.** Each evidence `unit_id` must exist in the named text, and a metadata
  `field` in the item's metadata as the source gave it (the candidate record). A `quote`, if
  given, must occur in that unit or field, compared after NFC, collapsing whitespace, and folding
  quotes and dashes (’ to ', – and — to -); otherwise the service answers 422
  `evidence_not_found`. The service stores the matched
  `[start, end)` offsets. This mechanical check catches an LLM quoting text that is not there.
- **English only in what stays.** For a non-English item, `quote` is checked against the
  original-language unit or field (staging and the candidate record may hold the original), and
  the stored evidence keeps its anchor, the match offsets and `quote_en` in place of `quote`.
  `quote_en` is stored without checking.
- **A reject's evidence keeps its quotes inline** with their anchors (for a non-English item,
  `quote_en`), so the record still stands after the texts are purged (§5.3).
- **Verdicts are immutable and append-only.** An item's current verdict is its latest verdict that
  has not been superseded.
- **Idempotency:** the same `verdict_id` with the same body returns the stored record (200). The
  same `verdict_id` with a different body gets 409 `idempotency_conflict`. aiagent's outbox
  (principle 3) makes a retry send the same bytes.
- **One current verdict per item.** A second verdict for an item that has a current verdict gets
  409 `verdict_exists`, unless it is an override (§5.5).
  - **The exception for scoped rejects (O1, answered by the owner on 2026-09-26):** an item whose
    current verdict is an `off_topic` reject for topic A may be fetched and judged for a topic
    B ≠ A without an override. The fetch item and the new verdict name `topic_id` B, and the verdict
    `supersedes` the scoped reject. The service checks only that the topics differ. devai builds
    the one-current-verdict rule with this exception (its Phase 1).

### 5.3 What happens on accept and on reject

- **Accept:**
  - All fetched artifacts of the version move from staging to the permanent store (arXiv: the PDF,
    and the arXiv HTML when the fetch staged it). The item record lists them: `{kind, path,
    sha256, size, url, retrieved_at, http_status}`.
  - The item record **must** carry `license` (§2.2: `{status, spdx, url, stated_by}`, never null
    on accept) and `versions.stored`. `status: none_stated` (arXiv papers before 2007, say) is
    accepted and means all rights reserved; a legacy licence URI is accepted as stated. The
    licence is copied from what the fetch job recorded: for arXiv, from the `arXivRaw` record it
    read; for another work, from the open-access location the copy came from (OpenAlex's
    `locations[].license` or Unpaywall's `best_oa_location.license`, named in `stated_by`), and
    `none_stated` when that location gives none. **An accept makes no external request**
    (2026-09-26, from devai).
  - The primary text is indexed by an asynchronous index job. Its id is returned, and the item
    record shows `index.status` ∈ `pending`, `indexed`, `failed`.
  - Listings by topic (§7.3) work at once; library search (§7.1) works once the item is indexed.
- **Reject:**
  - All staged artifacts and every text version of the item are deleted before the response
    returns, or are queued for deletion and reported by an `item.purged` event. Their unit
    addresses resolve to tombstones (§1.3).
  - The record keeps:
    - the metadata snapshot, including the abstract, in English: for a non-English item the
      verdict's `metadata_en`, with the original only as a link and its sha256 (§5.1);
    - every alias, including the `sha256:` of each artifact that passed the format check and
      extraction, so the same bytes are recognised under another URL (§1.5);
    - the verdict, with its evidence quotes inline.
- **Accepted items are never deleted except by a revoke** (§5.5). An item never disappears because
  of a TTL or cleanup (L5).

### 5.4 Topic assessments

Membership in the library is judged **once per item** (the one exception is a scoped `off_topic`
reject, O1). Relevance to a topic is judged **once per (item, topic)**.

- When a later query finds an item that is already accepted, aiagent does not re-judge its
  membership. It records a topic assessment:

  `POST /v1/items/{item_id}/topics`

  ```json
  {"assessment_id": "ta-4c3b2a1908f7e6d5c4b3a291", "topic_id": "aad-comp-finance",
   "query_id": "q-…", "query_text": "…", "relevant": true,
   "reasons": [{"code": "…", "text": "…",
                "evidence": [{"text_id": "tx-…", "unit_id": "u000091", "quote": "…"}]}],
   "scores": {"relevance": 0.8},
   "profile": {"name": "research", "version": 1, "sha256": "…"},
   "deciders": [{"role": "final", "kind": "llm", "model": "…", "served": {"…": "…"},
                 "program_sha256": "…", "temperature": 0.0}],
   "supersedes": null,
   "client": {"name": "aiagent", "version": "0.6.0"}, "decided_at": "…"}
  ```

- Assessments follow the verdict rules: at least one decider and exactly one `role: final` (422
  otherwise), evidence checked, `recorded_at` added.
- The verdict's `topic` creates the first assessment, with the verdict's reasons, scores and
  deciders.
- Assessments are append-only. The current assessment per (item, topic) is the latest one not
  superseded. A new assessment for the same pair needs `supersedes`, just as verdicts do.
- The listing in §7.3 uses these records, so "the reasons each item was kept" is correct for every
  topic, not only for the topic that first brought the item in.

### 5.5 Overrides, revokes and explicit re-judging

- Re-judging happens **only on explicit request** (L5). An override verdict carries both:
  - `"supersedes": "<current verdict_id>"`;
  - `"override": {"reason": "…", "requested_by": "owner", "request_ref": "cli:library rejudge"}`.
- **Fetching a rejected item** for a re-judge needs the same `override` object in the fetch item.
- **Finding what to re-judge (later, should):** `GET /v1/verdicts?basis=…&profile_version_lt=…`
  lists current verdicts by basis and profile version, so the owner can ask for a re-judge after
  the outer bound grows.
- **Revoke** (aiagent's accept → reject; arXiv's "withdrawal" is something else, §1.2) deletes the
  artifacts, as a reject does. That is explicit and recorded, so it is not silent.
  - **The 30-day trash** (O15, answered by the owner on 2026-09-26: devai's D28). A revoked item's
    artifacts first move to `trash/`, where they stay 30 days, as a safety net against a faulty
    client, and are deleted then; the item record says where they are until then. It is an
    exception to devai's D12 (rejects keep no artifacts).
- **Refresh to a new version** (arXiv `v4` after `v3` was accepted) is a fetch with
  `"version": "latest"` and `override.reason` set. It is followed by a new verdict that supersedes
  the old one. Until that verdict exists, the old version stays the stored one (O2).
- **Missing arXiv HTML.** When arXiv later adds the HTML for an item accepted with its PDF alone,
  or the HTML request used up the item's retry budget (§3.1), the HTML is fetched only on request,
  like a refresh (2026-09-26, from devai).
- **A merge conflict** (§1.5) is settled the same way: an owner-requested override that supersedes
  both verdicts.

---

## 6. Research profile

**Recommendation: aiagent owns the profile, and the service stores every version aiagent registers
as an immutable, opaque blob.**

- **Why aiagent owns it.**
  - The profile is input to thinking (L1): it holds query planning hints, the questions for the
    students, and quality and authority thresholds.
  - If the service owned it, devai would either have to understand it, which breaks the split, or
    offer a generic document editor, which is a UI that belongs to neither side.
- **Why the service stores copies.**
  - Every verdict stamps a profile version (L5). If the profile lived only in aiagent's config,
    those stamps could not be resolved from the library alone, and a lost lab home directory would
    break the audit trail.
  - A registered copy also serves as a backup: aiagent can restore the latest version from the
    service.

Contract:

- `PUT /v1/profiles/{name}/versions/{version}`, body
  `{"sha256": "<H(profile)>", "profile": {…}, "client": {…}, "created_at": "…"}`.
  - `version` is an integer, and each registered version is one higher than the last.
  - Registering the same version again is idempotent if the content is identical, and 409
    `profile_version_conflict` if it differs.
  - The service checks only that `sha256` matches the body (`H`, §1.4).
- `GET /v1/profiles/{name}/versions` and `/versions/{version|latest}`.
- The service stores the profile as a plain file in the library directory
  (`profiles/<name>/<version>.json`), readable through the lab mount, and **never interprets it**.
- A verdict or assessment that names an unregistered version is refused (422).

The content belongs to aiagent. It is listed here only for context:

- the outer bound, as areas with arXiv categories, MSC and ACM CCS codes;
- the source priorities (L3);
- the authority metric, its thresholds and the accepted `match` methods;
- the language policy;
- the code policy;
- the question sets for the students: fixed per profile version, never naming a topic (A6).

---

## 7. Library queries (for answering)

### 7.1 Search over accepted items

`POST /v1/library/search` (v1: optional for the acceptance test, step 14):

```json
{"query": {"text": "How are Bermudan swaption Greeks computed with adjoint differentiation?", "mode": "hybrid"},
 "filters": {"topics": ["aad-comp-finance"], "item_ids": null, "kinds": ["paragraph", "equation", "caption", "code"],
             "date_from": null, "authors": null, "machine_translated": null},
 "k": 20}
```

- `mode` ∈ `keyword`, `vector`, `hybrid`. The query must be English: aiagent translates a
  non-English question first (L6).
- The service embeds the query with the index's own model and fuses the two result lists
  mechanically (for example with reciprocal-rank fusion). v1 returns the fused score only.
- **Later (should):** each component's score and rank, so aiagent can re-rank with its students or
  the LLM, or fuse the lists itself; `per_item_max`; and `context: {before, after}` units per hit.
- **Hits are units, not index chunks.** If the embedding model's input limit is smaller than a unit
  (many English models stop at 512 tokens; devai's bge-small-en-v1.5 does, O-D5), how the index
  chunks is internal to the service. A hit still reports the unit, and optionally the character
  range inside it.

A hit (v1):

```json
{"item_id": "itm-5d41402abc4b2a76b9719d91", "text_id": "tx-7c9e6679f1a2b3c4d5e6f708", "unit_id": "u000091",
 "kind": "paragraph", "text": "…", "anchor": {"section_path": [], "block": 4, "pdf_page": 7},
 "scores": {"fused": 0.031},
 "citation": {"…": "see 7.2"}}
```

Each response also carries
`index: {keyword_engine, embedding_model, embedding_sha256, index_version, built_at}`, so an answer
can record what it was retrieved from.

### 7.2 Citation object

```json
{"item_id": "itm-5d41402abc4b2a76b9719d91",
 "title": "…", "authors": ["Ada Example", "Ben Sample"], "year": 2012,
 "ids": {"arxiv": "2403.01234v3", "doi": null},
 "permalink": "https://arxiv.org/abs/2403.01234v3",
 "locator": {"section": "3.2", "section_title": "Adjoint mode", "paragraph": 2,
             "equation": null, "equation_label": null, "page": 7, "page_method": "pdf_native"},
 "machine_translated": false, "translator": null,
 "original": null,
 "license": {"status": "stated", "spdx": "CC-BY-4.0", "url": "http://creativecommons.org/licenses/by/4.0/", "stated_by": "arxiv"},
 "address": "itm-5d41402abc4b2a76b9719d91/tx-7c9e6679f1a2b3c4d5e6f708/u000091"}
```

- v1 locators give the page, the section and the paragraph: from LaTeXML's ids for an arXiv HTML
  text (its page by `text_align`), and where GROBID found them for a PDF (L7: pages alone are
  acceptable). `equation` is the printed number (`"(2.1)"`) from arXiv HTML, in v1;
  `equation_label` stays null, since arXiv's HTML drops the authors' labels (L12 note).
- For a translated item, `original` = `{url, lang, sha256}`. The `locator` holds the **original's**
  section numbering and page (L6).
- `permalink` names the exact version. For code it is the commit permalink.
- aiagent formats the citation. The service only supplies the facts.

### 7.3 Items by topic (the acceptance test's listing)

`GET /v1/library/topics/{topic_id}/items?decision=accept&sort=score:relevance&cursor=…`

This returns, per item:
- `{item_id, kind, title, authors, year, ids, permalink, license, upstream_status}`;
- the membership verdict (`decision`, `basis`, `reasons`, `scores`, `deciders`, `profile`,
  `decided_at`, `recorded_at`);
- the current topic assessment for this topic (its `reasons`, `scores` and `deciders`);
- `stored_version` and `index.status`.

`sort=score:<name>` sorts by that score of the **topic assessment** for this topic.

In addition:
- `decision=reject` lists what was turned down for the topic, with the reasons ("known, not kept").
- Titles, reasons and quotes are English: for a non-English item they come from `metadata_en` and
  `quote_en` (§5.1).
- `GET /v1/library/topics` (v1) lists the topic ids with counts.

### 7.4 Item record and resolution

- `GET /v1/items/{item_id}` returns the whole record:
  - key, aliases and kind; `redirect_to` for a merged item, `merged_from` on the survivor, and
    `verdict_conflict`;
  - the metadata snapshot (English, §5.1), versions (the full list once a fetch has read
    `arXivRaw`) and `upstream_status`;
  - `license` (on accepted items, §5.3);
  - artifacts (path, sha256, size, url, retrieved_at);
  - texts (`text_id`, origin, role, lang, `staging_path` or store path, extractor, counts), and
    the tombstones of purged ones (v1);
  - the verdict chain, the topic assessments and the excerpts;
  - `index.status` and `last_fetch_error`.
- `GET /v1/items/resolve?key=doi:10.1234/example.5678` returns `{item_id, status}` for any alias,
  following redirects, or 404.
- `POST /v1/items/match` (later, §1.5).

---

## 8. Jobs and progress

All long work is a job: `search`, `enrich`, `fetch`, `index`, `maintenance` (the service's own
staging-TTL purges, §3.8), and later `execute`. Several jobs run at once. The queues, one per rate
group (§8.3), are shared by all jobs and are fair between them (first come, first served in v1,
O-D9; `interactive` and `background` priority classes are later).

### 8.1 Job object

```json
{"object": "library.job", "id": "lbjob-1a2b3c4d5e6f708192a3b4c5", "kind": "fetch",
 "status": "waiting",
 "created_at": "…", "started_at": "…", "finished_at": null, "updated_at": "2026-09-25T14:02:59Z",
 "server_time": "2026-09-25T14:03:00Z",
 "query_id": "q-3f9a0c1d2e4b5a6978877665",
 "progress": {"phase": "downloading",
              "items": {"total": 31, "done": 12, "staged": 10, "skipped": 1, "failed": 1},
              "bytes": {"done": 38400000, "total_est": 99000000},
              "current": [{"item_id": "itm-…", "step": "extract_pdf", "started_at": "…", "attempt": 1}]},
 "waits": [{"source": "arxiv", "host": "arxiv.org", "reason": "crawl_delay",
            "until": "2026-09-25T14:03:02.400Z", "queue_position": 2}],
 "eta": {"at": "2026-09-25T14:12:30Z", "seconds": 570, "basis": "schedule", "reason": null},
 "eta_inputs": {"per_group": {"arxiv-files": {"requests_left": 38, "min_interval_s": 15.0,
                                              "next_slot_at": "2026-09-25T14:03:02.400Z"},
                              "arxiv-legacy-api": {"requests_left": 19, "min_interval_s": 3.0,
                                                   "next_slot_at": "2026-09-25T14:03:01.000Z"}}},
 "upstream": {"requests": 36, "bytes": 38400000, "by_source": {"arxiv": {"requests": 36, "bytes": 38400000}}},
 "results": {"count": 12, "path": "jobs/lbjob-1a2b3c4d5e6f708192a3b4c5/results.jsonl"},
 "error": null,
 "devai": {"run_dir": "jobs/lbjob-1a2b3c4d5e6f708192a3b4c5"}}
```

- **`status`** ∈ `queued`, `running`, `waiting`, `succeeded`, `failed`, `cancelled`.
  - `waiting` means no work can go on until the earliest `waits[].until`.
  - `succeeded` means the job ran to the end. Failures of single items are counted in
    `progress.items.failed` and listed in the results; they do not fail the job.
  - `failed` is for the job itself: an internal error, or a restart with no resume (§8.5).
- **`waits[].reason`:**

  | reason | meaning | `until` |
  |---|---|---|
  | `rate_limit` | the service's own schedule, per rate group (§8.3) | the next slot |
  | `crawl_delay` | robots.txt | the next slot |
  | `retry_after` | a single upstream 429 or 503 with `Retry-After`, after a failed attempt that counts against the item's retry budget (§3.1) | when the upstream's time has passed |
  | `backoff` | upstream errors, a 429 or 503 without `Retry-After` included, or a pipelock failure (`egress_unavailable`), exponential backoff after a failed attempt that counts against the item's retry budget (§3.1); or a provider's overload pause (at least 15 min, §9.1), which does not count against it | the next attempt, or the pause's end |
  | `quota` | a daily API budget is used up (OpenAlex's); does not count against the retry budget | the budget's reset time |
  | `concurrency` | other jobs are ahead in the queue | – (`queue_position` says how far) |

  None of these is a stop (§9.1): each ends by itself. Only failed attempts count against an
  item's retry budget, and its 30 minutes exclude the time spent in these waits (§3.1). A stopped
  source is not a wait: its items fail promptly with the stop in `error.details.cause`, and the ETA
  says `source_down` (§8.3).

- **Result rows**, in `results.jsonl` and `GET /v1/jobs/{id}/results?cursor=…`, one per item:

  ```json
  {"item_id": "itm-…", "status": "staged", "version": "v3",
   "text_id": "tx-…", "origin": "html", "staging_path": "staging/…/tx-…", "lang": "en", "units": 298,
   "fallback_text": {"text_id": "tx-…", "origin": "pdf", "staging_path": "staging/…/tx-…", "units": 312},
   "artifacts": [{"kind": "pdf", "sha256": "…", "size": 1234567}, {"kind": "html", "sha256": "…", "size": 456789}],
   "upstream": {"requests": 3, "bytes": 1693404}, "error": null}
  ```

  - `text_id`, `origin`, `staging_path`, `lang` and `units` describe the primary text;
    `fallback_text` (the name agreed with devai) is the PDF's text when the primary text is arXiv
    HTML, and null otherwise (§3.2). The three requests are the PDF, the HTML and the `arXivRaw`
    record (§3.1).
  - A failed row has `status: failed` and `error {code, message, details}`, with a stop's cause
    in `details.cause` (§3.1).

  - For search jobs a row is a candidate record (§2.2).
  - The results carry, per source, `upstream_total`, `returned`, `next_cursor` and `error` (§2.1).

### 8.2 Events

Each job writes `jobs/<id>/events.jsonl` and serves the same events over HTTP.

```json
{"seq": 57, "ts": "2026-09-25T14:02:58.120Z", "job_id": "lbjob-…", "type": "item.staged",
 "item_id": "itm-…", "data": {"text_id": "tx-…", "staging_path": "staging/…/tx-…", "units": 298, "lang": "en",
                             "fallback_text_id": "tx-…"}}
```

| type | data |
|---|---|
| `job.created`, `job.started` | the request echo |
| `phase.changed` | `phase` (searching, downloading, extracting, indexing, …) |
| `item.queued`, `item.started` | `source`, `step`, `attempt` |
| `wait.started`, `wait.ended` | `source`, `host`, `reason`, `until`, `queue_position` |
| `item.downloaded` | `kind`, `bytes`, `sha256`, `http_status` |
| `item.extracted` | `text_id`, `units`, `lang`, `warnings` |
| `item.staged` | `text_id`, `staging_path`, `units`, `lang` (the primary text), `fallback_text_id` (or null) |
| `item.skipped` | `status` (already_stored, already_staged, rejected_skipped, duplicate_of_rejected, duplicate_of_stored, withdrawn_upstream, not_fetchable with its `reason`) |
| `item.failed` | `error {code, message, details}` (`egress_blocked` and `egress_unavailable` included; `details.cause` for a stopped source, §3.1) |
| `item.merged` | `into` (the surviving item id), `alias`, `verdict_conflict` |
| `item.purged` | `reason` (rejected, staging_ttl, revoked) |
| `job.progress` | the job's `progress`, `eta` and `waits` (heartbeat) |
| `job.succeeded`, `job.failed`, `job.cancelled` | the final counts, or `error` |

`seq` increases by one per event within a job, starting at 1. `ts` is the service's clock.

### 8.3 ETA

- **The service must compute an ETA, because it has the schedule:**
  - the remaining requests per rate group times that group's interval;
  - the work queued ahead;
  - measured transfer and extraction times.

  `basis` ∈ `schedule`, `throughput`, `unknown`.
- **A first estimate must come within 2 s** of the job starting, even a rough one, and it is
  updated at every event. If none can be given, the job says why within the same 2 s.
- `eta` may be null only with `basis: unknown` and a `reason`: `source_down` (a source the job
  needs is stopped, or stopped while the job ran, §9.1, and `/health` shows it `down`, or `limited`
  when the stop covers only part of its traffic, with its cause), `crawl_delay_unknown`, or
  `no_measurements`. aiagent then shows "ETA unknown: <reason>" with the elapsed time and the
  items done. With `source_down`, the job's remaining items for that source end promptly as
  `failed`, `upstream_unavailable` with the stop in `error.details.cause` (§3.1). An
  overload pause or a used-up daily budget is not `source_down`: it is a wait with an `until`
  (§8.1), and the ETA counts it.
- **`eta_inputs.per_group` must be exposed** (`requests_left`, `min_interval_s`, `next_slot_at`),
  because aiagent's end-to-end ETA also includes its own judging. It is reported **per rate group**
  (2026-09-26, from devai), since one source can span groups with different intervals, and
  `/v1/info` reports the groups (§9.1):
  - `arxiv-legacy-api`: `export.arxiv.org` and `oaipmh.arxiv.org`, one connection, 3 s, shared by
    searches and the fetch job's `arXivRaw` requests;
  - `arxiv-files`: `arxiv.org`, the 15 s crawl delay (O-D7);
  - one group per other host.

  So 25 papers with PDF and HTML mean 50 files at 15 s, about 12-13 min of waiting, while their 25
  `arXivRaw` requests (about 75 s at 3 s) run in the other group at the same time; the ETA shows
  it. ETAs assume 15 s per `arxiv.org` file: devai checks the anonymous arXiv PDF mirror in its
  Phase 1 and uses it only if its terms and robots.txt allow automated downloads (D27, O-D7).
  aiagent overlaps judging with downloading, starting on each `item.staged`. Its judging estimate
  is its measured LLM time per item, plus, once students are used, `units × questions × about
  80 ms`.
- `server_time` lets aiagent turn `until` into seconds without depending on clock skew.

### 8.4 Delivery: volume and polling

1. **The volume is the source of truth**, as with the trainer.
   - `jobs/<id>/job.json` is rewritten atomically (write, then rename) at every event, at most once
     per second.
   - `events.jsonl` is append-only, with whole lines.
   - Status reads therefore never need the service to be up.
2. **Polling**, which aiagent implements:
   - every job-creating POST returns 202 with the job at once; nothing blocks the client;
   - `GET /v1/jobs/{id}` every 1-2 s;
   - `GET /v1/jobs/{id}/events?after=<seq>&limit=500`, which returns events in ascending `seq`,
     as `{"object": "list", "data": [...], "has_more": …}`.
3. **SSE** (later, should): `GET /v1/jobs/{id}/events/stream`, `text/event-stream` with
   `id: <seq>`, resumable from `Last-Event-ID`, with a keep-alive comment every 15 s. Nothing
   depends on it.

### 8.5 Cancel and restart

- `POST /v1/jobs/{id}/cancel` (an empty body is fine):
  - aborts in-flight downloads and removes their partial files;
  - leaves already staged items staged;
  - sets `status: cancelled` with per-item states.
- aiagent enforces its own deadlines with this call.
- **After a service restart**, a job resumes (O-D9, answered): item steps are idempotent, and its
  unfinished items are queued again from `jobs/<id>/`. Should a job still end `failed` with
  `error.code: service_restarted`, aiagent resubmits with the same `Idempotency-Key` and gets the
  resumed or the new job. Items already staged are not downloaded again (§3.1).
- Job records are kept for at least 30 days.

### 8.6 Granularity (recommendations)

| what | granularity |
|---|---|
| item state changes | every one, as an event |
| byte counts | only in `job.progress`, never per chunk |
| heartbeat (`job.progress`, `updated_at`) | at most 1/s; **at least every 5 s** while `running` or `waiting` |
| rate-limit waits | an event when a wait starts (with `until`) and when it ends, plus the current `waits` in the job object |
| staleness | aiagent shows "no report from the library service for 30 s" when `updated_at` is 30 s old and the job is not terminal |

**aiagent's own LLM calls are never silent either.** They can block for minutes: a cold start of
about 2 minutes, a 503 `gpu_held_by_job` while a laya training job holds the GPU (up to 900 s), and
a request timeout of 900 s. While a call is in flight, aiagent's display ticks at least every 5 s
with the elapsed time and the router's state, from the router's `/health` (loading, or held by a
training job).

**What the owner sees** (an aiagent-side sketch, refreshed at most once per second):

```
fetch   12/31 staged · 38 MB · arxiv.org crawl delay, next slot in 2 s (38 requests left at 15 s)
judge   9 judged · 4 accepted · 5 rejected · LLM: 2 in flight, 41 s
ETA     10 min 30 s (downloads 9 min 30 s, judging overlaps)
```

---

## 9. Transport

### 9.1 Connection

- **HTTP/1.1 JSON on devai-net** at a configured base URL (`library_api_base`). devai's answer
  (O-D1): a plain compose container `devai-library` on devai-net and devai-lab-egress, with no
  host port. aiagent's default is `http://devai-library:8080`, reached directly (the host is in
  `NO_PROXY`).
- **No proxy.** httpx with `trust_env=False`.
- **Auth:** none on devai-net for now. aiagent sends `Authorization: Bearer <library_api_key>` when
  that setting is non-empty, so adding auth later is a configuration change. This token is internal
  and has nothing to do with the source credentials the service holds.
- **Timeouts:** every endpoint, including the job-creating POSTs (202), **should** answer within
  2 s (p95); aiagent's client timeout is 30 s, like the trainer client's.
- `GET /health` (at the root). `status` ∈ `ok`, `degraded` (`degraded` whenever any stop
  holds); `contact.configured` says whether the contact address is set (L16); each source has
  `status` ∈ `ok`, `limited`, `down`, and a `cause` when it is `limited` or `down` (stops, below).
  With the contact address unset:

  ```json
  {"status": "degraded",
   "contact": {"configured": false},
   "sources": {"arxiv": {"status": "down", "cause": "contact_not_configured"},
               "openalex": {"status": "down", "cause": "contact_not_configured"}},
   "index": {"status": "ok"}}
  ```
- `GET /v1/info` returns:
  - `api_version` and `service_version`;
  - the enabled `sources` with their `status` and `cause` (as in `/health`: `ok`, `limited` or
    `down`), what `native` query
    syntax they take, and which filters are native;
  - `per_group`: the **rate groups** the service schedules by (2026-09-26, from devai), each
    `{hosts, min_interval_s, max_connections}`. A group spans the hosts that share one budget, so
    one source can span groups with different intervals:

    ```json
    {"per_group": {"arxiv-legacy-api": {"hosts": ["export.arxiv.org", "oaipmh.arxiv.org"],
                                        "min_interval_s": 3.0, "max_connections": 1},
                   "arxiv-files": {"hosts": ["arxiv.org"], "min_interval_s": 15.0, "max_connections": 1}}}
    ```

    Every other host is a group of its own. Jobs report the same groups in `eta_inputs.per_group`
    (§8.3).
  - `segmenting` (`unit_max_chars`, §3.4);
  - `embedding {model, revision, sha256, dim, max_tokens}` and `keyword {engine, analyzer}`;
  - `limits` (§9.4), including the retry budget (§3.1);
  - the measured index rebuild times (O-D5), for information.
- **Stops** (2026-09-26, from devai). Three conditions stop a source until someone acts, each
  with its cause code:

  | `cause` | condition | what it stops | source status | cleared by |
  |---|---|---|---|---|
  | `contact_not_configured` | the contact address is unset (L16) | every external source | `down` | configuring the address |
  | `breaker_open` | a denial: an origin 403 (below) | the provider; for arXiv, all arXiv traffic | `down` | an operator, explicitly |
  | `arxiv_fulltext_cap` | arXiv full-text downloads reached a configured count below 1,000, after which arXiv asks to be contacted (each paper counted once, whatever its formats; O-D7) | arXiv's PDF and HTML downloads only, not its search or OAI-PMH (`arXivRaw`) | `limited` | the owner recording that arXiv was contacted |

  Each stop shows the same way:
  - `/health`: `status: degraded`, and the source with its `cause`: `down` when all of its
    traffic is stopped (`contact_not_configured`, `breaker_open`), `limited` when only part of it
    is (`arxiv_fulltext_cap`: arXiv's downloads stop, its searches go on) (agreed with devai). For
    the contact, also `contact.configured: false`;
  - `/v1/info`: the source `down` or `limited` with its `cause`, as in `/health`;
  - a job-creating request whose **every** named source is stopped (for what the request needs)
    answers 503 `unavailable` with the cause in `error.details.cause` (§9.3), so aiagent fails fast
    (§10 step 0);
  - a request naming a stopped and a working source is accepted (202): in search results the
    stopped source gets a per-source `error` (§2.1), and fetch items that need it end promptly as
    `failed`, `upstream_unavailable`, still fetchable (§3.1); in both, the cause is at
    `error.details.cause`, as in the 503;
  - a source that stops while a job runs gives `eta: {basis: unknown, reason: source_down}`, and
    the job's remaining items for it end the same way (§8.3);
  - `arxiv_fulltext_cap` fails promptly too, with no `quota` wait, since only the owner can lift
    it.
- **Upstream 429s, overload and denial** (2026-09-26, from devai). They are keyed on the
  provider:
  - a single 429 or 503 is a failed attempt within the item's retry budget (§3.1), followed by a
    wait (`retry_after` when the answer has a `Retry-After` header, otherwise `backoff`), and
    never opens a breaker: arXiv answers 429 even at compliant rates;
  - **overload**, sustained origin 429s or 5xx despite waiting (a configured count per window, for
    example 5 in 10 minutes), pauses the provider for at least 15 minutes; the pause clears by
    itself, and jobs wait for it (`backoff`), outside the items' retry budgets;
  - **denial**, an origin 403, opens the breaker (`breaker_open`, above), which only an operator
    clears; for arXiv it stops all arXiv traffic, since arXiv treats continued requests after a
    403 as an attack;
  - a used-up daily budget (OpenAlex's) is not a stop either: jobs wait until its reset time
    (`quota`), outside the items' retry budgets;
  - an answer of the service's own egress proxy is never an origin answer, so it cannot open a
    breaker: a DLP block is `egress_blocked`, final for that request; a pipelock 5xx or connect
    failure is `egress_unavailable`, a wait retried within the item's retry budget (§3.1, §9.3).

### 9.2 Conventions

- UTF-8 JSON. Timestamps are RFC 3339 in UTC with `Z`. Durations are float seconds. Sizes are
  integer bytes.
- Every object has `object` (`library.job`, `library.item`, `library.verdict`, `list`, …).
- Ids use one style, the trainer's: `<prefix>-<hex>` (§1.4).
- **Requests:** an unknown field is refused (400 `unknown_field`), so a typo fails fast.
  **Responses:** new fields may appear at any time, and aiagent ignores them (pydantic
  `extra="ignore"`, as in `FineTuningJob`).
- `/v1` in the path. Every file format on the volume has its own `schema_version`. A breaking change
  is a new version, and aiagent declares which versions it reads, as it does with `format_version`
  for the trainer.
- An `Idempotency-Key` header goes on every POST that creates a job. The service keeps it for at
  least 24 h. Records with client-minted ids (verdicts, assessments, translations) use the id
  itself, and aiagent's outbox replays them byte for byte (principle 3).

### 9.3 Errors

The error body is OpenAI-style:

```json
{"error": {"type": "invalid_request_error", "code": "verdict_exists",
           "message": "itm-5d41402abc4b2a76b9719d91 already has verdict vd-… (reject, low_quality); send supersedes + override to re-judge",
           "param": "item_id", "retry_after": null, "details": {"current_verdict_id": "vd-…"}}}
```

| HTTP | `error.code` | when | aiagent's reaction |
|---|---|---|---|
| 400 | `invalid_request`, `unknown_field` | malformed request | a bug: fail |
| 404 | `item_not_found`, `text_not_found`, `job_not_found`, `profile_not_found` | unknown id | fail, naming the id |
| 409 | `verdict_exists` | a current verdict exists and there is no override | if accepted: record a topic assessment for its own topic (§5.4); if rejected: skip the item |
| 409 | `alias_conflict` | the item was merged with a conflicting verdict (§1.5) | show the conflict to the owner |
| 409 | `idempotency_conflict` | same key, different body | a bug: fail |
| 409 | `profile_version_conflict` | a version was re-registered with other content | fail |
| 410 | `text_purged` | a unit address of a purged text; `details` is the tombstone | use the verdict's inline quotes |
| 412 | `not_staged`, `translation_incomplete` | a precondition for the verdict is missing | fetch or translate first |
| 413 | `payload_too_large` | over the inline limit | split the batch (later: resend via `inbox/`) |
| 422 | `evidence_not_found`, `profile_not_registered`, `deciders_invalid`, `excerpt_mismatch` (later) | integrity check failed | fix the record; for evidence, re-ask the LLM |
| 429 | `rate_limited` | **the service's own** limit on clients, with `Retry-After` | wait and retry |
| 503 | `unavailable` | starting up, index rebuild, maintenance; with `Retry-After` | wait and retry, showing the reason |
| 503 | `unavailable`, with `details.cause` ∈ `contact_not_configured`, `breaker_open`, `arxiv_fulltext_cap` | a job-creating request whose every named source is stopped (§9.1); one that also names a working source gets 202, with the stop reported per source or per item | stop, and show the owner the cause; retry only once `/health` shows the source `ok` again |
| 500 | `internal_error` | anything else | fail with the message |

**Upstream rate limits never reach aiagent as 429.** In a job they are `waits`; on a synchronous
path they are 503 with `Retry-After`. A fetch of a rejected item is never an error: it is the
per-item outcome `rejected_skipped`.

**The cause of a stop is always at `error.details.cause`** (agreed with devai): in the 503 above,
in a per-source search `error` (§2.1) and in a per-item failure (§3.1). No `error` carries `cause`
directly.

**The service's own egress proxy is never the source's answer** (agreed with devai). Its answers
are per-item or per-search errors (§2.1, §3.1), never HTTP errors, and never open a breaker:
- `egress_blocked`: a pipelock DLP block. It is final for that request: aiagent reports it as the
  service's block, and does not resend the same query or item unchanged;
- `egress_unavailable`: a pipelock 5xx or connect failure. It is a wait, retried within the item's
  retry budget; aiagent sees it only once the budget is used up, and may fetch the item again
  later.

### 9.4 Pagination and limits

- Every list uses cursors:
  `{"object": "list", "data": [...], "has_more": true, "next_cursor": "…"}` with `?cursor=` and
  `?limit=`. A cursor is opaque and stays valid for at least 1 h.
- **Proposed limits** (reported by `/v1/info`):

  | limit | value |
  |---|---|
  | request body | 8 MiB |
  | page | 500 units, 1,000 candidates or events, or 4 MiB |
  | verdicts per batch | 500 |
  | fetch retry budget | 5 failed attempts or 30 min per item, wait time excluded (§3.1) |
  | files per repo fetch (later) | 50 |
  | size per code file (later) | 1 MiB |

- **Anything larger goes through the volume:** `staging/` and `jobs/` for service → aiagent, and
  later `inbox/` for aiagent → service.

### 9.5 Shared volume (what aiagent reads and writes)

| path under the library root | lab (aiagent) | written by |
|---|---|---|
| `staging/…/<text_id>/` | read | the service. Exactly `text.json`, `units.jsonl` and `SHA256SUMS`; no symlinks, no subdirectories. |
| store paths returned by the API (artifacts, texts) | read | the service. Artifacts are the PDF and, for arXiv, the arXiv HTML (L12). |
| record files at paths the API returns (item records, aliases, verdicts, topic assessments, translation records, excerpts) | read | the service, from aiagent's submissions. Append-only; the source of truth (principle 9). |
| `jobs/<job_id>/{job.json, events.jsonl, results.jsonl}` | read | the service |
| `profiles/<name>/<version>.json` | read | the service, from aiagent's registration |
| `inbox/<kind>-<sha12>/` (later) | **read-write** | aiagent. Each submission holds a `manifest.json` (`schema_version`, `kind`, `files {name: {sha256, rows}}`), its data files and `SHA256SUMS`. The id must name its manifest, exactly as for `ds-` datasets. The service imports it and never modifies `inbox/`; aiagent deletes a submission once the import is acknowledged. |

- The library root is exactly one configured directory, `LIBRARY_DIR` (O-D2). Secrets are never in
  it: devai renders them from sops/age to a tmpfs inside the service only. The derived indexes
  (`index/`) and the service's own state (`state/`) live under the library root, but aiagent never
  depends on their files. Any database the service keeps is a
  derived cache, never the store of record.
- Files are world-readable (0444), and directories 0555 once final, as on `/laya`.

### 9.6 Endpoints aiagent needs

| method and path | need | v1 | section |
|---|---|---|---|
| `GET /health`, `GET /v1/info` | must | yes | 9.1 |
| `PUT /v1/profiles/{name}/versions/{version}`, `GET …/versions[/{version\|latest}]` | must | yes | 6 |
| `POST /v1/searches` | must | yes | 2.1 |
| `POST /v1/fetches` | must | yes | 3.1 |
| `GET /v1/jobs[/{id}]`, `…/events?after=`, `…/results?cursor=`, `POST …/cancel` | must | yes | 8 |
| `GET /v1/items/{item_id}` (with tombstones), `GET /v1/items/resolve?key=` | must | yes | 7.4 |
| `GET /v1/items/{item_id}/texts/{text_id}/units` (410 `text_purged` with the tombstone) | must (§10 check 3) | yes | 3.2 |
| `POST /v1/verdicts`, `POST /v1/verdicts/batch`, `GET /v1/verdicts/{id}` | must | yes | 5 |
| `POST /v1/items/{item_id}/topics` | must | yes | 5.4 |
| `GET /v1/library/topics`, `GET /v1/library/topics/{topic_id}/items` | must | yes | 7.3 |
| `POST /v1/library/search` | must | optional (fused score only) | 7.1 |
| `POST /v1/items/{item_id}/translations` | must | no | 4 |
| `POST /v1/enrichments` | should | yes (`author_metrics`, `citations`; devai Phase 2) | 2.3 |
| `GET /v1/candidates/{item_id}` | should | yes (devai Phase 2) | 2.3 |
| `POST /v1/items/match` | should | no | 1.5 |
| `GET /v1/verdicts?basis=…&profile_version_lt=…` | should | no | 5.5 |
| `GET /v1/jobs/{id}/events/stream` | should | no | 8.4 |

---

## 10. The acceptance test, as API calls

**Topic:** "Automatic Adjoint Differentiation method in computational finance", with topic id
`aad-comp-finance` and profile `research` v1. The aiagent commands are a sketch (the CLI does not
exist yet). The counts are illustrative. v1 judges with the LLM alone; its times are not measured
yet (A5), so the owner sees an ETA computed from measured per-item times as the run proceeds.

| # | call | aiagent | the owner sees |
|---|---|---|---|
| 0 | `GET /health`, `GET /v1/info` | Checks the API version, that arXiv and OpenAlex are enabled and `ok` (neither `down` nor `limited`), and `unit_max_chars`. A stopped source ends the run here, naming the cause (§9.1): with the contact address unset, `/health` shows `contact.configured: false` and every source `down` with `contact_not_configured`; `breaker_open` (`down`) and `arxiv_fulltext_cap` (arXiv `limited`) end it too (the test needs arXiv's full text). An overload pause or a used-up OpenAlex budget does not: the jobs wait, and the ETA shows it. | `library ok · arxiv ok · openalex ok`, or `library degraded · arxiv down: contact_not_configured · …`, or `library degraded · arxiv limited: arxiv_fulltext_cap · openalex ok` |
| 1 | `PUT /v1/profiles/research/versions/1` | Registers the profile (idempotent). | – |
| 2 | – | **Plans with the LLM:** mints the topic id and about 6 queries, each with a `query_id`, and plans the OpenAlex side explicitly, since a non-arXiv work is a pass condition (check 7). **arXiv:** native syntax over `q-fin.CP`, `q-fin.PR`, `q-fin.RM`, `math.NA` and `cs.MS`, with phrases such as "adjoint algorithmic differentiation", "adjoint method" + "Greeks", "AAD" + "Monte Carlo" and "algorithmic differentiation" + "XVA". **OpenAlex:** every query also carries a native OpenAlex filter string (`query.native.openalex`) limited to open-access works, to find the literature outside arXiv (journals, repositories; L15), and at least one query is OpenAlex-only (`sources: ["openalex"]`), aimed at the journal literature on AAD in finance. The OpenAlex queries stay within its daily budget (L15). **Enrichment:** the step-4 shortlist, arXiv and non-arXiv works alike, is enriched from OpenAlex in step 6. | `plan · 6 queries · arxiv + openalex (1 openalex-only) · enrich the shortlist` |
| 3 | `POST /v1/searches` × 6 (202 each, `sources: ["arxiv", "openalex"]`, or `["openalex"]` for the OpenAlex-only query), then poll and `GET /v1/jobs/{id}/results` | Collects about 150 unique candidates, checking `upstream_total` against `returned` and each source's `error` (§2.1); an OpenAlex record with the same DOI or arXiv id as an arXiv record merges into one candidate (§2.1). The ones with `library.status` ≠ none are set aside; candidates with `access.free: false` or `access.fetchable: false` (SSRN and other link-only works, L14, unless OpenAlex or Unpaywall found another open location) are skipped without a verdict, and kept for the report as "known, not held" (step 13). | `search 4/6 · arxiv + openalex · arXiv rate limit, next slot 2 s · 131 candidates · ETA 15 s` |
| 4 | – | **Pre-judges on metadata.** v1: the LLM over each candidate record, 4 in flight. Later, with a student first: 150 × 50-110 ms plus a 1 s load, about 10-20 s. About 30 are shortlisted. | `prejudge 42/150 · LLM · ETA 1 min 10 s` |
| 5 | `POST /v1/verdicts/batch` | Records the metadata rejects (`stage: metadata`, basis `out_of_bounds`, `off_topic` or `low_quality`), each with reasons and metadata evidence; for a non-English candidate with `metadata_en` and `quote_en` (§5.1). | – |
| 6 | `POST /v1/enrichments` (the whole shortlist, arXiv and non-arXiv works alike, `author_metrics` + `citations`, as planned in step 2) | Gets the authors' h-index and the `cited_by` counts from OpenAlex (advisory, with provenance; L15), then runs an LLM pre-judge on the enriched records. A non-arXiv work that stays on the shortlist goes to step 7 by the `link_id` of its open-access location when that is fetchable; otherwise it is reported as "known, not held", with its link and reason (check 7). Code links come later. | `enrich 30 · openalex · 4 non-arXiv: 2 fetchable, 2 known, not held` |
| 7 | `POST /v1/fetches` (about 25 items: arXiv items with `pdf` + `html`, works found through OpenAlex with `pdf` by the `link_id` of an open-access location), then poll the job and `events?after=` | Shows progress and the ETA (§8.6). At arxiv.org's Crawl-delay of 15 s, 25 papers with PDF and HTML mean about 12-13 min of waiting (O-D7); their `arXivRaw` requests run meanwhile in the other rate group (§8.3). | `fetch 12/25 · 38 MB · arxiv.org crawl delay, next slot 9 s · ETA 6 min 30 s` |
| 8 | on each `item.staged`: read `staging_path/units.jsonl` | **Judges the full text.** v1: the LLM reads the whole staged text (about 118k context, A5) and gives a verdict with evidence anchors. Later, a unit-filter student with a fixed profile question (never the topic, A6) screens units first: about 25 papers × 300 units × 80 ms, about 10 min of student time. | `judge 9/25 · 4 accepted · 5 rejected · LLM 41 s` |
| 9 | later: `POST /v1/items/{id}/translations` | Not needed: the topic is English. A non-English item stays staged and is reported. | – |
| 10 | later: repo fetches (§3.6) | Not in v1: code links stay in the candidate metadata. | – |
| 11 | `POST /v1/verdicts` per item | Sends accept (full content kept, indexed) or reject (artifacts and texts purged). | `done · 11 accepted · 14 rejected (full text) · 120 rejected (metadata) · <elapsed>`. The downloads alone wait about 12-13 min (step 7); judging overlaps them. |
| 12 | `GET /v1/items/{id}` until `index.status: indexed` (only for step 14) | Waits for search readiness. | – |
| 13 | `GET /v1/library/topics/aad-comp-finance/items?decision=accept&sort=score:relevance` | **Prints the listing:** title, authors, year, permalink, and the reasons, each with an evidence citation (§7.2). Below it, the works found but not held ("known, not held": link-only, or not on an allowed host), each with its link and reason. | the list the owner asked for |
| 14 | (optional) `POST /v1/library/search` with a question | Answers with citations: section, paragraph and equation number from arXiv HTML (for an item without HTML, where GROBID found them), and page. | the answer |

**Checks that make the test pass** (devai's Phase 3 exit criteria cite all seven, check 7
included):

1. The listing is non-empty. Every accepted item has at least one reason whose evidence resolves to
   a unit, and its `profile`, `deciders` and `recorded_at` stamps are present.
2. Every accepted item has in its artifacts the PDF of the version recorded at accept, plus, for
   an arXiv item, the arXiv HTML when the fetch staged it (otherwise its primary text's
   `quality.warnings` gives the reason, §3.2), and a `license` field, which may say `none_stated`
   (§5.3; final agreement with devai, 2026-09-26).
3. No rejected item has an artifact or a text version left: its item record lists only
   tombstones, its `staging_path` is gone, its unit addresses answer 410 with a tombstone
   (`GET /v1/items/{id}/texts/{text_id}/units`), and its evidence quotes are in its verdict.
4. **The re-run is idempotent.** Running steps 3-7 again:
   - every candidate shows `library.status`;
   - the fetch job reports `upstream.requests == 0` for known items;
   - no new verdicts are created, and a verdict replayed from the outbox returns 200 with the same
     record.
5. **Progress never goes silent.** During steps 3-11, no gap between the service's heartbeats or
   `updated_at` changes, or between aiagent's ticks during its LLM calls, exceeds 5 s. Within 2 s of
   every job's start, aiagent shows an ETA, or "ETA unknown: <reason>" with the elapsed time and
   the items done.
6. **The indexes are derived** (run by devai, once): after every derived index is deleted (the
   alias index, the candidate cache, the topic listings, and the keyword and vector indexes) and
   rebuilt from the plain files, the §7.3 listing returns byte-identical rows, and every alias
   resolves as before.
7. **Not arXiv only** (added 2026-09-26, with devai: its Phase 3 exit criterion). The run searches
   OpenAlex as well as arXiv (steps 2-3), and its results hold at least one non-arXiv work: an
   open-access copy fetched from an allowed host (step 7), or a "known, not held" link record with
   its reason (steps 6 and 13).

---

## 11. Open questions

### 11.1 For the owner (O1-O15 answered 2026-09-26)

**The owner answered "go with recommendations" (2026-09-26):** for O1-O14, each recommendation in
the right-hand column below is the decision. O13 was resolved separately (L15: not arXiv-only,
OpenAlex in v1). For O5, the h-index is advisory (it raises a score and never gates), so no
threshold is needed. **O15 was answered later the same day:** devai moved O-D10 to the owner, who
decided it in devai's session (devai's D28, relayed by devai), as recommended below.

| # | Question | aiagent's recommendation |
|---|---|---|
| O1 | Is a rejection of an item that is **within the outer bound but off topic** for this query permanent? A paper on PDE numerics, rejected for the AAD topic, would otherwise never be judged for a later PDE topic. devai needs the answer before it builds the one-current-verdict rule. | An `off_topic` reject is scoped to its topic (`scope_topic_id`). Judging the item for a different topic counts as "asked" under L5, so it may be fetched and judged again for that topic (§5.2). `out_of_bounds`, `low_quality` and `duplicate` stay final. |
| O2 | **A new arXiv version after acceptance.** Refresh (replace the PDF and HTML, and re-judge) automatically, or only on request? And should the old version's text units be kept, so old citations still resolve? | Report `latest_seen` and refresh only on request. Keep the old text units (small), and drop the old PDF and HTML ("latest version only"). |
| O3 | **Code licences and length.** Keep excerpts from a repo without a licence ("all rights reserved"), or only link to it? What counts as "short"? | Keep excerpts only when the licence is detected, otherwise link only. At most 80 lines per excerpt and 10 per repo. |
| O4 | **"Independent" means what?** | Independent describes who publishes the document fetched. Company or government-agency publications (vendor white papers, bank research pages, agency reports) are out. arXiv, open-access journals and author pages are in, whatever the authors' affiliations: much of the AAD-in-finance literature is written by bank quants. SSRN is link only (L14): its works are known, never fetched. `affiliations[].type` is information only, never a gate. |
| O5 | **The authority metric.** h-index from OpenAlex or Semantic Scholar? What threshold, and do `name_only` matches count? | h-index from one declared source, stamped with its as-of date: OpenAlex in v1 (L15); the threshold stays open. It raises a score and is never a hard gate. `name_only` matches are down-weighted. |
| O6 | **Non-English originals.** Keep only the link and sha256 (as L6 reads)? If the link dies, a citation to the original can no longer be checked. | As decided: the link plus sha256 only. The original's bytes and its source-language text are deleted once the anchors are copied into the English text. |
| O7 | **Unreadable full text** (scanned PDFs): OCR in the service, or not at all? | OCR is mechanical, so it belongs in the service, later. For now a scanned PDF is the fetch failure `no_text_layer`, with no verdict, so it stays fetchable when OCR arrives. |
| O8 | **Who may override a verdict?** | Only the owner, through an explicit aiagent command that confirms. The service records `requested_by: owner`. |
| O9 | **Metadata-only rejections:** keep a record for each one (a few KB each; possibly many)? | Yes. Otherwise the same candidates would be pre-judged again on every run. |
| O10 | **Wikipedia and blogs:** which revision is cited, and is CC BY-SA text stored whole? | Cite the `oldid`, and store it whole: the library is private (principle 10). |
| O11 | **Staging TTL** for fetched items that were never judged. | 30 days. |
| O12 | **Topics:** free topic ids minted by aiagent per request, or a curated list the owner maintains? | Free ids, which aiagent reuses when a new request matches an existing topic (it asks the owner when unsure). |
| O13 | **Resolved (L15, 2026-09-25/26).** Was: is an arXiv-only first pass, with no author metric, acceptable? The owner: "try to provide sources if feasible, do not limit them to arXiv only". | v1 searches arXiv and OpenAlex, fetches the open-access works OpenAlex finds on allowed hosts, and carries OpenAlex citations and author h-index (advisory); Unpaywall by DOI. Semantic Scholar later; SSRN link only (L14); author pages later (O-D13). |
| O14 | **The first run is judged by the LLM alone.** Students need a distill campaign each (teacher labelling of 2,400 documents took 23.6 min, plus GPU holds). Acceptable for the first test? | Yes. Students follow, with fixed per-profile questions (A6), once the LLM verdicts give a labelled pool. |
| O15 | **Answered (2026-09-26, the owner, devai's D28; was O-D10).** A 30-day trash for the artifacts of revoked items? A revoke (the owner turns an accept into a reject) deletes the artifacts like any reject. devai's D12 keeps no artifacts for rejects, so a grace period is an exception to it that the owner must approve (2026-09-26, from devai). | Keep it: 30 days, as a safety net against a faulty client (devai recommends the same). **Decided (D28):** a revoked item's artifacts stay 30 days in `trash/`, then are deleted; an exception to devai's D12 (§5.5). |

### 11.2 For devai (answered 2026-09-26, relayed by devai)

| # | Question | devai's answer |
|---|---|---|
| O-D1 | Where the service runs and on which port, and which base URL aiagent defaults to. | A plain compose container `devai-library` on devai-net and devai-lab-egress (like the MCP gateway), with no host port. Default base URL `http://devai-library:8080`; the lab reaches it directly (in `NO_PROXY`). §9.1. |
| O-D2 | The lab mount, like `/laya`, and whether secrets and index files stay outside the mounted tree. | The library root read-only at `/library` (`inbox/` read-write later), like `/laya`; exactly one configured directory, `LIBRARY_DIR`. Secrets are never in the library directory (sops/age rendered to a tmpfs, only in the service). The derived indexes (`index/`) and the service's own state (`state/`: politeness counters, breakers, the arXiv contact record) live under the library root, and aiagent must not depend on their files. A2, §9.5. |
| O-D3 | **Obsolete** (L12): no e-print is fetched, so there is nothing to convert. | Replaced by arXiv HTML (LaTeXML's section and equation numbers and ids, `equation_number_method: html`) and GROBID for PDFs without HTML. **2026-09-26, from devai:** arXiv's HTML does not keep the authors' `\label` keys, only LaTeXML's ids; HTML-primary text is v1 (L12 notes). |
| O-D4 | Dropped: the service counts no tokens (§3.4). | – |
| O-D5 | The embedding model (English, CPU), its input limit, how chunks map back to units, and the rebuild time (10k papers × 300 units ≈ 3M vectors). | bge-small-en-v1.5 (MIT, 384 dimensions, 512-token input); a unit over 512 tokens is chunked internally, and hits report the unit. Embeddings are derived files keyed by (unit text sha256, model sha256). Exact numpy search over memory-mapped shards (fp32 first, int8 later). Rebuild times will be measured, not promised, and reported by `/v1/info` (§9.1). **2026-09-26, the owner (devai's D24):** the model gets its own volume, `/var/cache/devai/embed`, internal to the service. |
| O-D6 | The keyword engine and English analyser, and the rebuild time. | SQLite FTS5 (porter, unicode61), derived from the plain files. |
| O-D7 | arXiv access at the published pace, and how PDF and HTML downloads are counted. | The API (`export.arxiv.org`): 1 request per 3 s, one connection, GET only, cached 24 h. `arxiv.org` `/pdf` and `/html`: Crawl-delay 15 s, one queue per host (25 papers with PDF and HTML ≈ 12-13 min of waiting, shown in the ETA). devai is checking an anonymous PDF mirror (Cornell's public GCS bucket). arXiv asks for contact before more than 1,000 full-text downloads; the service counts them. **2026-09-26, from devai:** the service schedules by rate group: `arxiv-legacy-api` (`export.arxiv.org` and `oaipmh.arxiv.org`: one connection, 3 s, shared by searches and the fetch job's `arXivRaw` requests) and `arxiv-files` (`arxiv.org`: 15 s), one group per other host, reported as `per_group` by `/v1/info` and in `eta_inputs` (§8.3, §9.1). **The PDF mirror (the owner, devai's D27, 2026-09-26):** checked in devai's Phase 1 and used only if its terms and robots.txt allow automated downloads; until then ETAs assume 15 s per `arxiv.org` file. At the full-text cap (a configured count below 1,000, each paper counted once) arXiv full text stops until the owner records having contacted arXiv, shown like any stop with cause `arxiv_fulltext_cap` (§9.1). It stops arXiv's PDF and HTML downloads only, not its search or OAI-PMH, and fails them promptly, with no quota wait. A single 429 or 503 from arXiv is a wait, never a breaker event; an origin 403 opens the breaker (`breaker_open`), which stops all arXiv traffic until an operator clears it (§9.1). |
| O-D8 | The metadata aggregators for open-access works and author metrics, and their credentials. | OpenAlex (metadata, open-access locations, citations, h-index; $0.10/day keyless, $1/day with a free key) and Unpaywall by DOI (open-access location, licence). SSRN is link only (L14). Semantic Scholar later. **2026-09-26, from devai:** a work with neither an arXiv id nor a DOI gets the key `openalex:W<id>`, and an OpenAlex id is always an alias (§1.1); enrichments and `GET /v1/candidates/{item_id}` are v1, in devai's Phase 2 (§2.3). |
| O-D9 | Whether jobs resume after a restart, and how per-source queues are shared between jobs. | Jobs resume: item steps are idempotent, and unfinished items are re-queued from `jobs/<id>/`. Per-source FIFO queues across jobs in v1. §8.5. **2026-09-26, from devai:** the queues are per rate group (O-D7). |
| O-D10 | A grace period for the artifacts of revoked items (§5.5). | **Moved to the owner as O15 (2026-09-26, from devai):** a grace period is an exception to devai's D12 (rejects keep no artifacts), so the owner decides; devai recommends a 30-day trash. **Answered the same day (the owner, devai's D28):** revoked items keep their artifacts 30 days in `trash/` (§5.5, §11.1). |
| O-D11 | (v1 since 2026-09-26) Printed equation and section numbers without compiling, and how they are reported (`equation_number_method`). | From arXiv HTML: LaTeXML's printed numbers (`equation_number_method: html`); GROBID for PDFs without HTML. **2026-09-26, from devai (verified):** every formula carries its TeX (`application/x-tex`, 78 of 78 on `2609.27602`); the authors' `\label` keys are not in the page, so `equation_labels` stays empty for HTML texts (§3.3). |
| O-D12 | The HTTP identity the service presents to sources, and how robots.txt decisions are cached and reported. | `User-Agent: devai-library/<ver> (+mailto:<address>)`, with a dedicated address; no external fetch until it is configured (L16). robots.txt through Protego with an RFC 9309 fetch wrapper (24 h cache; 500 KiB cap); API calls follow their API terms instead (L13). Candidates report `access.robots_allowed` and `reason`. **2026-09-26, from devai (replacing the earlier rule for 4xx and 5xx answers):** the rules forbid the path: `robots_disallowed`; robots.txt answers 401 or 403, or a challenge: `robots_denied`; robots.txt answers 429 or 5xx, or fails on the network: `robots_unreachable`, treated as a disallow for now, with the cached copy kept and a retry later; 404, 410 and any other 4xx mean "allow". The three codes are `not_fetchable` reasons and fetch failure codes (§2.2, §3.1). **2026-09-26, from devai:** with the address unset, `/health` is `degraded` with `contact.configured: false`, `/v1/info` shows each source `down` with cause `contact_not_configured`, and a job-creating request for an external source answers 503 `unavailable` with that cause (§9.1). |
| O-D13 | How author pages and blogs are discovered. | Later: author pages from OpenAlex open-access locations, references parsed by GROBID and LaTeXML (arXiv HTML), and ORCID. No web-search API. Blogs from a curated list the owner approves. |

devai's further answers and adjustments of 2026-09-26 that no O-D question asked for (arXiv HTML
facts, HTML-primary text in v1, versions and licences from `arXivRaw`, English-only permanent
records, stops, `egress_blocked`, Medium, tombstones and enrichments in v1, and the revised
acceptance test), and its last answers of the same day (the stop causes and how a partial stop
shows, 429s as waits and the two breaker kinds, retries of the HTML request, the agreed names,
licences of non-arXiv works, the HTML extras `anchor.html_id`, `anchor.env`, `refs.cites` and
`refs.labels` in v1, and `GET /v1/verdicts/{id}` and `GET /v1/library/topics` in v1), and the final
agreement of the same day (check 2, waits outside the retry budget, `error.details.cause`
everywhere, the agreed shapes, `limited` sources, what counts as "no HTML", `egress_blocked` and
`egress_unavailable`, `ssrn:` fetch keys, the HTML-only fields, "allowed host", and the excerpt
rule after a repo refresh), are recorded in the §0.1 notes, as are the owner's answers to devai's
own questions (D24-D28). devai's plan matches this document as revised.

---

## 12. Critique disposition

The adversarial review of 2026-09-25 (findings 1-20, cuts C1-C8) is applied throughout, except for
the points below.

- **Finding 8, a 200 for a verdict equal to the current one, ignoring `verdict_id` and
  `decided_at`: rejected.** The outbox replays exact bytes, so a retry always carries the same
  `verdict_id`. A field-by-field equality rule would be a second definition for a case the outbox
  already covers, and a concurrent duplicate gets `verdict_exists`, to which aiagent reacts by
  recording a topic assessment.
- **Finding 12, a per-job `deadline_s`: rejected.** aiagent enforces its own deadline with
  `POST /v1/jobs/{id}/cancel`; a server-side deadline would duplicate it. The per-item retry budget
  and `eta.reason: source_down` are applied.
- **Finding 14, `counts.state_tokens` reported by the service: rejected.** C7 removes token counting
  from the service. aiagent measures the escaped state with its own tokenizer port (A4).
- **Finding 14, lowering the cap to 512 tokens: not taken.** The review's other option is used
  instead: topic text never goes into a unit state (A3, A6). The cap is now 1,300 characters (C7).
- **Finding 16, stamping the served model's weights revision: partly.** The router's `/health`
  reports the model, context and MTP spec, not a weights revision. aiagent stamps what `/health`
  reports (`served`), and `revision` only when the router reports one.
- **Finding 19c, 422 for an arXiv accept whose `keep` omits one of the fetched formats: replaced by
  a simpler rule.** `keep.formats` is removed: an accept always keeps every fetched artifact of the
  version (L5), so there is nothing to refuse.
- **Finding 20, the embedding cache as a requirement: kept as a suggestion (O-D5).** How devai
  stores derived data is its business (principle 8). aiagent asks only for a rebuild-time target.
  devai has since adopted the suggestion, and will measure rebuild times rather than promise a
  target (O-D5, answered).

Applied beyond the review: `license` also leaves the reject bases (a licence can change, and a
private library can keep any freely downloadable paper; for code it decides only whether excerpts
are kept, O3). Partial translations are cut for good, not only from v1: aiagent can judge from its
own unsubmitted translation, and only an accept needs a stored English text.
