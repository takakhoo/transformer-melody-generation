# Target venue

Decided 8 October 2026 after checking official pages for the venues below. Anything marked *estimate* is inferred from a previous cycle because the 2027 call is not posted yet; replace it when the call appears.

## Recommendation: TISMIR, Open Music Data special collection

**Submit to TISMIR (Transactions of ISMIR) as a Research Article in the special collection "Open Music Data for Music Processing Research". Deadline: 1 December 2026.** If that date slips, the same manuscript goes in as a regular TISMIR research article whenever it is ready, since the journal takes submissions year-round.

Why this venue:

- **Topical fit is direct.** The special collection lists "Reproducibility, benchmarking, evaluation, and dataset versioning", "Design, curation, and documentation of music datasets", "Methods for analyzing, enriching, validating, and integrating music data", and "Copyright, licensing, ethics, and responsible data sharing" as topics, and it explicitly welcomes method-driven work on dataset analysis ([call PDF](https://account.transactions.ismir.net/index.php/up-j-tismir/libraryFiles/downloadPublic/9), [announcements](https://transactions.ismir.net/en/announcements)). A leakage audit of the Nottingham and The Session splits (transposed copies, near-duplicates), released deduplicated splits, and evidence that leakage inflates scores and drives copying is that kind of paper.
- **Room for the whole study.** The limit is 8,000 words including references ([submissions](https://transactions.ismir.net/about/submissions)). The leakage audit, model-size sweep, dedup ablation, temperature sweep, and objective evaluation would all have to be squeezed into six ISMIR pages with no appendix.
- **Preprints are allowed.** TISMIR says authors are "permitted and encouraged" to post work online before and during submission ([submissions](https://transactions.ismir.net/about/submissions)). ISMIR "strongly discourages" near-duplicate preprints during review ([ISMIR 2026 author guidelines](https://ismir2026.ismir.net/authors/author-guidelines)). A paper about benchmark leakage gains from going on arXiv early so people can use the clean splits.
- **It keeps ISMIR 2027 free for Audio Sliders.** No per-author cap appears in the ISMIR 2026 guidelines, so a second ISMIR submission is allowed. The cost is the workload: Audio Sliders already targets ISMIR 2027 ([`../../audio-diffusion-control/docs/VENUE.md`](../../audio-diffusion-control/docs/VENUE.md)), and Lacquer lists it as an option ([`../../remaster/docs/research/venues.md`](../../remaster/docs/research/venues.md)). Three solo papers in one late-March window is too many.
- **Same audience as ISMIR.** TISMIR is the society's own journal. The ISMIR 2025 business meeting slides describe it as "Submit any time; can present at ISMIR", with roughly 50% acceptance ([slides PDF](https://ismir.net/wp-content/uploads/2026/01/2025_business_meeting.pdf)).

What the framing has to do: the call says "A substantial connection to open music data is essential" and that data-centred topics should be the primary focus. Lead with the data (audit method, leakage statistics per benchmark split, released clean splits and dedup tool). Present memorization against model size, dedup, and temperature as the evidence that the data problem matters, and keep the in-browser generator as a short section or supplementary link.

What happens to the generator: submit it to the ISMIR 2027 Late-Breaking/Demo track (ISMIR 2026 had a [call for LBD](https://ismir2026.ismir.net/call-for-late-breaking-demo)), so the work still gets a slot at the London conference.

## TISMIR rules (verified 8 October 2026)

| Item | Rule | Source |
|---|---|---|
| Deadline | Special collection: 1 December 2026 (written "01.12.2026"; the call was posted 20 May 2026, so the only possible reading is 1 Dec). Regular articles: no deadline. | [call PDF](https://account.transactions.ismir.net/index.php/up-j-tismir/libraryFiles/downloadPublic/9), [announcements](https://transactions.ismir.net/en/announcements) |
| How to submit | Through transactions.ismir.net; the cover letter must say the paper is meant for the "Open Music Data for Music Processing Research" special collection. | call PDF |
| Length | At most 8,000 words, including references, citations, and notes. Over-length papers can be desk-rejected. | [submissions](https://transactions.ismir.net/about/submissions) |
| Template | Optional at submission; accepted papers are typeset in the journal style. Official LaTeX: [new template](https://github.com/ismir/paper_templates_TISMIR_new) (v1.0.0, downloaded to `paper/template/`), [old template](https://github.com/ismir/paper_templates_TISMIR_old), [Word](https://s3-eu-west-1.amazonaws.com/ubiquity-partner-network/up/journal/tismir/TISMIR-Word-Template.docx). | submissions |
| Review | Double-blind for Research, Dataset, Overview and Educational articles. Dataset articles may be non-anonymised. | submissions |
| Preprints | Allowed and encouraged before and during submission. | submissions |
| Abstract | About 250 words (template text). | `paper/template/TISMIRtemplate.tex` |
| Prior work | Extensions of published work need at least 50% new content. | call PDF, submissions |
| Cost | APC £530 plus tax where applicable; waiver requests up to 100 words, made at submission. | submissions |
| Guest editors | Stefan Balke, Magdalena Fuentes, Dasaem Jeong, Meinard Müller. | call PDF |

Template notes: the template's `\author`/`\aff` blocks carry the names, so delete them for the anonymous version. `\linenumbers` is on for review. Bibliography style is `apaTISMIR` (author-year). The template ships its own fonts (FSMe, SimSun) and `Styles/Jnls-Template-V3.cls`; compile from inside `paper/template/` or on Overleaf (the README has an "Open in Overleaf" link). It was not test-compiled here because there is no TeX install on this machine.

## All venues compared

Fit: how well the leakage/memorization study matches the venue's audience (High, Medium, Low).

| Venue | Fit | Format | Review | Next deadline | Preprints |
|---|---|---|---|---|---|
| **TISMIR** (recommended) | High | 8,000 words incl. refs; LaTeX or Word template, optional | Double-blind | Special collection 1 Dec 2026; otherwise rolling | Encouraged |
| **ISMIR 2027** (London, QMUL, 12-16 Sep 2027; tentative theme "Responsible Music Information Research") | High | 2026 rules: 6 pages of content + refs/ethics/AI statement pages, no appendices, ISMIR template ([2026v1](https://github.com/ismir/paper_templates/releases/tag/2026v1); a 2026v2 release also exists) | Double-blind | Not announced. *Estimate* late March 2027: ISMIR 2025, also a September conference, had its full-paper deadline on 28 Mar 2025; ISMIR 2026 (November) used 20 Apr abstract / 27 Apr paper | Near-duplicate preprints strongly discouraged; no promotion during review |
| **ICASSP 2027** (Toronto, 16-21 May 2027) | Medium-low | 4 pages + 1 page of references; OJSP track 8+1 | Not stated on the pages checked | **Passed.** CFP page lists 23 Sep 2026 (an earlier CFP PDF said 16 Sep) | Not stated |
| **EvoMUSART 2027** (EvoStar, Mainz + hybrid, 31 Mar-2 Apr 2027) | Medium | 14 pages + unlimited refs/acks, Springer LNCS, EasyChair | Double-blind | **1 Nov 2026** | Not addressed; "must be original and not published elsewhere" |
| **AIMC 2027** (Montréal, CIRMMT McGill, 18-20 Aug 2027) | Medium | Call for submissions "Coming soon" | Not posted | Not posted; AIMC 2025 had its deadline in April | Not posted |
| **SMC 2027** | Medium | Not announced. SMC 2026: up to 8 pages (6 encouraged), SMC LaTeX template | SMC 2026: single-blind | Not announced; SMC 2026 deadline was 27 Mar 2026 | Not stated |
| **NIME 2027** (Paris, Devinci, 21-25 Jun 2027) | Low (generator only) | NIME 2026: 6,000 / 4,000 / 2,000 words excl. refs | Double-blind | Not posted; NIME 2026 paper deadline was 12 Feb 2026 | Not stated |
| **CMMR** | Low-medium | No 2027 call found. CMMR 2025: CC-licensed proceedings with DOI, selected papers extended into Springer LNCS | CMMR 2025: single-blind | None announced; CMMR 2025 deadline was in May 2025 | Not stated |
| **ICLR 2027** | Low-medium | 9 pages main text (10 camera-ready), [style files](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip) | Double-blind | **Passed** (abstract 18 Sep, paper 25 Sep 2026); ICLR 2028 *estimate* Sep 2027 | arXiv allowed |
| **NeurIPS 2027 Evaluations & Datasets** (renamed from Datasets & Benchmarks in 2026) | Medium | 2026: follows main-track format; data hosted on a dedicated platform with Croissant metadata at submission | 2026: double-blind by default, single-blind opt-in | Not announced; 2026 was 4 May abstract / 6 May paper, so *estimate* early May 2027 | Per main-track rules |
| **ACL / EMNLP 2027** (via ACL Rolling Review) | Medium-low | 8 pages long / 4 short + required Limitations, [acl-style-files](https://github.com/acl-org/acl-style-files) | Double-blind | ARR October cycle 12 Oct 2026; ACL 2027 final ARR cycle "January 2027"; EMNLP 2027 not listed yet | No anonymity period |
| **JNMR** (Taylor & Francis) | Medium | Official instructions not verified (the Taylor & Francis page is behind bot verification) | Secondary sources: double-blind, ScholarOne | Rolling | Not verified |

### Notes per venue

- **ISMIR 2027.** This is the best fit on topic, and the tentative theme suits a memorization paper. It loses on page budget, on the preprint rule, and on clashing with Audio Sliders (and possibly Lacquer) in the same deadline window. It stays the backup if TISMIR rejects, or if a conference paper is preferred over a journal article. Sources: [Wikipedia ISMIR conference table](https://en.wikipedia.org/wiki/International_Society_for_Music_Information_Retrieval) for dates, [2025 business meeting slides](https://ismir.net/wp-content/uploads/2026/01/2025_business_meeting.pdf) for QMUL, chairs, and theme, [ISMIR 2026 call](https://ismir2026.ismir.net/authors/call-for-papers) and [author guidelines](https://ismir2026.ismir.net/authors/author-guidelines) for rules, [ismir2025.ismir.net](https://ismir2025.ismir.net/) for the 2025 dates (as recorded in the Lacquer venue note on 2 Oct 2026). `ismir2027.ismir.net` did not resolve on 8 Oct 2026.
- **ICASSP 2027.** Closed, and four pages is too short for this study anyway. [CFP](https://2027.ieeeicassp.org/?p=3718), [author guidelines](https://2027.ieeeicassp.org/?p=4445).
- **EvoMUSART 2027.** This is the only open deadline that would accept the paper nearly as it stands (24 days out), and LNCS has room. The audience is generative and creative AI, so the leakage audit is a weaker fit than the generator. It is archival, so it would use the paper up. [EvoStar 2027](https://www.evostar.org/2027/), [EvoMUSART page](https://www.evostar.org/2027/evomusart/), [submission rules](https://www.evostar.org/2027/submit-paper/).
- **AIMC 2027.** The audience leans toward creative practice. Recheck once the call is posted. [aimusiccreativity.org](https://aimusiccreativity.org/), [2027 site](https://2027.aimusiccreativity.org).
- **SMC.** The SMC Network homepage lists SMC 2026 (Zagreb, 2-7 Nov 2026) and nothing for 2027. [smcnetwork.org](https://smcnetwork.org/), [SMC 2026 CFP](https://smc26.mbz.hr/en/submissions/call-for-papers).
- **NIME.** Only the in-browser generator fits. [nime.org](https://nime.org/), [NIME 2026 paper call](https://nime.org/web_archive/2026/call/papers/index.html).
- **CMMR.** No 2027 edition found. [CMMR 2025 call](https://cmmr2025.prism.cnrs.fr/?p=413).
- **ICLR / NeurIPS E&D.** ML reviewers would read the leakage result as a known problem (benchmark contamination) applied to a small domain, and they would expect larger models. ICLR also requires an author who qualifies as a reviewer through a prior publication at a listed venue. [ICLR 2027 CFP](https://www.iclr.cc/Conferences/2027/CallForPapers), [author guidelines](https://www.iclr.cc/Conferences/2027/AuthorGuidelines), [NeurIPS 2026 E&D CFP](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets).
- **ACL / EMNLP.** The memorization angle is relevant, but the reviewers work on text LMs. ARR's service requirement (effective October 2026) makes each submission name a qualified service contributor, which a solo author has to be. [ARR dates](https://aclrollingreview.org/dates), [ARR CFP](https://aclrollingreview.org/cfp).
- **JNMR.** A reasonable journal alternative if TISMIR declines, though it is less MIR-central. Check the official [instructions for authors](https://www.tandfonline.com/action/authorSubmission?show=instructions&journalCode=nnmr20) in a normal browser before relying on it.

## Plan to 1 December 2026

1. Finish the leakage audit on Nottingham and The Session first: exact, transposition-invariant, and near-duplicate matching across the standard splits. Publish the clean splits and the dedup tool. This is the part the special collection is judged on.
2. Run the model-size × dedup × temperature memorization grid on the clean and original splits.
3. Write it in `paper/template/TISMIRtemplate.tex`, anonymised, at or under 8,000 words, with an abstract of about 250 words.
4. Post to arXiv when submitting (allowed), with anonymised code and data links in the submission.
5. Cover letter: name the special collection. Request an APC waiver at submission if needed.
6. Later: send the generator to the ISMIR 2027 late-breaking/demo track once that call is out.
