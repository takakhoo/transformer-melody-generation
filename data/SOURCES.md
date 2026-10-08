# Raw data sources

All files live under `data/raw/<dataset>/`. Everything was fetched on **2026-10-08** with `curl` or from pinned git commits, with no account, login, click-through or API key. GitHub repos were pulled as `codeload.github.com` zip archives of a specific commit so the archive hash is reproducible. Archives were kept next to their extracted trees where extracted.

Hashes are SHA-256 (`shasum -a 256`). Sizes are bytes. "Members" is the number of files inside the archive.

Inventory script: `scripts/inventory.py`, run with `.venv/bin/python -I scripts/inventory.py`. Its output is pasted at the bottom.

## License summary (read this before training anything)

| Dataset | License as published | Practical note |
|---|---|---|
| The Session | Custom "License for contents" with an explicit **prohibition on LLM use**, plus ODbL for the database | See section 1. Decide whether a melody LM counts before using it. |
| Nottingham (jukedeck) | GPL-3.0 (repo LICENSE.md) | Original NMD by Eric Foxley has no stated license |
| Nottingham (Seymour Shlien edit) | None stated | Redistributed freely since 2011 |
| Boulanger-Lewandowski 2012 splits | None stated for the pickles; per-source terms linked on the original page (Piano-midi.de copyright page, MuseData license agreement) | MuseData is CCARH-licensed academic data |
| Essen (**kern, CCARH) | CCARH license: academic research, single-user, no redistribution | Do not redistribute the files |
| PDMX | Zenodo record: CC-BY-4.0. Per-score field: `publicdomain` or `cc-zero` | Authors recommend the `no_license_conflict` subset |
| Lakh MIDI | CC-BY 4.0 (dataset). Underlying songs are mostly copyrighted | Research use is the norm |
| POP909 | MIT (repo LICENSE) | Songs are copyrighted pop arrangements |
| Hooktheory (Sheet Sage release) | CC BY-NC-SA 3.0 | Non-commercial, share-alike |
| IrishMAN | MIT (HF card) plus "research use only and not for commercial purposes" disclaimer | Derived from thesession.org and abcnotation.com; see note in section 8 |
| Wikifonia | not downloaded | No legal no-login source exists (site closed 2013 over licensing) |

---

## 1. The Session (thesession.org data dump)

- Source: https://github.com/adactio/TheSession-data, commit `f5b64eb7230510b59cb2c34caf05c04a7ebaa9cc` (commit date 2026-10-05)
- Archive URL: https://codeload.github.com/adactio/TheSession-data/zip/f5b64eb7230510b59cb2c34caf05c04a7ebaa9cc
- Local: `raw/thesession/TheSession-data-f5b64eb.zip` (extracted alongside)
- Size: 50,827,435 bytes zipped, 250 MB on disk with extraction; 24 members
- SHA-256: `2a9b92cc42806bed294812d8a50527ed8be7c4a3fb64dfd6f6d9f9e26ec9bb5b`
- Content used for melodies: `csv/tunes.csv` and `json/tunes.json` (one row per *setting*; columns `tune_id,setting_id,name,type,meter,mode,abc,date,username,composer`).
- Not fetched: `json/sets.json` and `thesession.db` are Git LFS objects (146 MB, 144 MB), so the codeload zip holds only LFS pointer stubs for them. `csv/sets.csv` has the same set data and is present. Neither file is needed for melody work.
- License: `LICENSE.md` in the repo. Full text is in the extracted tree. History: plain ODbL from 2020-01-30; on **2025-10-08** (commit `2c07104`) the author added a prohibition on LLM use, reworded on 2026-06-16 and 2026-06-28. The current opening section reads, verbatim:

  > ## Prohibition on LLM Use:
  >
  > You may not use, adapt, modify, or process the material in any way with Large Language Models. This includes but is not limited to training Large Language Models, utilizing LLM tools, or incorporating the material into any LLM-related applications or systems, except as expressly allowed in the following section.

  The following section allows LLM use only for accessibility. The rest of the file is the ODC Open Database License (ODbL) 1.0.

  **Consequence for this study:** whether a melody-only "language model" over ABC counts as a Large Language Model under this clause is a judgment call that has to be made by you (and possibly checked with the author, Jeremy Keith). Because Claude is itself an LLM tool, the inventory below prints only Session metadata (ids, names, type, meter, mode, ABC length) and not the ABC bodies. Run `scripts/inventory.py --show-session-abc` yourself to see them.

## 2. Nottingham Music Database (ABC)

### 2a. jukedeck/nottingham-dataset (cleaned ABC + MIDI)
- Source: https://github.com/jukedeck/nottingham-dataset, commit `0992bb6cd864f6d4b90d6663e08c0ef53ffaa08f`
- Archive URL: https://codeload.github.com/jukedeck/nottingham-dataset/zip/0992bb6cd864f6d4b90d6663e08c0ef53ffaa08f
- Local: `raw/nottingham_abc/nottingham-dataset-0992bb6.zip` (extracted alongside)
- Size: 1,828,800 bytes; 3,120 members
- SHA-256: `9c9c87d745779af617ad4cd0d715210bfc17ff89bc427928593462872542f9a8`
- Contents: `ABC_original/` (untouched copy of the abc.sourceforge.net NMD files), `ABC_cleaned/` (jukedeck's edits for machine readability), `MIDI/` (full, `melody/`, `chords/`).
- License: GPL-3.0 (`LICENSE.md` in the repo, GNU GPL v3 text).

### 2b. Seymour Shlien's corrected edition
- Source page: https://ifdo.ca/~seymour/nottingham/nottingham.html (saved as `seymour_nottingham.html`, SHA-256 `5bc6a058ba19b33697b0ad207e1f9122aeb7de07d356500bd11aed9625ac3c59`; page says "last updated on October 2 2011")
- Archive URL: https://ifdo.ca/~seymour/nottingham/nottingham_database.zip
- Local: `raw/nottingham_abc/nottingham_database.zip` (not extracted; the inventory reads it in place)
- Size: 142,934 bytes; 16 members
- SHA-256: `f79a4bffe78b16d630d4d69f9c62775a7aa246d0973c4d8714ab6c5139ff5a3b`
- License: none stated. `NOTES.txt` inside credits Eric Foxley (original database), Jay Glanville (conversion) and James Allwright (abc.sourceforge.net hosting).
- The abc.sourceforge.net copy (`https://abc.sourceforge.net/NMD/nmd/NMD.zip`) now returns 404. Its index page is saved as `abc_sourceforge_NMD_index.html` (SHA-256 `d23e347c295ff31bbb2ca92ee198422a062358d34ea21387e1b9e04cb2ad184f`). jukedeck's `ABC_original/` is a copy of that collection.

## 3. Boulanger-Lewandowski, Bengio & Vincent (ICML 2012) piano-roll splits

The original host `www-etud.iro.umontreal.ca/~boulanni/` no longer resolves (DNS failure on 2026-10-08). Mirrors used:

| File | Source URL | Bytes | SHA-256 |
|---|---|---|---|
| `wayback_JSB Chorales.pickle` | https://web.archive.org/web/20141211164413id_/http://www-etud.iro.umontreal.ca:80/~boulanni/JSB%20Chorales.pickle | 2,051,809 | `69dc4c3e52800959ac0ab3b82aa3e5f55285ce64f1705acb3e3a8ef77a5c051c` |
| `wayback_Piano-midi.de.pickle` | https://web.archive.org/web/20141211164423id_/http://www-etud.iro.umontreal.ca:80/~boulanni/Piano-midi.de.pickle | 7,413,484 | `273d593b13d9a348410d261c0223d47523e2a9b41702137aaab5922242518296` |
| `wayback_Nottingham.zip` (source MIDI, train/valid/test dirs) | https://web.archive.org/web/20190502175445id_/http://www-etud.iro.umontreal.ca/~boulanni/Nottingham.zip | 692,283 | `fd4eb21b656002790cb8429c3482518876c48881ce58270f5296fe5aad50075f` |
| `wayback_icml2012.html` (original dataset page) | https://web.archive.org/web/20130430003325id_/http://www-etud.iro.umontreal.ca:80/~boulanni/icml2012 | 2,384 | `26518a87cd28c620d485973c5aa5a0b8da77016656379a1b7f18d37b9ccbdceb` |
| `tcn_JSB_Chorales.mat` | https://raw.githubusercontent.com/locuslab/TCN/2f8c2b817050206397458dfd1f5a25ce8a32fe65/TCN/poly_music/mdata/JSB_Chorales.mat | 96,095 | `ac0e608527cc411c7e21ef7d38be47f0de7fb6b13b3ae168b8eccf1c2a9b18d4` |
| `tcn_MuseData.mat` | same path, `MuseData.mat` | 947,219 | `e8d9eb422ec3833c25f43a9f854d4ee0b9118a20b45f2c183395cdc30ffa14ff` |
| `tcn_Nottingham.mat` | same path, `Nottingham.mat` | 371,688 | `d154f7ca2e90799dc1dc011d8fc048f2fc6ef3f7c69d974b9cf27a471634fbea` |
| `tcn_Piano_midi.mat` | same path, `Piano_midi.mat` | 223,732 | `990a8a71d5764a7eaa3e78d9e5023d1470380fe2db3a1448265a97a24c8b1c34` |

- **Gap:** `Nottingham.pickle` and `MuseData.pickle` are not in the Wayback Machine (CDX query on 2026-10-08 returned no captures) and no other verifiable mirror was found. The Pyro mirror (`d2hg8soec8ck9v.cloudfront.net/datasets/polyphonic/*.pickle`) returns HTTP 403. As a substitute, the four `.mat` files from locuslab/TCN (Bai, Kolter & Koltun 2018, commit `2f8c2b8`) hold the same piano-roll splits as 88-key binary matrices (`traindata/validdata/testdata`). Their split sizes match the original exactly (JSB 229/76/77 and Piano-midi 87/12/25 equal the Wayback pickles; Nottingham 694/173/170 equals the Wayback source-MIDI zip; MuseData 524/135/124 sums to the 783 files on the original page).
- Original train/valid/test splits are preserved in all of these files.
- Loading the pickles: they are Python 2 pickles that reference numpy scalars. `inventory.py` loads them with a restricted unpickler that allows only `numpy.dtype`, `numpy.ndarray` and `numpy.core.multiarray.{scalar,_reconstruct}`. Do not `pickle.load` them unrestricted.
- License: none stated for the pickles. The original page footnotes say: Piano-midi.de, see http://piano-midi.de/copy.htm; MuseData, see http://www.musedata.org/legal/lcr.html; Nottingham is also available in ABC format.

## 4. Essen Folksong Collection (**kern)

- Source: https://github.com/ccarh/essen-folksong-collection, commit `2d0ca75e87dc7a725556c8090e3681c1fa3a0452` (last push 2024-03-14). kern.humdrum.org returned HTTP 503 on the download date; the CCARH GitHub repo is the same collection maintained by the same group.
- Archive URL: https://codeload.github.com/ccarh/essen-folksong-collection/zip/2d0ca75e87dc7a725556c8090e3681c1fa3a0452
- Local: `raw/essen/essen-folksong-collection-2d0ca75.zip` (extracted alongside)
- Size: 6,099,414 bytes; 8,482 members (8,473 `.krn`)
- SHA-256: `3971a8be56e9f8f50903c1b3e17ed1c9e4218d979e32c395b700fba0aed086e0`
- License: `license.txt` in the repo, the CCARH MuseData license. Key clauses, verbatim:
  > (4) The enclosed MuseData files are for personal use (single-user)* only.

  > (5) MuseData files are made available only for the purpose of academic research;

  > (6) CCARH must be acknowledged as the source of data in any publications, computer programs or other products resulting from or involving the use of this data.

  `README.txt` also says "The accompanying files are protected by copyright and are distributed by license only." Citation given there: Schaffrath, Helmut. *The Essen Folksong Collection in Kern Format.* D. Huron (ed.). Menlo Park, CA: CCARH.
- EsAC original format (esac-data.org) timed out on the download date and was not fetched.

## 5. PDMX: Public Domain MusicXML (Long, Novack, McAuley, Berg-Kirkpatrick, ICASSP 2025)

- Zenodo record: https://zenodo.org/records/15571083 (DOI 10.5281/zenodo.15571083, published 2025-06-01). This is the latest version per `https://zenodo.org/api/records/15571083/versions/latest`. Record JSON saved as `zenodo_record_15571083.json` (SHA-256 `ce3d3f9cf6a657cd15d6cfec0375d832622881797885085f8725ab8e0477d285`).
- Download URL pattern: `https://zenodo.org/api/records/15571083/files/<file>/content`
- Every file's MD5 was checked against the checksum Zenodo publishes for it; all matched.
- Subset criterion: all symbolic files were taken (MusicXML, MIDI, MusicRender JSON, metadata, subset lists). Only `pdf.tar.gz` (9,621,915,440 bytes of rendered sheet music) was skipped. No scores were filtered out at download time; the official subsets are in `subset_paths/` and as `subset:*` columns in `PDMX.csv`. The authors recommend `no_license_conflict` (222,856 scores) because 31,221 scores (12.29%) have internal MuseScore license data that disagrees with the public website metadata.
- Archives were kept packed except `subset_paths.tar.gz` (extracted to `subset_paths/*.txt`).

| File | Bytes | Zenodo MD5 (verified) | SHA-256 | Members |
|---|---|---|---|---|
| `PDMX.csv` | 225,399,738 | `30392ccf38bb63ce70e7afae70f9c88c` | `fc2187e7e09185f4b28be57b6478a96a1243a037f8fd13624f17de2d8bfd44bd` | 254,077 rows |
| `subset_paths.tar.gz` | 29,258,714 | `092eee416ece8060f77d08575b94a43d` | `17529103f66af71bd313369029c0901a143049bf6fcaecdfd11f5ca588df4377` | 6 path lists |
| `metadata.tar.gz` | 159,444,765 | `5bc79445090dd2fe5e96cffa77a3461c` | `325b57ad69c34d45b05b8f307d88ebd10ff67554a7a9b4c183676e7586deffa3` | per-score metadata JSON |
| `mxl.tar.gz` | 1,894,335,797 | `49ffd75ecf5489c0be6d41182eb11ff7` | `01949cc0d9215e2f30924eadc8b1dd148aca90c02e2cf1999bb83df5ffe4d7a4` | 254,035 `.mxl` |
| `mid.tar.gz` | 214,395,208 | `d920a21b2fcd99a56d9c381b39debbb2` | `e444f9b466f02c9a054d31478c9886847f39c575a65ec45a0aaa1a5ee088c1d1` | 254,035 `.mid` |
| `data.tar.gz` | 2,237,580,506 | `f38dfa7b75f95e5a3d8d70459c1f9b72` | `d9f0221e564aaf0895f6d1b71e6201b5a79a7d8514a12b16bf8ec59370f2eca1` | 254,077 MusicRender `.json` |

- License: the Zenodo record itself is **CC-BY-4.0** (`metadata.license.id = "cc-by-4.0"`), which is not CC0. The per-score `license` column in `PDMX.csv` is `publicdomain` for 210,364 scores and `cc-zero` for 43,713. So the scores are public domain or CC0 as labelled on MuseScore, and the compiled dataset asks for attribution (cite Long et al., ICASSP 2025, and the authors' GitHub).
- 42 scores have no MXL/MIDI because the source files were corrupt (`all_valid` = 254,035).

## 6. Lakh MIDI Dataset (Raffel 2016)

- Project page: https://colinraffel.com/projects/lmd/ (saved as `lmd_project_page.html`, SHA-256 `85ebfe312ec304e586e531f83a2279a5c0958bc51025aa9c10a76db5e6713a0d`)
- License, from that page: "The Lakh MIDI Dataset is distributed with a CC-BY 4.0 license; if you use this data in any capacity, please reference this page and my thesis" (Colin Raffel, "Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching", PhD thesis, 2016).
- Archives were kept packed (the inventory lists them without extracting).

| File | URL | Bytes | SHA-256 |
|---|---|---|---|
| `lmd_full.tar.gz` | http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz | 1,768,163,879 | `6fcfe2ac49ca08f3f214cec86ab138d4fc4dabcd7f27f491a838dae6db45a12b` |
| `lmd_matched.tar.gz` | http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz | 1,407,072,670 | `621ff830aed771f469e5bfa13dc12a33c6ed69090adeda63d0b5c47783af0191` |
| `clean_midi.tar.gz` | http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz | 234,283,029 | `de1bb64cbc0cf35545a05b5c3e786aa6890cfa144edffc4b827ff41bf8c33dc5` |
| `md5_to_paths.json` | http://hog.ee.columbia.edu/craffel/lmd/md5_to_paths.json | 24,857,245 | `9002b7723f3edeca779e91688802fdd283b8df0c278162a4040f95bde5895805` |
| `match_scores.json` | http://hog.ee.columbia.edu/craffel/lmd/match_scores.json | 7,237,036 | `267bc606dfa21f0ad0601a4a080972cd4ae8088fe4003b9bb2811b5be060a102` |

- File counts: `lmd_full` 178,561 .mid (each named by its MD5, so files are already deduplicated by content); `lmd_matched` 116,189 .mid entries, which are the matched MIDIs repeated once per matched MSD track (`match_scores.json` lists 31,034 tracks); `clean_midi` 17,256 .mid. `md5_to_paths.json` maps each of the 178,561 MD5s to the original filenames it was scraped under, which is what you use to find duplicates and artist/title names.

## 7. POP909 (Wang et al., ISMIR 2020)

- Source: https://github.com/music-x-lab/POP909-Dataset, commit `d83e6edba6872a704f5d3b8b32f5cb540088dae6`
- Archive URL: https://codeload.github.com/music-x-lab/POP909-Dataset/zip/d83e6edba6872a704f5d3b8b32f5cb540088dae6
- Local: `raw/pop909/POP909-Dataset-d83e6ed.zip` (extracted alongside; the repo also contains its own `POP909.zip`, 59,832,526 bytes uncompressed / 9,263 entries, which duplicates the `POP909/` folder)
- Size: 50,105,665 bytes; 7,450 members
- SHA-256: `fe3d861f6eeaee6b40987e64ae26552b4c7b824f6fa40d29b0c5e5ebd62b03ea`
- 909 songs; every main `.mid` has tracks `MELODY`, `BRIDGE`, `PIANO`, so the melody is separable by track name. 1,989 alternate arrangements under `*/versions/`. Beat, chord and key annotations per song.
- License: MIT (`LICENSE`, "Copyright (c) 2020 Music X Lab"). The songs themselves are copyrighted Chinese pop; MIT covers the authors' arrangements and annotations.

## 8. IrishMAN (Wu et al., TunesFormer, 2023)

- Source: https://huggingface.co/datasets/sander-wood/irishman, revision `30902e69ca45266207f8466e0d04e4bc742c5604` (last modified 2024-03-16). Not gated; downloaded through `https://huggingface.co/datasets/sander-wood/irishman/resolve/<rev>/<file>` with no token.

| File | Bytes | SHA-256 |
|---|---|---|
| `train.json` | 79,975,421 | `5706880c0f935e69108244e010784935b7aba855f37ae2ce7f4e4d170851835c` |
| `validation.json` | 796,938 | `a33935f9047edc976e087b146737ed3737a9315c2bf52ff02d8873b057c575ae` |
| `leadsheet_ids.json` | 323,873 | `3862574a719d8e78f861f4039d34d5d75fc28a2f23095d3cf7cd2ed3b2fd25a9` |
| `variation_ids.json` | 300,468 | `7ae85c124d834cab88cab94bc0117e2f3cf50049caa5ffeeb49b05a3ba2e4cda` |
| `irishman-midi.zip` | 98,587,670 | `029a7661260bed87c7b392507efb48898f80eb249645ad418156727c92469e46` |
| `irishman-xml.zip` | 398,633,100 | `76e1249cfc59d96509e7f6b10aded15f5e7a780bb5b091a3440a1152d9a20557` |
| `README.md` | 6,268 | `0bfd3aa9830f001d2d69e42112aa61f394240a7f9b4ba0428ea52fb4d91a8235` |

- The three LFS files' SHA-256 match the oids reported by the HF API tree for that revision.
- License: `license: mit` in the card metadata. The card also says: "This dataset is for research use only and not for commercial purposes. We believe all data in this dataset is in the public domain."
- Provenance caveat: the card says the tunes "were collected from thesession.org and abcnotation.com". The collection predates The Session's 2025-10-08 LLM clause (section 1), when The Session data was plain ODbL, and titles were stripped. Whether the later clause reaches this derivative is unclear; treat it as the same question as section 1.

## 9. Hooktheory lead sheets (Sheet Sage release, Donahue, Thickstun & Liang 2022)

- Source: https://github.com/chrisdonahue/sheetsage-data, commit `06113c04b109a2f27517b0399ff47550099f2466`, files under `hooktheory/`, fetched as `https://github.com/chrisdonahue/sheetsage-data/raw/<commit>/hooktheory/<file>`. Description and license from https://github.com/chrisdonahue/sheetsage README at commit `bbdd7b7b6a5fb845828f82790acdceb03a197779` (saved as `sheetsage_README.md`, SHA-256 `693760246bac98ccbc5f73ecd3757e6c440ef2fbc677bff82e4b432cc3080227`).
- License, from that README: the dataset is released "under a CC BY-NC-SA 3.0 license" (https://creativecommons.org/licenses/by-nc-sa/3.0/). No audio is included.
- All SHA-256 values below match the hashes published in the Sheet Sage README.

| File | Bytes | SHA-256 |
|---|---|---|
| `Hooktheory.json.gz` | 20,075,896 | `917b7cd58f5f4e07d6c36acf7bfad958c99ee05472dab3555399141094698e0c` |
| `Hooktheory_Train_Segments.json` | 1,831,047 | `f2601eb544f2e5028ffad54d3827912865578fb9ad96e6768b35e4714d5c7207` |
| `Hooktheory_Train_MIDI.tar.gz` | 2,700,002 | `a2345e13564c81740c087b79731b47e8323c4ceb7b85d40a1a11582af58145cb` |
| `Hooktheory_Valid_Segments.json` | 180,037 | `12526962f77c2eb41cd117c8effa678b39c2b350384a7b048b327aff287b0c48` |
| `Hooktheory_Valid_MIDI.tar.gz` | 266,989 | `e369fd4a3072c7524e3cafe506bbdad6e908de969d0f1ba7abf08bf5148989fe` |
| `Hooktheory_Test_Segments.json` | 200,130 | `72be80045d4d28842352383e605e8712d50b3437a07b15faa541ee9d17283d5a` |
| `Hooktheory_Test_MIDI.tar.gz` | 296,058 | `3baebe9d4e19a5006d0f24bc7f0c92a4f66039ab376be86bf7b37a136d4fb6c8` |

- Not fetched: `Hooktheory_Raw.json.gz` (the raw proprietary HookTheory format; auxiliary).

## 10. Skipped

- **Wikifonia:** the site shut down in 2013 over licensing. Copies that circulate (e.g. MusicXML zips on personal sites) have no clear license, so none was downloaded.
- **Hooktheory TheoryTab API / site data:** requires an account and API key. The Sheet Sage release above is the legal no-login route.
- **PDMX `pdf.tar.gz`** (9.6 GB of rendered sheet music): not symbolic, skipped.
- **The Session LFS files** (`sets.json`, `thesession.db`): see section 1.
- **EsAC-format Essen** (esac-data.org): host timed out.

## Inventory output

Command: `.venv/bin/python -I scripts/inventory.py` (run 2026-10-08). Session ABC bodies intentionally not printed; see section 1.

````text
# Inventory of /Users/takakhoo/Dev/melody-research/data/raw

## The Session (adactio/TheSession-data)
settings (rows in tunes.csv): 55474
unique tunes (tune_id): 23325
tunes by type: {'reel': 8077, 'jig': 6052, 'waltz': 2038, 'polka': 1715, 'hornpipe': 1583, 'slip jig': 813, 'march': 795, 'barndance': 718, 'strathspey': 627, 'slide': 472, 'mazurka': 256, 'three-two': 179}
example tunes (first setting of tune_id 1, 2, 3):
  tune_id=1 setting_id=21423 name="Cooley's" type=reel meter=4/4 mode=Eminor abc_chars=192
  tune_id=2 setting_id=12344 name='Bucks Of Oranmore, The' type=reel meter=4/4 mode=Dmajor abc_chars=1637
  tune_id=3 setting_id=49093 name='Boil The Breakfast Early' type=reel meter=4/4 mode=Gmajor abc_chars=284

## Nottingham Music Database
jukedeck ABC_original: 14 files, 1037 tunes
jukedeck ABC_cleaned: 14 files, 1034 tunes
jukedeck MIDI: 3089 .mid files; by dir (. = full arrangement): {'.': 1034, 'melody': 1034, 'chords': 1021}
Seymour Shlien edit (nottingham_database.zip): 1037 tunes
example tunes (jukedeck ABC_cleaned/jigs.abc, first 3):
~~~abc
X: 1
T:A and D
% Nottingham Music Database
S:EF
Y:AB
M:4/4
K:A
M:6/8
P:A
f|"A"ecc c2f|"A"ecc c2f|"A"ecc c2f|"Bm"BcB "E7"B2f|
"A"ecc c2f|"A"ecc c2c/2d/2|"D"efe "E7"dcB| [1"A"Ace a2:|
 [2"A"Ace ag=g||
K:D
P:B
"D"f2f Fdd|"D"AFA f2e/2f/2|"G"g2g ecd|"Em"efd "A7"cBA|
"D"f^ef dcd|"D"AFA f=ef|"G"gfg "A7"ABc |[1"D"d3 d2e:|[2"D"d3 d2||
~~~
~~~abc
X: 2
T:Abacus
% Nottingham Music Database
S:By Hugh Barwell, via Phil Rowe
M:6/8
K:G
"G"g2g B^AB|d2d G3|"Em"GAB "Am"A2A|"D7"ABc "G"BAG|
"G"g2g B^AB|d2d G2G|"Em"GAB "Am"A2G|"D7"FGA "G"G3:||:
"D7"A^GA DFA|"G"B^AB G3|"A7"^c=c^c A^ce|"D7"fef def|
"G"g2g de=f|"E7"e2e Bcd|"Am"c2c "D7"Adc| [1"G"B2A G3:|
 [2"G"B2A G2F||"Em"E2E G2G|B2B e2e|"Am"c2A "B7"FBA|"Em"G2F E3|"Em"EFG "Am"ABc|
"B7"B^c^d "Em"e2e|"F#7"f2f f2e|"B7"^def BAF|"Em"E2E G2G|B2B e2e|
"Am"c2A "B7"FBA|"Em"G2F E3|"Em"EFG "Am"ABc|"B7"B^c^d "Em"e2e|
"F#7"f2e "B7"^def |[1"Em"e3 "D7"d3:|[2"Em"e3 "E7"e3||
~~~
~~~abc
X: 3
T:The American Dwarf
% Nottingham Music Database
S:FTB, via EF
M:6/8
K:D
A|"D" def fed|"G" BdB AFD| "D"DFA "G"B2 A|"Em" cee "A7" e2 A|
"D" def fed|"G" BdB "D"AFD|"D" DFA "G"B2 A|"A7" Add "D" d2:|
"B"e|"D"fga agf|"G" gab "A7"bag|"D"fga "D"agf|"Em" gfg "A7"e2 g|
"D"fga agf|"G"gab "A7"bag|"D" fga "A7"efg|"D" fdd d2 :|
~~~

## Boulanger-Lewandowski et al. 2012 piano-roll splits
wayback_JSB Chorales.pickle: test=77, train=229, valid=76
wayback_Piano-midi.de.pickle: test=25, train=87, valid=12
tcn_JSB_Chorales.mat: traindata=229, validdata=76, testdata=77
tcn_MuseData.mat: traindata=524, validdata=135, testdata=124
tcn_Nottingham.mat: traindata=694, validdata=173, testdata=170
tcn_Piano_midi.mat: traindata=87, validdata=12, testdata=25
wayback_Nottingham.zip (source MIDI): {'test': 170, 'valid': 173, 'train': 694}

## Essen Folksong Collection (**kern)
.krn files: 8473; by top-level dir: {'europa': 6213, 'asia': 2246, 'america': 13, 'africa': 1}

## PDMX
PDMX.csv rows (scores): 254077; license field: {'publicdomain': 210364, 'cc-zero': 43713}
subset flags: {'subset:all': 254077, 'subset:rated': 14182, 'subset:deduplicated': 102635, 'subset:rated_deduplicated': 13187, 'subset:no_license_conflict': 222856, 'subset:all_valid': 254035}
mxl.tar.gz: {'.mxl': 254035, '_files': 254035}
mid.tar.gz: {'.mid': 254035, '_files': 254035}
data.tar.gz: {'.json': 254077, '_files': 254077}

## Lakh MIDI Dataset
lmd_full.tar.gz: 178561 .mid files (178561 files total)
lmd_matched.tar.gz: 116189 .mid files (116189 files total)
clean_midi.tar.gz: 17256 .mid files (17259 files total)
md5_to_paths.json: 178561 md5 entries
match_scores.json: 31034 MSD tracks, 116189 (track, midi) pairs

## POP909
songs: 909; main .mid: 909; alternate versions .mid: 1989
track names across main .mid: {'MELODY': 909, 'BRIDGE': 909, 'PIANO': 909}

## IrishMAN
train.json: 214122 entries
validation.json: 2162 entries
leadsheet_ids.json: train=33897, validation=314
variation_ids.json: 4331 entries
irishman-midi.zip: {'.mid': 216266}
irishman-xml.zip: {'.xml': 216281}

## Hooktheory (Sheet Sage release)
annotations: 26175; by split: {'TRAIN': 21230, 'TEST': 2761, 'VALID': 2184}; with MELODY tag: 23833
````
