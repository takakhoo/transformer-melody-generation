# Literature map: symbolic melody generation, 2018-2026

Compiled 8 October 2026 for the melody leakage/memorization paper (target venue in [venue.md](venue.md), data provenance in [../data/SOURCES.md](../data/SOURCES.md)). BibTeX for every entry marked VERIFIED is in [refs.bib](refs.bib).

## How entries were checked

- **VERIFIED** means title, full author list, and year were read from a primary record during this session: the arXiv API (`export.arxiv.org`), Crossref (DOI, venue, pages), a publisher or proceedings page (USENIX, PMLR, ICLR proceedings, NeurIPS virtual site, mlanthology mirror of ICLR/TMLR), or the dataset's own page.
- Venue is stated as "accepted at X (arXiv comment)" when the only evidence is the authors' arXiv comment. Pages are given only when Crossref or a proceedings page returned them; "pages not checked" means the venue is confirmed but the page range was not.
- **UNVERIFIED** means the item exists in secondary sources but I could not confirm it from a primary record. Nothing marked UNVERIFIED is in `refs.bib`.
- DBLP and OpenReview served bot-check pages to the tools, so they were not used. OpenReview IDs below come from proceedings pages or search results that link to them.
- Model sizes, data, and numbers in the notes come from the paper abstracts or pages fetched in this session. Where a detail is from my own memory and was not re-read, it says "(not re-checked)".

Within each cluster, entries run newest first. Classic anchors sit at the end of the cluster.

---

## 1. Model families for symbolic music

### 2026

- **YuE2: Unifying Symbolic and Audio Music Generation at Frontier Quality.** Ruibin Yuan, Jiahao Pan, Junyan Jiang, Zhiyue Wu, Ziya Zhou, Jiankai Sun, Yizhi Li, Ge Zhang, Yicheng Gu, Zeyue Tian, Junyu Dai, Hanfeng Lin, Kai Li, Shangda Wu, Xuanjie Liu, Jiaming Wang, Zihan Liu, Yue Wang, Yinghao Ma, Hanzhi Yin, Kangrui Chen, Xinyue Zhang, Ziyang Ma, Mengqi Liao, Hejia Zhao, Guowei Huang, Chao Yan, Lei Ke, Jianwei Yu, Bei Liu, Joe Guo, Liumeng Xue, Gus Xia, Wei Xue, Yike Guo. 2026. Technical report. arXiv:2609.33757. VERIFIED.
  One AR-NAR mixture-of-transformers writes a readable score (melody + harmony), then semantic tokens, then audio. Introduces SheetSage2 for lead-sheet transcription as symbolic supervision. Relevant as the first frontier-scale system that plans in lead-sheet space; it is an audio product, and its symbolic stage is trained on transcriptions.
- **Text2Score: Generating Sheet Music From Textual Prompts.** Keshav Bhandari, Sungkyun Chang, Abhinaba Roy, Francesca Ronchini, Emmanouil Benetos, Dorien Herremans, Simon Colton. 2026. arXiv:2605.13431. VERIFIED.
  LLM planner produces a bar-wise plan; a generator writes interleaved ABC. Supervision derived from symbolic XML rather than captions. Evaluates playability, readability, prompt adherence, plus expert ratings.
- **RPPNet: Perceptually-Grouped Rhythm-Pitch Primitives for Long-Term Structure Melody Generation via Boundary-Aware Modeling.** Tieyao Zhang, Yuke Liu, Jiaxing Yu, Xinda Wu, Kejun Zhang, Genfang Chen. 2026. arXiv:2607.19776. VERIFIED (preprint, no venue).
- **Aligning Language Models for Lyric-to-Melody Generation with Rule-Based Musical Constraints.** Hao Meng, Siyuan Zheng, Shuran Zhou, Qiangqiang Wang, Yang Song. 2026. Accepted at ICASSP 2026 (arXiv comment). arXiv:2604.18489. VERIFIED.
- **Efficient Long-Sequence Diffusion Modeling for Symbolic Music Generation.** Jinhan Xu, Xing Tang, Houpeng Yang, Haoran Zhang, Shenghua Yuan, Jiatao Chen, Tianming Xi, Jing Wang, Jiaojiao Yu, Guangli Xiang. 2026. arXiv:2603.00576. VERIFIED.
  Replaces the withdrawn arXiv:2507.20128 (SMDIM, Mamba layers inside discrete diffusion; that version's comment says to disregard it). Cite 2603.00576.
- **D3PIA: A Discrete Denoising Diffusion Model for Piano Accompaniment Generation From Lead sheet.** Eunjin Choi, Hounsu Kim, Hayeon Bang, Taegyun Kwon, Juhan Nam. 2026. Accepted at ICASSP 2026 (arXiv comment). arXiv:2602.03523. VERIFIED.

### 2025

- **MIDI-LLM: Improving Text-to-MIDI Music Generation via Adapting Large Language Models.** Shih-Lun Wu, Dave Carlton, Ryan Miyakawa, Yoon Kim, Chris Donahue, Cheng-Zhi Anna Huang. 2025 (v2). Accepted at ISMIR 2026 (arXiv comment). arXiv:2511.03942. VERIFIED.
  Llama 3.2 1B with MIDI tokens added to the vocabulary; continued pretraining on music text + standalone MIDI, then SFT on text-MIDI pairs. Beats text2midi on text control and quality. Fine-tuned on TheoryTab (Hooktheory) for **text-conditioned lead-sheet generation and infilling**; in-the-wild study with 58 Hookpad Aria users and 4,002 outputs. This is the strongest published lead-sheet generator with a real-user evaluation that I found.
- **Chord-conditioned Melody and Bass Generation.** Alexandra C Salem, Mohammad Shokri, Johanna Devaney. 2025. NeurIPS 2025 Workshop on AI for Music (arXiv comment). arXiv:2511.08755. VERIFIED.
- **Versatile Symbolic Music-for-Music Modeling via Function Alignment.** Junyan Jiang, Daniel Chin, Liwei Lin, Xuanjie Liu, Gus Xia. 2025. ISMIR 2025 (arXiv journal-ref). arXiv:2506.15548. VERIFIED.
  Two pretrained LMs (reference and target) joined by a light adapter. Tasks include chord recognition and chord-conditioned melody generation.
- **Scaling Self-Supervised Representation Learning for Symbolic Piano Performance** (the "Aria" model). Louis Bradshaw, Honglu Fan, Alexander Spangher, Stella Biderman, Simon Colton. 2025. ISMIR 2025 (arXiv comment). arXiv:2506.23869. VERIFIED.
  Autoregressive transformer pretrained on about 60,000 h of solo-piano transcriptions (Aria-MIDI), then fine-tuned for continuation, classification, and SimCLR-style embeddings. Piano performance, so tangential to melody, but it is the current reference for scale in symbolic pretraining.
- **TOMI: Transforming and Organizing Music Ideas for Multi-Track Compositions with Full-Song Structure.** Qi He, Gus Xia, Ziyu Wang. 2025. ISMIR 2025 (arXiv comment). arXiv:2506.23094. VERIFIED.
- **Moonbeam: A MIDI Foundation Model Using Both Absolute and Relative Music Attributes.** Zixun Guo, Simon Dixon. 2025. arXiv:2505.15559. VERIFIED (preprint; no venue found).
  Pretrained on 81.6K hours of MIDI, 18B tokens. Tokenizer and "Multidimensional Relative Attention" encode both absolute and relative pitch/time attributes. Fine-tuned for classification and conditional generation/infilling.
- **Mamba-Diffusion Model with Learnable Wavelet for Controllable Symbolic Music Generation.** Jincheng Zhang, György Fazekas, Charalampos Saitis. 2025. arXiv:2505.03314. VERIFIED (preprint).
- **NotaGen: Advancing Musicality in Symbolic Music Generation with Large Language Model Training Paradigms.** Yashan Wang, Shangda Wu, Jianhuai Hu, Xingjian Du, Yueqi Peng, Yongxin Huang, Shuai Fan, Xiaobing Li, Feng Yu, Maosong Sun. 2025. IJCAI 2025, pp. 10207-10215. DOI 10.24963/ijcai.2025/1134. arXiv:2502.18008. VERIFIED.
  Pretrained on 1.6M ABC pieces, fine-tuned on about 9K classical works with period-composer-instrumentation prompts, then CLaMP-DPO (reward from CLaMP 2 similarity, no human labels). Evaluation: subjective A/B tests against human compositions. (Crossref also lists a stray "IJCAI 2024" record with the same DOI suffix; the 2025 record is correct.)
- **CLaMP 3: Universal Music Information Retrieval Across Unaligned Modalities and Unseen Languages.** Shangda Wu, Zhancheng Guo, Ruibin Yuan, Junyan Jiang, Seungheon Doh, Gus Xia, Juhan Nam, Xiaobing Li, Feng Yu, Maosong Sun. 2025. Findings of ACL 2025, pp. 2605-2625. DOI 10.18653/v1/2025.findings-acl.133. arXiv:2502.10362. VERIFIED.
  Retrieval rather than generation, but CLaMP 3 embeddings were the best duplicate retriever in the Lakh de-duplication study (cluster 4).
- **MIDI-GPT: A Controllable Generative Model for Computer-Assisted Multitrack Music Composition.** Philippe Pasquier, Jeff Ens, Nathan Fradet, Paul Triana, Davide Rizzotti, Jean-Baptiste Rolland, Maryam Safi. 2025. AAAI 2025, 39(2):1474-1482. DOI 10.1609/aaai.v39i2.32138. arXiv:2501.17011. VERIFIED.
- **Text2midi: Generating Symbolic Music from Captions.** Keshav Bhandari, Abhinaba Roy, Kyra Wang, Geeta Puri, Simon Colton, Dorien Herremans. 2025. AAAI 2025, 39(22):23478-23486. DOI 10.1609/aaai.v39i22.34516. arXiv:2412.16526. VERIFIED.
  Pretrained LLM text encoder conditioning an autoregressive MIDI decoder. Follow-up: Text2midi-InferAlign (Roy, Puri, Herremans, arXiv:2505.12669, VERIFIED).
- **CLaMP 2: Multimodal Music Information Retrieval Across 101 Languages Using Large Language Models.** Shangda Wu, Yashan Wang, Ruibin Yuan, Zhancheng Guo, Xu Tan, Ge Zhang, Monan Zhou, Jing Chen, Xuefeng Mu, Yuejie Gao, Yuanliang Dong, Jiafeng Liu, Xiaobing Li, Feng Yu, Maosong Sun. 2025. Findings of NAACL 2025, pp. 435-451. DOI 10.18653/v1/2025.findings-naacl.27. arXiv:2410.13267. VERIFIED.
- **MusicMamba: A Dual-Feature Modeling Approach for Generating Chinese Traditional Music with Modal Precision.** Jiatao Chen, Tianming Xie, Xing Tang, Jing Wang, Wenjing Dong, Bing Shi. 2025. ICASSP 2025 (arXiv comment). arXiv:2409.02421. VERIFIED.
  Mamba applied to melody generation. The only state-space melody generator with a peer-reviewed venue that I found.
- **MuPT: A Generative Symbolic Music Pretrained Transformer.** Xingwei Qu, Yuelin Bai, Yinghao Ma, Ziya Zhou, Ka Man Lo, Jiaheng Liu, Ruibin Yuan, Lejun Min, Xueling Liu, Tianyu Zhang, Xinrun Du, Shuyue Guo, Yiming Liang, Yizhi Li, Shangda Wu, Junting Zhou, Tianyu Zheng, Ziyang Ma, Fengze Han, Wei Xue, Gus Xia, Emmanouil Benetos, Xiang Yue, Chenghua Lin, Xu Tan, Stephen W. Huang, Jie Fu, Ge Zhang. 2025. ICLR 2025 (ICLR proceedings page). arXiv:2404.06393. VERIFIED.
  Argues ABC suits LLMs better than MIDI; proposes Synchronized Multi-Track ABC (SMT-ABC); context up to 8,192 tokens (covers 90% of training pieces); fits a "Symbolic Music Scaling (SMS) law" by modifying Chinchilla for repeated epochs on limited data. Evaluation: intra-similarity, repetition rate, listener preference vs GPT-4.
- **Small Tunes Transformer: Exploring Macro & Micro-Level Hierarchies for Skeleton-Conditioned Melody Generation.** Yishan Lv, Jing Luo, Boyuan Ju, Xinyu Yang. 2025. MMM 2025 (arXiv comment). arXiv:2410.08626. VERIFIED.
- **CSL-L2M: Controllable Song-Level Lyric-to-Melody Generation Based on Conditional Transformer with Fine-Grained Lyric and Musical Controls.** Li Chai, Donglin Wang. 2025. AAAI 2025 (arXiv comment). arXiv:2412.09887. VERIFIED.
- **SongComposer: A Large Language Model for Lyric and Melody Generation in Song Composition.** Shuangrui Ding, Zihan Liu, Xiaoyi Dong, Pan Zhang, Rui Qian, Junhao Huang, Conghui He, Dahua Lin, Jiaqi Wang. 2025. ACL 2025 (Long), pp. 7108-7127. DOI 10.18653/v1/2025.acl-long.352. arXiv:2402.17645. VERIFIED.

### 2024

- **Hookpad Aria: A Copilot for Songwriters.** Chris Donahue, Shih-Lun Wu, Yewon Kim, Dave Carlton, Ryan Miyakawa, John Thickstun. ISMIR 2024 Late-Breaking Demo (arXiv comment); arXiv posted 2025. arXiv:2502.08122. VERIFIED.
  Deployed lead-sheet copilot (continuation, infilling, melody-to-harmony and back). Reports 318k suggestions, 3k users, 74k accepted since March 2024. Acceptance rate is the real-world metric MIDI-LLM later reuses.
- **MelodyT5: A Unified Score-to-Score Transformer for Symbolic Music Processing.** Shangda Wu, Yashan Wang, Xiaobing Li, Feng Yu, Maosong Sun. ISMIR 2024 (arXiv comment). arXiv:2407.02277. VERIFIED.
  Encoder-decoder over ABC; seven melody-centric tasks (generation, harmonization, segmentation, ...) in one model. Pretrained on MelodyHub, "over 261K unique melodies" (the abstract says unique; how uniqueness was established is a question for cluster 4).
- **Nested Music Transformer: Sequentially Decoding Compound Tokens in Symbolic Music and Audio Generation.** Jiwoo Ryu, Hao-Wen Dong, Jongmin Jung, Dasaem Jeong. ISMIR 2024, pp. 588-595 (pages from the Sogang University repository record). arXiv:2408.01180. VERIFIED.
  First-author mismatch: the current arXiv metadata (v2) lists "HaeJun Yoo" first, while arXiv v1 and the ISMIR 2024 proceedings index list Jiwoo Ryu. Possibly a name change. The `.bib` follows the proceedings; check with the authors before camera-ready.
- **Whole-Song Hierarchical Generation of Symbolic Music Using Cascaded Diffusion Models.** Ziyu Wang, Lejun Min, Gus Xia. ICLR 2024. arXiv:2405.09901. VERIFIED.
  Evaluates against "copy-bot" baselines that deliberately plagiarize training pieces, one of the few generation papers that builds copying into the evaluation design.
- **ChatMusician: Understanding and Generating Music Intrinsically with LLM.** Ruibin Yuan, Hanfeng Lin, Yi Wang, Zeyue Tian, Shangda Wu, Tianhao Shen, Ge Zhang, Yuhang Wu, Cong Liu, Ziya Zhou, Ziyang Ma, Liumeng Xue, Ziyu Wang, Qin Liu, Tianyu Zheng, Yizhi Li, Yinghao Ma, Yiming Liang, Xiaowei Chi, Ruibo Liu, Zili Wang, Pengfei Li, Jingcheng Wu, Chenghua Lin, Qifeng Liu, Tao Jiang, Wenhao Huang, Wenhu Chen, Emmanouil Benetos, Jie Fu, Gus Xia, Roger Dannenberg, Wei Xue, Shiyin Kang, Yike Guo. Findings of ACL 2024, pp. 6252-6271. DOI 10.18653/v1/2024.findings-acl.373. arXiv:2402.16153. VERIFIED.
  LLaMA2 continually pretrained and fine-tuned on ABC with the plain text tokenizer; releases MusicPile (4B tokens) and MusicTheoryBench.
- **Symbolic Music Generation with Non-Differentiable Rule Guided Diffusion.** Yujia Huang, Adishree Ghatare, Yuanzhe Liu, Ziniu Hu, Qinsheng Zhang, Chandramouli S Sastry, Siddharth Gururani, Sageev Oore, Yisong Yue. ICML 2024 (oral; arXiv comment). arXiv:2402.14285. VERIFIED.
- **Anticipatory Music Transformer.** John Thickstun, David Hall, Chris Donahue, Percy Liang. Transactions on Machine Learning Research, 2024. OpenReview EBNJ33Fcrl. arXiv:2306.08620. VERIFIED.
  Interleaves event and control tokens so the model can infill and accompany. Trained on Lakh MIDI. Human evaluators rated anticipatory accompaniments over 20-s clips as similar in musicality to human-composed ones. Released models are on Hugging Face under Apache 2.0. Model sizes up to several hundred million parameters (not re-checked).

### 2023

- **Unsupervised Lead Sheet Generation via Semantic Compression.** Zachary Novack, Nikita Srivatsan, Taylor Berg-Kirkpatrick, Julian McAuley. 2023. arXiv:2310.10772. VERIFIED (preprint).
- **MelodyGLM: Multi-task Pre-training for Symbolic Melody Generation.** Xinda Wu, Zhijie Huang, Kejun Zhang, Jiaxing Yu, Xu Tan, Tieyao Zhang, Zihao Wang, Lingyun Sun. 2023. arXiv:2309.10738. VERIFIED (preprint).
- **Polyffusion: A Diffusion Model for Polyphonic Score Generation with Internal and External Controls.** Lejun Min, Junyan Jiang, Gus Xia, Jingwei Zhao. ISMIR 2023. arXiv:2307.10304. VERIFIED.
- **MuseCoco: Generating Symbolic Music from Text.** Peiling Lu, Xin Xu, Chenfei Kang, Botao Yu, Chengyi Xing, Xu Tan, Jiang Bian. 2023. arXiv:2306.00110. VERIFIED (preprint).
  Two stages: text-to-attribute, attribute-to-music (self-supervised). Largest model 1.2B parameters.
- **GETMusic: Generating Any Music Tracks with a Unified Representation and Diffusion Framework.** Ang Lv, Xu Tan, Peiling Lu, Wei Ye, Shikun Zhang, Jiang Bian, Rui Yan. 2023. arXiv:2305.10841. VERIFIED (preprint).
- **CLaMP: Contrastive Language-Music Pre-training for Cross-Modal Symbolic Music Information Retrieval.** Shangda Wu, Dingyao Yu, Xu Tan, Maosong Sun. ISMIR 2023. arXiv:2304.11029. VERIFIED.
- **TunesFormer: Forming Irish Tunes with Control Codes by Bar Patching.** Shangda Wu, Xiaobing Li, Feng Yu, Maosong Sun. HCMIR 2023 workshop (arXiv comment). arXiv:2301.02884. VERIFIED.
  Dual-decoder transformer with bar patching (shorter sequences) and form control codes. Trained on 214,122 Irish tunes, which is exactly the IrishMAN train split (cluster 6). Compared with GPT-2 on speed and controllability.
- **Multitrack Music Transformer.** Hao-Wen Dong, Ke Chen, Shlomo Dubnov, Julian McAuley, Taylor Berg-Kirkpatrick. ICASSP 2023. arXiv:2207.06983. VERIFIED.
- **FIGARO: Generating Symbolic Music with Fine-Grained Artistic Control.** Dimitri von Rütte, Luca Biggio, Yannic Kilcher, Thomas Hofmann. ICLR 2023. arXiv:2201.10936. VERIFIED.
  Self-supervised description-to-sequence; REMI+ token output (not re-checked); evaluated on control fidelity.
- **Compose & Embellish: Well-Structured Piano Performance Generation via A Two-Stage Approach.** Shih-Lun Wu, Yi-Hsuan Yang. ICASSP 2023. arXiv:2209.08212. VERIFIED.
  Generates a lead sheet first, then embellishes it into piano. Useful precedent for melody-first pipelines.
- **Theme Transformer: Symbolic Music Generation With Theme-Conditioned Transformer.** Yi-Jen Shih, Shih-Lun Wu, Frank Zalkow, Meinard Müller, Yi-Hsuan Yang. IEEE Transactions on Multimedia 25:3495-3508, 2023. DOI 10.1109/TMM.2022.3161851. arXiv:2111.04093. VERIFIED.

### 2018-2022

- **Museformer: Transformer with Fine- and Coarse-Grained Attention for Music Generation.** Botao Yu, Peiling Lu, Rui Wang, Wei Hu, Xu Tan, Wei Ye, Shikun Zhang, Tao Qin, Tie-Yan Liu. NeurIPS 2022. arXiv:2210.10349. VERIFIED.
- **Melody transcription via generative pre-training.** Chris Donahue, John Thickstun, Percy Liang. ISMIR 2022. arXiv:2212.01884. VERIFIED. Source of the Hooktheory dataset release (cluster 6).
- **Symbolic Music Generation with Diffusion Models.** Gautam Mittal, Jesse Engel, Curtis Hawthorne, Ian Simon. ISMIR 2021. arXiv:2103.16091. VERIFIED.
- **Compound Word Transformer: Learning to Compose Full-Song Music over Dynamic Directed Hypergraphs.** Wen-Yi Hsiao, Jen-Yu Liu, Yin-Cheng Yeh, Yi-Hsuan Yang. AAAI 2021, 35(1):178-186. DOI 10.1609/aaai.v35i1.16091. arXiv:2101.02402. VERIFIED.
  Groups co-occurring tokens into compound words with per-type output heads; shortens sequences relative to REMI.
- **Pop Music Transformer: Beat-based Modeling and Generation of Expressive Pop Piano Compositions.** Yu-Siang Huang, Yi-Hsuan Yang. ACM Multimedia 2020, pp. 1180-1188. DOI 10.1145/3394171.3413671. arXiv:2002.00212. VERIFIED.
  Introduces REMI (bar/position/tempo/chord tokens). Transformer-XL backbone (not re-checked).
- **The Jazz Transformer on the Front Line: Exploring the Shortcomings of AI-composed Music through Quantitative Measures.** Shih-Lun Wu, Yi-Hsuan Yang. ISMIR 2020. arXiv:2008.01307. VERIFIED. Trained on the Weimar Jazz Database; defines pitch-class histogram entropy and grooving-pattern similarity (cluster 3).
- **MuseNet.** Christine Payne. OpenAI blog, 25 April 2019. https://openai.com/blog/musenet/ (redirects to openai.com/index/musenet). VERIFIED via OpenAI's own suggested citation as quoted by search results; the page itself was not fetched.
  72-layer transformer, 24 heads, 4,096-token context with Sparse Transformer kernels; trained on "hundreds of thousands of MIDI files". No paper, no released weights, no held-out evaluation.
- **Music Transformer: Generating Music with Long-Term Structure.** Cheng-Zhi Anna Huang, Ashish Vaswani, Jakob Uszkoreit, Ian Simon, Curtis Hawthorne, Noam Shazeer, Andrew M. Dai, Matthew D. Hoffman, Monica Dinculescu, Douglas Eck. ICLR 2019. OpenReview rJe4ShAcF7. arXiv:1809.04281. VERIFIED.
  Memory-efficient relative self-attention (skewing). Reports NLL on JSB Chorales and Piano-e-Competition/MAESTRO (not re-checked). arXiv lists Shazeer third; the ICLR record lists Simon and Hawthorne before Shazeer. The `.bib` uses the ICLR order.

### Classic anchors

- **Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset.** Curtis Hawthorne, Andriy Stasyuk, Adam Roberts, Ian Simon, Cheng-Zhi Anna Huang, Sander Dieleman, Erich Elsen, Jesse Engel, Douglas Eck. ICLR 2019 (not re-checked). arXiv:1810.12247. VERIFIED (arXiv).
- **MuseGAN: Multi-track Sequential Generative Adversarial Networks for Symbolic Music Generation and Accompaniment.** Hao-Wen Dong, Wen-Yi Hsiao, Li-Chia Yang, Yi-Hsuan Yang. AAAI 2018. arXiv:1709.06298. VERIFIED.
- **DeepBach: a Steerable Model for Bach Chorales Generation.** Gaëtan Hadjeres, François Pachet, Frank Nielsen. ICML 2017, PMLR 70:1362-1371. arXiv:1612.01010. VERIFIED. Includes a plagiarism analysis (longest subsequence copied from the training set).
- **Counterpoint by Convolution.** Cheng-Zhi Anna Huang, Tim Cooijmans, Adam Roberts, Aaron Courville, Douglas Eck. ISMIR 2017. arXiv:1903.07227. VERIFIED. Coconet; trained and evaluated on JSB Chorales.
- **C-RNN-GAN: Continuous recurrent neural networks with adversarial training.** Olof Mogren. NIPS 2016 Constructive Machine Learning workshop. arXiv:1611.09904. VERIFIED. Early source of "scale consistency" and related objective metrics.
- **Modeling Temporal Dependencies in High-Dimensional Sequences: Application to Polyphonic Music Generation and Transcription.** Nicolas Boulanger-Lewandowski, Yoshua Bengio, Pascal Vincent. ICML 2012. arXiv:1206.6392. VERIFIED (pages not checked). Source of the standard Nottingham / JSB / MuseData / Piano-midi.de piano-roll splits.

---

## 2. Tokenization and representation

- **Equivariant Music Transformer.** Zixun Guo, Simon Dixon. ISMIR 2026 (arXiv journal-ref). arXiv:2608.03920. VERIFIED.
  Shows standard music transformers map time-shifted or pitch-transposed inputs to uncorrelated representations, and that they get **less** equivariant as they scale or train longer. The authors read this as capacity going to "memorizing absolute patterns". Adds a self-distillation equivariance loss that also improves next-token prediction. Directly relevant to transposition-aware leakage: a non-equivariant model can memorize a test tune's transposed copy without any exact-match overlap.
- **Moonbeam** (cluster 1): tokenizer with both absolute and relative attributes.
- **Exploring Tokenization Methods for Multitrack Sheet Music Generation.** Yashan Wang, Shangda Wu, Xingjian Du, Maosong Sun. 2024. arXiv:2410.17584. VERIFIED (3-page preprint). Compares ABC-based tokenizations.
- **MuPT** (cluster 1): the main published argument for ABC over MIDI for LLM-style training, plus SMT-ABC.
- **Natural Language Processing Methods for Symbolic Music Generation and Information Retrieval: A Survey.** Dinh-Viet-Toan Le, Louis Bigo, Dorien Herremans, Mikaela Keller. ACM Computing Surveys 57(7):1-40, 2025. DOI 10.1145/3714457. arXiv:2402.17467. VERIFIED. Author order differs: arXiv lists Keller before Herremans, the publisher record lists Herremans before Keller. The `.bib` follows the publisher.
- **Byte Pair Encoding for Symbolic Music.** Nathan Fradet, Nicolas Gutowski, Fabien Chhel, Jean-Pierre Briot. EMNLP 2023, pp. 2001-2020. DOI 10.18653/v1/2023.emnlp-main.123. arXiv:2301.11975. VERIFIED.
- **Impact of time and note duration tokenizations on deep learning symbolic music modeling.** Nathan Fradet, Nicolas Gutowski, Fabien Chhel, Jean-Pierre Briot. ISMIR 2023. arXiv:2310.08497. VERIFIED (pages not checked).
- **From Words to Music: A Study of Subword Tokenization Techniques in Symbolic Music Generation.** Adarsh Kumar, Pedro Sarmento. 2023. arXiv:2304.08953. VERIFIED (preprint).
- **MidiTok: A Python package for MIDI file tokenization.** Nathan Fradet, Jean-Pierre Briot, Fabien Chhel, Amal El Fallah Seghrouchni, Nicolas Gutowski. Extended Abstracts for the Late-Breaking Demo Session, ISMIR 2021. PDF: https://archives.ismir.net/ismir2021/latebreaking/000005.pdf. VERIFIED via the MidiTok documentation's citation block (the PDF itself was not opened).
- **Learning Transposition-Invariant Interval Features from Symbolic Music and Audio.** Stefan Lattner, Maarten Grachten, Gerhard Widmer. ISMIR 2018. arXiv:1806.08236. VERIFIED.
- **Pop Music Transformer** (REMI) and **Compound Word Transformer** (cluster 1) are the two reference MIDI tokenizations.
- UNVERIFIED: Kermarec, Bigo, Keller, "Improving tokenization expressiveness with pitch intervals", ISMIR 2022 LBD. I recall it but did not confirm it this session. Not in `refs.bib`.

**ABC vs MIDI, state of evidence.** MuPT, ChatMusician, NotaGen, TunesFormer, MelodyT5, and Text2Score all use ABC, and MuPT claims ABC is a better fit for LLMs. I found no controlled study that holds model, data content, and compute fixed and swaps only ABC vs a MIDI tokenization for monophonic melody. That is a small, cheap experiment the paper could include.

---

## 3. Evaluation

- **Survey on the Evaluation of Generative Models in Music.** Alexander Lerch, Claire Arthur, Nick Bryan-Kinns, Corey Ford, Qianyi Sun, Ashvala Vinay. ACM Computing Surveys 58(4):1-36, 2025. DOI 10.1145/3769106. arXiv:2506.05104. VERIFIED. The current reference survey for objective metrics, listening tests, and their failure modes.
- **Frechet Music Distance: A Metric For Generative Symbolic Music Evaluation.** Jan Retkowski, Jakub Stępniak, Mateusz Modrzejewski. 2024. arXiv:2412.07948. VERIFIED (preprint; no venue found as of today). Fréchet distance over pretrained symbolic-music embeddings (CLaMP family, not re-checked); code at github.com/jryban/frechet-music-distance.
- **Deep learning's shallow gains: a comparative evaluation of algorithms for automatic music generation.** Zongyu Yin, Federico Reuben, Susan Stepney, Tom Collins. Machine Learning 112(5):1785-1822, 2023. DOI 10.1007/s10994-023-06309-w. VERIFIED.
  Listening study (50 participants with musical training, six rating dimensions, Bayesian nonparametric tests). The best deep model (a Music Transformer reimplementation) did not beat MAIA Markov, and all systems sat well below human excerpts. Key evaluation critique and a ready-made argument for strong n-gram/Markov baselines.
- **A Comprehensive Survey for Evaluation Methodologies of AI-Generated Music.** Zeyu Xiong, Weitao Wang, Jing Yu, Yue Lin, Ziyan Wang. 2023. arXiv:2308.13736. VERIFIED (preprint).
- **A Survey on Deep Learning for Symbolic Music Generation: Representations, Algorithms, Evaluations, and Challenges.** Shulei Ji, Xinyu Yang, Jing Luo. ACM Computing Surveys 56(1):1-39, 2023. DOI 10.1145/3597493. VERIFIED.
- **ABC-Eval: Benchmarking Large Language Models on Symbolic Music Understanding and Instruction Following.** Jiahao Zhao, Yunjia Li, Wei Li, Kazuyoshi Yoshii. 2025. arXiv:2509.23350. VERIFIED (preprint). 1,086 items, 10 subtasks. Covers understanding rather than generation.
- **MusPy: A Toolkit for Symbolic Music Generation.** Hao-Wen Dong, Ke Chen, Julian McAuley, Taylor Berg-Kirkpatrick. ISMIR 2020. arXiv:2008.01951. VERIFIED (pages not checked). Dataset loaders (including Nottingham, JSB, Essen, Lakh) and metrics (pitch-class entropy, scale consistency, groove consistency, empty-beat rate; list not re-checked).
- **The Jazz Transformer on the Front Line** (cluster 1). Defines pitch-class histogram entropy, grooving-pattern similarity, chord progression irregularity, and structureness indicator (not re-checked).
- **On the evaluation of generative models in music.** Li-Chia Yang, Alexander Lerch. Neural Computing and Applications 32(9):4773-4784, 2020 (online 2018). DOI 10.1007/s00521-018-3849-7. VERIFIED. Absolute and relative feature-distribution metrics with KL divergence and overlapping area; released as the "mgeval" toolkit (not re-checked).
- **Taking the Models back to Music Practice: Evaluating Generative Transcription Models built using Deep Learning.** Bob L. Sturm, Oded Ben-Tal. Journal of Creative Music Systems 2(1), 2017. DOI 10.5920/jcms.2017.09. VERIFIED. Practice-based evaluation of folk-rnn.
- **Machine learning research that matters for music creation: A case study.** Bob L. Sturm, Oded Ben-Tal, Úna Monaghan, Nick Collins, Dorien Herremans, Elaine Chew, Gaëtan Hadjeres, Emmanuel Deruty, François Pachet. Journal of New Music Research 48(1):36-55, 2019 (online 2018). DOI 10.1080/09298215.2018.1515233. VERIFIED.
- **C-RNN-GAN** (cluster 1 anchors): early "scale consistency" metric.

**Evaluation critiques, 2023-2026, in one line each.** Yin et al. 2023 (deep models no better than Markov in listening tests); Lerch et al. 2025 (survey of metric validity and listening-test practice); Fathi 2026 (long-context NLL gains on folk corpora are mostly exact copying of repeats, see cluster 4); Equivariant Music Transformer 2026 (scale reduces transposition equivariance); Choi et al. 2025 (Lakh duplicates make random-split evaluation unreliable).

---

## 4. Memorization, copying, and data leakage

### 4a. Music-specific work, newest first

- **Towards AI-Generated Music Plagiarism Detection as a Version Identification Problem.** Fotis Koutsikos, Ioannis Prokopiou, Spyridon Kantarelis, Vassilis Lyberatos, Pantelis Vikatos, Athanasios Aidinis, Themos Stafylakis, Athanasios Voulodimos, Giorgos Stamou. 6 Oct 2026. arXiv:2610.09075. VERIFIED (submitted to ICASSP 2027). Audio.
- **How Far Back Should a Transformer Look? Repetition and Copying in Music Sequence Models.** Amir Fathi. 2 Oct 2026. arXiv:2610.02837. VERIFIED (single-author preprint, not peer reviewed). **Closest prior work to this project.**
  Datasets: Nottingham (Jukedeck MIDI, repeats expanded), O'Neill's 1903 Irish tunes (ABC), MusicNet, MAESTRO v3. Small 2-layer transformer (about 2.5M params) at context lengths 6 to 336 tokens.
  Leakage findings worth quoting: in the author's own earlier draft, renderings of the same tune were split at random and "92.4% of test sequences had another rendering of the same tune in training", with 58.4% of test blocks occurring verbatim in training. After grouping tunes that share at least half of their non-constant 16-token patterns, verbatim 48-token overlap fell to 0.0% (Nottingham) and 0.6% (O'Neill's). The grouping is token-level and is **not described as transposition-invariant**. Training windows are randomly transposed by -5 to +6 semitones; 0.9% of targets match only under transposition.
  Main result: NLL on Nottingham falls from 1.391 to 0.405 bits/token as context grows from 6 to 336 tokens, and a copy-retrieval baseline mixed with a 6-token predictor recovers 94% of that reduction (97% on O'Neill's). Long-context "gains" on folk melodies are mostly copying of repeated passages within a tune.
  What it does not do: audit the **standard** Boulanger-Lewandowski Nottingham split, JSB, Essen, or The Session; test transposition-invariant cross-split duplicates; or measure how leakage changes published likelihoods.
- **Equivariant Music Transformer** (cluster 2, ISMIR 2026): larger models memorize absolute patterns rather than transposition-shared structure.
- **Understanding Human Perception of Music Plagiarism Through a Computational Approach.** Daeun Hwang, Hyeonbin Hwang. ISMIR 2024 Late-Breaking Demo (arXiv comment); posted to arXiv 2026. arXiv:2601.02586. VERIFIED (arXiv).
- **Music Plagiarism Detection: Problem Formulation and a Segment-based Solution.** Seonghyeon Go, Yumin Kim. 2026. arXiv:2601.21260. VERIFIED (preprint).
- **Why Do Music Models Plagiarize? A Motif-Centric Perspective.** Tatsuro Inaba, Kentaro Inui. NeurIPS 2025 Workshop "AI for Music: Where Creativity Meets Computation". OpenReview Ula7U5a0yH; https://neurips.cc/virtual/2025/123788. VERIFIED (NeurIPS virtual-site page). No arXiv version found.
  Hypothesis: plagiarism in symbolic transformers comes from local overfitting of motifs that repeat within a piece, rather than whole-piece memorization. Frequently repeated motifs get lower perplexity and reappear more in samples. Tests label smoothing, transposition augmentation, and top-k as mitigations (effect sizes not read; OpenReview blocked the fetch).
- **On the de-duplication of the Lakh MIDI dataset.** Eunjin Choi, Hyerin Kim, Jiwoo Ryu, Juhan Nam, Dasaem Jeong. ISMIR 2025 (arXiv comment; ISMIR 2025 program poster P1-5). arXiv:2509.16662. VERIFIED.
  LMD-full has 178,561 MD5-unique files (the LMD page says 176,581; the two counts disagree). MIDI-hash dedup finds 26,167 duplicate files but has recall 0.172 on the LMD-clean benchmark. Their best configuration (CLaMP-1024 + contrastive "CAugBERT") flags 68,075 duplicates; the conservative list (similarity >= 0.99) flags 38,134 (21.4%). The paper argues duplicates across random splits skew cross-entropy/PPL but **leaves measuring that effect to future work**.
- **SSIMuse: Assessing Data Replication in Symbolic Music via an Adapted Structural Similarity Index Measure.** Shulei Ji, Zihao Wang, Le Ma, Jiaxing Yu. 2025. arXiv:2509.13658. VERIFIED (preprint). Piano-roll SSIM that searches over time shifts, pitch transpositions, and duration ratios; detects replication of at least one bar.
- **Bob's Confetti: Phonetic Memorization Attacks in Music and Video Generation.** Jaechul Roh, Zachary Novack, Yuefeng Peng, Niloofar Mireshghallah, Taylor Berg-Kirkpatrick, Amir Houmansadr. 2025. arXiv:2507.17937. VERIFIED (preprint). Audio/lyrics.
- **MelodySim: Measuring Melody-aware Music Similarity for Plagiarism Detection.** Tongyu Lu, Charlotta-Marlena Geist, Jan Melechovsky, Abhinaba Roy, Dorien Herremans. 2025. arXiv:2505.20979. VERIFIED (preprint).
- **Real-world Music Plagiarism Detection With Music Segment Transcription System.** Seonghyeon Go. APSIPA ASC 2025. DOI 10.1109/APSIPAASC65261.2025.11249201. arXiv:2509.08282. VERIFIED (arXiv; DOI from arXiv record).
- **Watermarking Training Data of Music Generation Models.** Pascal Epple, Igor Shilov, Bozhidar Stevanoski, Yves-Alexandre de Montjoye. 2024. arXiv:2412.08549. VERIFIED (preprint). Audio.
- **Towards Assessing Data Replication in Music Generation with Music Similarity Metrics on Raw Audio.** Roser Batlle-Roca, Wei-Hsiang Liao, Xavier Serra, Yuki Mitsufuji, Emilia Gómez. ISMIR 2024. arXiv:2407.14364. VERIFIED. Audio.
- **Exploring Musical Roots: Applying Audio Embeddings to Empower Influence Attribution for a Generative Music Model.** Julia Barnett, Hugo Flores Garcia, Bryan Pardo. 2024. arXiv:2401.14542. VERIFIED (preprint). Audio.
- **Generation or Replication: Auscultating Audio Latent Diffusion Models.** Dimitrios Bralios, Gordon Wichern, François G. Germain, Zexu Pan, Sameer Khurana, Chiori Hori, Jonathan Le Roux. ICASSP 2024, pp. 1156-1160. DOI 10.1109/ICASSP48485.2024.10447705. arXiv:2310.10604. VERIFIED. Audio.
- **MusicLDM: Enhancing Novelty in Text-to-Music Generation Using Beat-Synchronous Mixup Strategies.** Ke Chen, Yusong Wu, Haohe Liu, Marianna Nezhurina, Taylor Berg-Kirkpatrick, Shlomo Dubnov. 2023. arXiv:2308.01546. VERIFIED (arXiv). Audio; mixup to reduce training-set replication, with a novelty/plagiarism evaluation (not re-checked).
- **MusicLM: Generating Music From Text.** Andrea Agostinelli et al. (13 authors, see `.bib`). 2023. arXiv:2301.11325. VERIFIED. Audio; contains a memorization analysis (not re-checked).
- **Simple and Controllable Music Generation** (MusicGen). Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, Alexandre Défossez. NeurIPS 2023, pp. 47704-47720. arXiv:2306.05284. VERIFIED. Audio; included because later replication studies use it as the subject model.
- **Measuring When a Music Generation Algorithm Copies Too Much: The Originality Report, Cardinality Score, and Symbolic Fingerprinting by Geometric Hashing.** Zongyu Yin, Federico Reuben, Susan Stepney, Tom Collins. SN Computer Science 3(5), 2022. DOI 10.1007/s42979-022-01220-y. VERIFIED. **The main symbolic memorization study before 2025.**
  Originality report computed against a baseline of how much human composers borrow from themselves and each other. Music Transformer trained on 64 string-quartet movements (transposition-augmented) falls below the baseline's 95% interval: early in training it emits repetitive single notes, later it copies excerpts of training pieces. A larger dataset did not remove the copying. MAIA Markov stayed within the human interval.
- **"A Good Algorithm Does Not Steal - It Imitates": The Originality Report as a Means of Measuring When a Music Generation Algorithm Copies Too Much.** Same four authors. EvoMUSART 2021, LNCS pp. 360-375. DOI 10.1007/978-3-030-72914-1_24. VERIFIED. Conference precursor of the SN CS paper.
- **Folk the Algorithms: (Mis)Applying Artificial Intelligence to Folk Music.** Bob L. T. Sturm, Oded Ben-Tal. Handbook of Artificial Intelligence for Music (Springer, 2021), pp. 423-454. DOI 10.1007/978-3-030-72116-9_16. VERIFIED (Crossref). I did not read it, so I cannot say whether it contains a systematic copying analysis of folk-rnn.
- **Bacher than Bach? On Musicologically Informed AI-Based Bach Chorale Harmonization.** Alexander Leemhuis, Simon Waloschek, Aristotelis Hadjakos. ECML PKDD 2019 Workshops, CCIS 1168, pp. 462-469 (Springer, 2020). DOI 10.1007/978-3-030-43887-6_39. VERIFIED (publisher-linked repository record).
  Notes that a chorale melody may appear in both training and test because Bach harmonized the same hymn tune several times and the split was random, then chooses not to correct for it. This is the only explicit acknowledgment of JSB melody leakage I found, and it is not quantified.
- **Modelling Symbolic Music: Beyond the Piano Roll.** Christian Walder. ACML 2016, PMLR 63:174-189. arXiv:1606.01368. VERIFIED. Builds splits that reduce dependence between multiple transcriptions of the same piece in a MIDI archive; keeps the original JSB split. Companion data note: Walder, "Symbolic Music Data Version 1.0", arXiv:1606.02542 (VERIFIED).
- **DeepBach** (cluster 1 anchors): histogram of the longest subsequence copied from training.
- **Whole-Song Hierarchical Generation** (cluster 1): copy-bot baselines.
- Dataset-level dedup in the corpora themselves: PDMX removes about 60% of scores as duplicates using metadata embeddings plus note counts; Aria-MIDI ships pruned, deduped, and compositionally-unique subsets; LMD-full is only MD5-deduplicated (see cluster 6).
- Non-paper evidence: `stefan-balke/nottingham_match` (GitHub) maps the Jukedeck Nottingham files onto the Boulanger-Lewandowski split and notes "a couple of wrong matches due to derived songs in the database, i.e., two songs only deviate very slightly". That is direct evidence that near-duplicate tunes exist inside Nottingham. Jukedeck has 1,034 MIDIs with repetitions included; the Montreal (Boulanger-Lewandowski) release has 1,037 with no repetitions.

### 4b. General LM memorization and deduplication

- **Extracting Training Data from Large Language Models.** Nicholas Carlini, Florian Tramèr, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Úlfar Erlingsson, Alina Oprea, Colin Raffel. USENIX Security 2021, pp. 2633-2650. arXiv:2012.07805. VERIFIED (USENIX page).
- **Deduplicating Training Data Makes Language Models Better.** Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, Nicholas Carlini. ACL 2022 (Long), pp. 8424-8445. DOI 10.18653/v1/2022.acl-long.577. arXiv:2107.06499. VERIFIED. Over 4% of validation examples in standard LM datasets have a near-duplicate in training, and Transformer-XL perplexity on those examples is roughly half that on the rest. This is the text-domain number to replicate for melody.
- **Deduplicating Training Data Mitigates Privacy Risks in Language Models.** Nikhil Kandpal, Eric Wallace, Colin Raffel. ICML 2022, PMLR 162:10697-10707. arXiv:2202.06539. VERIFIED. Extraction rate grows superlinearly with how often a sequence is duplicated (not re-checked).
- **Quantifying Memorization Across Neural Language Models.** Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramer, Chiyuan Zhang. ICLR 2023. arXiv:2202.07646. VERIFIED. Memorization grows log-linearly with model size, duplication count, and prompt length (not re-checked). The three axes map directly onto a melody study (model size, tune duplication across settings/transpositions, prefix length in bars).
- **Extracting Training Data from Diffusion Models.** Nicholas Carlini, Jamie Hayes, Milad Nasr, Matthew Jagielski, Vikash Sehwag, Florian Tramèr, Borja Balle, Daphne Ippolito, Eric Wallace. USENIX Security 2023 (venue not re-checked). arXiv:2301.13188. VERIFIED (arXiv record only).
- **Large Language Models Struggle to Learn Long-Tail Knowledge.** Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, Colin Raffel. ICML 2023, PMLR 202:15696-15707. arXiv:2211.08411. VERIFIED.

### 4c. Answer to the specific question

**Has anyone shown that Nottingham, JSB Chorales, Essen, or The Session/IrishMAN contain transposed or near-duplicate tunes across standard train/test splits, and how much that inflates reported likelihoods?** Not that I could find, as of 8 October 2026. What exists:

1. Evidence that near-duplicates exist inside the corpora: derived songs in Nottingham (`nottingham_match`), repeated hymn tunes in JSB (Leemhuis et al. 2019, unquantified), variant renderings in Essen (folk-song tune-family literature; one 2026 affect paper dedups Essen by title before sampling), multiple settings per tune in The Session (IrishMAN's 99/1 random split does not describe any dedup).
2. Evidence that random splits on related data leak heavily: Fathi 2026's own earlier draft (92.4% of test sequences had another rendering of the same tune in training) and Choi et al. 2025 on Lakh (at least 21.4% high-confidence duplicates).
3. Evidence that leakage would lower NLL: only from text (Lee et al. 2022: perplexity halves on near-duplicate validation examples).
4. Evidence that transposed copies matter: transposition augmentation is standard practice (e.g., Yin et al. 2022, Fathi 2026), and EMT 2026 shows large models do not treat transposed inputs as the same, so they can memorize a transposed copy without any exact overlap. SSIMuse 2025 searches over transpositions when detecting replication, but only for generated outputs. Nobody has applied it to train/test splits.

Nobody has (a) audited the **published** Boulanger-Lewandowski Nottingham and JSB splits, the common Essen and IrishMAN splits, with a transposition-invariant near-duplicate detector, or (b) re-scored published or reproduced models on leaked vs clean test subsets to measure the NLL/perplexity gap. Both are open.

---

## 5. Melody-specific modeling, expectation, and scaling

### Folk and melody generation

- **folk-rnn: Music transcription modelling and composition using deep learning.** Bob L. Sturm, João Felipe Santos, Oded Ben-Tal, Iryna Korshunova. 1st Conference on Computer Simulation of Musical Creativity (CSMC), 2016. arXiv:1604.08723. VERIFIED. Character/token LSTM on The Session ABC. A survey reports 23,636 transcriptions after filtering for the v2 model (not re-checked against the paper).
- Sturm & Ben-Tal 2017 (JCMS), Sturm et al. 2019 (JNMR), Sturm & Ben-Tal 2021 (Folk the Algorithms): see clusters 3 and 4.
- **TunesFormer**, **MelodyT5**, **MelodyGLM**, **Small Tunes Transformer**, **RPPNet**, **MusicMamba**: see cluster 1.
- **Exploring Sampling Techniques for Generating Melodies with a Transformer Language Model.** Mathias Rose Bjare, Stefan Lattner, Gerhard Widmer. ISMIR 2023. arXiv:2308.09454. VERIFIED. Nucleus/typical sampling on melody LMs; relevant to the temperature-vs-copying axis.
- UNVERIFIED as a citable source: Gwern Branwen's GPT-2 folk experiments (blog) report that generated titles were copied while tunes mostly were not. Blog only; do not cite as evidence.

### Information-theoretic models of melodic expectation (IDyOM / PPM lineage)

- **GraphIDyOM: A graph-native Python reimplementation of IDyOM for musical expectation modelling.** Lluc Bono Rosselló. 2026. arXiv:2607.25787. VERIFIED (preprint). Reports that IDyOMpy covers only part of the Lisp IDyOM configuration space and deviates more from the reference. Useful if the paper uses IDyOM as a baseline from Python.
- **Striking a New Chord: Neural Networks in Music Information Dynamics.** Farshad Jafari, Claire Arthur. SMPC 2024 (arXiv comment). arXiv:2410.17989. VERIFIED. Chords rather than melody: LSTM/Transformer/GPT vs a Markov model.
- **Deep Generative Models of Music Expectation.** Ninon Lizé Masclef, T. Anderson Keller. 2023. arXiv:2310.03500. VERIFIED (preprint).
- **Cortical activity during naturalistic music listening reflects short-range predictions based on long-term experience.** Pius Kern, Micha Heilbron, Floris P. de Lange, Eelke Spaak. eLife 11:e80935, 2022. DOI 10.7554/eLife.80935. VERIFIED. Compares IDyOM (STM/LTM) and Music Transformer on melodies. IDyOM STM had the highest next-note accuracy (57.9% vs 54.8% for Music Transformer, per search-result summary, not re-read), while MT and IDyOM LTM best explained neural surprise.
- **PPM-Decay: A computational model of auditory prediction with memory decay.** Peter M. C. Harrison, Roberta Bianco, Maria Chait, Marcus T. Pearce. PLOS Computational Biology 16(11):e1008304, 2020. DOI 10.1371/journal.pcbi.1008304. VERIFIED.
- **Statistical learning and probabilistic prediction in music cognition: mechanisms of stylistic enculturation.** Marcus T. Pearce. Annals of the New York Academy of Sciences 1423(1):378-395, 2018. DOI 10.1111/nyas.13654. VERIFIED.
- **The construction and evaluation of statistical models of melodic structure in music perception and composition.** Marcus T. Pearce. PhD thesis, City University London, 2005. https://openaccess.city.ac.uk/id/eprint/8459/. VERIFIED (repository record). The IDyOM thesis; PPM variants and viewpoint combination evaluated by cross-entropy on melody corpora (corpus list not re-checked).
- **Improved Methods for Statistical Modelling of Monophonic Music.** Marcus Pearce, Geraint Wiggins. Journal of New Music Research 33(4):367-385, 2004. DOI 10.1080/0929821052000343840. VERIFIED.
- **Multiple viewpoint systems for music prediction.** Darrell Conklin, Ian H. Witten. Journal of New Music Research 24(1):51-73, 1995. DOI 10.1080/09298219508570672. VERIFIED.

### n-gram/PPM vs neural on melody, state of evidence

- Yin et al. 2023: a Markov model (MAIA) matched Music Transformer in listening tests.
- Fathi 2026: a short-context predictor plus exact-copy retrieval recovers 94-97% of a transformer's long-context NLL gain on folk tunes.
- Kern et al. 2022: IDyOM STM out-predicts Music Transformer on next-note accuracy for the tested melodies.
- I found no paper that reports IDyOM/PPM cross-entropy and a modern transformer's cross-entropy on the **same** standard folk split with matched training data. PPM is itself a copy-heavy model (long-context exact matches), so a leakage audit should report how much each model class gains from leaked tunes.

### Scaling laws for symbolic music

- **MuPT** (ICLR 2025) is the only paper I found that fits a scaling law for symbolic music LMs (SMS law, Chinchilla modified for repeated data). It is ABC, multi-track, and does not separate memorization from generalization.
- **Aria** (ISMIR 2025) scales pretraining data to about 60k hours of piano but reports downstream quality without fitting a law.
- **Equivariant Music Transformer** (ISMIR 2026) reports that equivariance drops with scale and training length.
- UNVERIFIED: a standalone "Scaling laws for symbolic music" paper. I searched arXiv and the web and did not find one.

---

## 6. Datasets and licenses

Licenses checked on the dataset's own page this session unless noted. `../data/SOURCES.md` has commit hashes and verbatim license text for the copies already downloaded; where both exist they agree.

| Dataset | Content and size | License / terms | Primary reference | Notes for leakage work |
|---|---|---|---|---|
| **PDMX** | 254,077 MusicXML scores from MuseScore (initial scrape); 102,635 after dedup; 14,182 rated | Scores filtered to Public Domain Mark or CC0. Zenodo record CC-BY-4.0 (per SOURCES.md). Authors recommend the `no_license_conflict` subset (12.29% of 31,221 checked songs had conflicting metadata) | Long, Novack, Berg-Kirkpatrick, McAuley, ICASSP 2025, DOI 10.1109/ICASSP49660.2025.10890217, arXiv:2409.10831. VERIFIED | Dedup = Sentence-BERT on title/artist/composer metadata (>= 0.8 cosine), then split by instrumentation and note count (5%). Content-blind to transposition; removes about 60% |
| **The Session** (thesession.org dump) | Tunes, settings, sets, aliases (GitHub `adactio/TheSession-data`); count not stated on the page | ODbL for the database **plus** a "License for contents" that prohibits using, adapting, or processing the material with Large Language Models (accessibility exception). Added 2025-10-08 per SOURCES.md | Data dump, no paper | Many tunes have several user "settings". Whether a melody LM counts as an LLM under the clause is unresolved |
| **IrishMAN** | 216,284 Irish tunes in ABC from thesession.org and abcnotation.com; 99% train (214,122) / 1% validation (2,162); 34,211-tune lead-sheet subset with chords | HF card: MIT tag; "research use only"; claims tunes are public domain | Wu et al., TunesFormer (HCMIR 2023), arXiv:2301.02884. VERIFIED | No dedup described; random 1% validation is very likely to contain other settings of training tunes |
| **Nottingham Music Database** | "Over 1000" folk tunes (Eric Foxley), ABC by James Allwright, corrected by Seymour Shlien | No license stated at abc.sourceforge.net/NMD; Ashover subset IP belongs to Mick Peat | https://abc.sourceforge.net/NMD/ | Original source of the B-L split |
| **Nottingham (Jukedeck cleaned)** | ABC_original, ABC_cleaned, MIDI; 1,034 MIDIs per `nottingham_match` | GNU GPLv3 | github.com/jukedeck/nottingham-dataset | Some offending pieces removed; repeats normalized |
| **Nottingham / JSB / MuseData / Piano-midi.de piano-roll splits** | JSB: 382 chorales, 229/76/77 (per search-result summary of the czhuang README; counts not re-read) | None stated for the pickles; per-source terms (SOURCES.md) | Boulanger-Lewandowski et al., ICML 2012, arXiv:1206.6392. VERIFIED | JSB split taken from Allan & Williams 2005 (not verified here); repeated hymn tunes across chorales |
| **Essen Folksong Collection** | 8,473 folksongs encoded by Helmut Schaffrath (KernScores count, per search result; site refused connection when fetched) | music21's copy: "legal status ... is unclear"; non-commercial permission for music21 from Ewa Dahlig-Turek. CCARH **kern copy: academic, single-user, no redistribution (SOURCES.md) | Schaffrath (ed. Huron), CCARH; Sapp, "Online Database of Scores in the Humdrum File Format", ISMIR 2005, pp. 664-665 (from search result; UNVERIFIED pages) | Contains variant renderings of the same song; no tune-family labels. Meertens Tune Collections (MTC-ANN) have tune-family labels for Dutch songs and are a better testbed for variant-aware splits |
| **Lakh MIDI (LMD)** | LMD-full 176,581 files (project page; Choi et al. count 178,561); LMD-matched 45,129 | CC-BY 4.0 | Raffel, PhD thesis, Columbia University, 2016 (title per LMD page). VERIFIED | MD5 dedup only; at least 38,134 near-duplicates per Choi et al. 2025 |
| **POP909** | 909 pop songs as piano arrangements with melody, bridge, accompaniment tracks, plus beat/chord/key annotations | MIT (repo); underlying songs are copyrighted pop | Wang et al., ISMIR 2020, arXiv:2008.07142. VERIFIED | "versions" folder holds multiple versions of the same arrangement |
| **Hooktheory (Sheet Sage release)** | About 50 h of aligned melody + harmony; 22k segments of 13k recordings; 8:1:1 split stratified by artist | CC BY-NC-SA 3.0 (data and models); code MIT | Donahue, Thickstun, Liang, ISMIR 2022, arXiv:2212.01884. VERIFIED | Artist-stratified split is the only melody dataset here with an explicit leakage-aware split. TheoryTab (used by MIDI-LLM) is a separate, non-public Hooktheory corpus |
| **Wikifonia** | About 6,405-6,675 lead sheets (counts vary by snapshot) | None; site closed 25 Dec 2013 over licensing. EWLD (Zenodo) is restricted, non-commercial; OpenEWLD is a public-domain subset | No canonical paper | PDMX paper calls it "unsafe for copyright issues" |
| **Weimar Jazz Database** | 456 monophonic solo transcriptions by 78 performers (secondary sources) | No specific license found on the project page ("available to the public ... for jazz and MIR research"). UNVERIFIED license | Pfleiderer, Frieler, Abeßer, Zaddach, Burkhart (eds.), *Inside the Jazzomat*, Schott Campus, 2017 (citation given on the project page) | Used by Jazz Transformer |
| **GiantMIDI-Piano** | 10,855 MIDI files, 2,786 composers; curated subset 7,236 files | CC BY 4.0 (repo); download behind a disclaimer | Kong, Li, Chen, Wang, TISMIR 5(1):87-98, 2022, DOI 10.5334/tismir.80. VERIFIED | Transcribed from YouTube audio |
| **Aria-MIDI** | 1,186,253 files, about 100,629 h; pruned 820,944; deduped 371,053; compositionally unique 32,522 | CC-BY-NC-SA 4.0 plus disclaimer | Bradshaw & Colton, ICLR 2025, arXiv:2504.15071. VERIFIED | Ships explicit dedup levels |
| **MetaMIDI (MMD)** | 436,631 MIDI files; artist/title metadata for 221,504; genre for 143,868; matched to Spotify clips and MusicBrainz | No license named; access on request for research (data mining/ML) only; no redistribution | Ens & Pasquier, ISMIR 2021, pp. 182-188; Zenodo DOI 10.5281/zenodo.5142664. VERIFIED | |
| **Million Song Dataset** | Features and metadata for 1M tracks (no audio, no MIDI) | Free for research (license not checked) | Bertin-Mahieux, Ellis, Whitman, Lamere, ISMIR 2011, pp. 591-596. VERIFIED | Only relevant through LMD-matched |
| **GigaMIDI** | Over 1M MIDI files (not re-checked) | Not checked | Lee, Ens, Adkins, Sarmento, Barthet, Pasquier, TISMIR 8(1), 2025, DOI 10.5334/tismir.203, arXiv:2502.17726. VERIFIED (paper only) | |
| **PianoCoRe** | Combined and refined piano MIDI | Not checked | Borovik, TISMIR 9(1):144-163, 2026, DOI 10.5334/tismir.333, arXiv:2605.06627. VERIFIED (paper only) | Refinement includes dedup across merged sources (per title/abstract; not read in detail) |

---

## 7. Current state of the art for melody and lead-sheet generation

- **Lead sheets, text-controlled, with real users:** MIDI-LLM (Llama 3.2 1B adapted to MIDI, fine-tuned on TheoryTab; ISMIR 2026) on top of Hookpad Aria (deployed anticipatory-style copilot). The evaluation that matters there is user acceptance rate in a real editor, which no academic benchmark reproduces.
- **ABC-native LLM pipeline:** NotaGen (pretrain on 1.6M ABC pieces, fine-tune, CLaMP-DPO) is the strongest published recipe for sheet music, aimed at classical scores. MuPT and ChatMusician are the open ABC foundation models; MelodyT5 is the multi-task melody model; TunesFormer is the Irish-tune generator.
- **Infilling and control:** Anticipatory Music Transformer (TMLR 2024) is the standard open model for infilling/accompaniment over MIDI.
- **Frontier systems that plan in lead-sheet space:** YuE2 (Sept 2026 tech report) writes a melody+harmony score before rendering audio. This makes symbolic melody quality a component of audio song generation, which raises the stakes for honest symbolic evaluation.
- **Non-transformer families:** diffusion (whole-song cascaded diffusion, ICLR 2024; rule-guided diffusion, ICML 2024; D3PIA, ICASSP 2026) and Mamba/SSM hybrids (MusicMamba ICASSP 2025; arXiv-only SSM diffusion work). None has displaced autoregressive transformers for monophonic melody.
- **Melody-only benchmarks** (Nottingham, JSB, Essen, IrishMAN validation) are still used for NLL comparisons, but recent large models mostly report subjective tests and feature-distribution metrics instead, so there is no shared, leakage-checked likelihood benchmark for melody.

---

## 8. Gaps and opportunities

Ordered by how directly they serve the TISMIR paper.

1. **Transposition-invariant leakage audit of the standard splits.** Run a near-duplicate detector that canonicalizes key (interval sequences, or transpose to a reference tonic) and handles small edits (n-gram Jaccard, edit distance on interval/IOI strings, or SSIMuse-style search) over: Boulanger-Lewandowski Nottingham and JSB, the IrishMAN 99/1 split, common Essen splits, and The Session settings. Report leaked fractions per split. No published audit exists (section 4c).
2. **Measure the likelihood inflation.** Re-score reproduced models (LSTM, small transformer, PPM/IDyOM-style n-gram) on leaked vs clean test subsets and report the NLL gap, the music analogue of Lee et al. 2022's "perplexity halves". Choi et al. 2025 explicitly leave this as future work for Lakh; Fathi 2026 avoids it by re-splitting. Releasing clean splits plus the gap table is the paper's data contribution.
3. **Separate within-tune copying from cross-split leakage.** Fathi 2026 shows long-context NLL on folk tunes is mostly copying of repeats inside the tune. A leakage study has to control for that (e.g., report NLL on first occurrences only, or on targets with no in-context 16-token match) so cross-split leakage is not confused with legitimate repetition modeling.
4. **Memorization vs model size, duplication count, and sampling temperature for melody.** Carlini et al. 2023's three axes have not been measured for symbolic melody. EMT 2026 suggests scale increases absolute-pattern memorization; Inaba & Inui 2025 suggest motif-level copying. A size sweep x {raw, dedup} x temperature grid with an originality metric (Yin et al. 2022 originality report, or extraction rate on held-out prefixes) fills it.
5. **Does deduplication help melody models?** Lee et al. 2022 and Kandpal et al. 2022 show dedup reduces memorization and improves quality in text. No melody paper reports a dedup ablation on generation quality and copying together.
6. **Fair baselines.** Yin et al. 2023 and Fathi 2026 both point at Markov/copy baselines closing most of the gap. Include PPM (IDyOM or GraphIDyOM) on the same clean splits.
7. **ABC vs MIDI under controlled conditions.** MuPT's claim has not been tested with model, data, and compute held fixed for monophonic melody.
8. **Licensing hygiene as a contribution.** The Session's 2025 LLM clause, IrishMAN's MIT tag over Session-derived content, Essen's CCARH no-redistribution terms, and Wikifonia's lack of any license all bear on which "clean splits" can be redistributed. PDMX (public domain/CC0) is the safest corpus for releasing derived data; a PDMX melody subset with a transposition-aware dedup would be a reusable benchmark.
9. **Tune-family-aware splits.** Folk corpora have tune families (variants). MTC-ANN has expert family labels and could validate a near-duplicate detector before it is applied to Essen and The Session.

Things I looked for and did not find (so they are not citable): a dedicated symbolic-music scaling-law paper; any audit of JSB/Nottingham/Essen/Session splits for transposed duplicates; a peer-reviewed measurement of verbatim extraction from a symbolic melody LM in the Carlini sense.
