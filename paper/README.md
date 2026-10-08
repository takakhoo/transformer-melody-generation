# Paper

`transposed-twins.pdf` is the current manuscript, in preparation for TISMIR and not yet peer reviewed (Transactions of the International Society for Music Information Retrieval), special collection *Open Music Data for Music Processing Research*.

The source is `tismir/main.tex` with `references.bib`; `numbers/` holds tables written by the experiment scripts. It compiles inside the official template ([ismir/paper_templates_TISMIR_new](https://github.com/ismir/paper_templates_TISMIR_new)): copy `main.tex`, `references.bib`, `numbers/` and `figures/` into the template folder, keep `\pdfmapfile{+FSMe.map}` as the first line, and run `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`. The template's fonts and class are not redistributed here.
