# IEEE Paper Draft

This folder contains an IEEE-style LaTeX draft paper built from the current project outputs.

## Files

- `main.tex`: main IEEE conference paper source
- `references.bib`: bibliography file
- `figures/`: figures used inside the manuscript

## How to compile

If you have a LaTeX distribution installed, compile with:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Notes

- Replace the placeholder author names and affiliations before submission.
- The wording is meant to be editable, so you can refine claims, add institution details, or modify the results discussion.
- The figures are copied from the project outputs so the paper matches the actual experiments.
