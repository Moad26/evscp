#import "@preview/cmarker:0.1.8"
#import "@preview/mitex:0.2.6": mitex

// Configure the document as a presentation
#set page(paper: "presentation-16-9", margin: 2em)
#set text(size: 20pt)

// Give tables more padding so math fractions don't hit the borders
#set table(inset: 0.6em)

// Turn markdown horizontal rules (---) into clean slide page breaks
#show line.where(length: 100%): pagebreak()

#cmarker.render(
  read("presentation.md"),
  math: mitex,
  scope: (
    image: (path, ..args) => image(path, ..args),
  ),
)
