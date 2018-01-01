library(stylo)
setwd("~/Documents/cs384/sty-aa/federalist/")
getwd()
classify(gui = TRUE, training.corpus.dir = "primary_set", test.corpus.dir = "secondary_set", ngram.size = 1)
#svm, unigram
?rolling.classify
rolling.classify(training.corpus.dir = "reference_set", test.corpus.dir = "test_set", classification.method="svm", slice.size = 1500, slice.overlap = 500)
zeta.results <- oppose(primary.corpus.dir = "hamilton", secondary.corpus.dir = "madison")
zeta.results$words.preferred[0:20]
zeta.results$words.avoided[0:20]

stylo(corpus.dir = "primary_set", analysis.type = "BCT",
      mfw.min = 100, mfw.max = 1000, custom.graph.title = "PresConflict",
      write.png.file = TRUE, gui = FALSE)

stylo(gui = TRUE, corpus.dir = "primary_set")

setwd("~/Documents/cs384/sty-aa/oz/")
getwd()
rolling.classify(gui=TRUE, training.corpus.dir = "reference_set", test.corpus.dir = "test_set", classification.method="svm")
stylo(corpus.dir = "data")

setwd("~/Documents/cs384/sty-aa/presconflict-chapters/")
stylo(corpus.dir = "corpus/")

setwd("~/Documents/cs384/sty-aa/presconflict/")
stylo.results <- stylo(corpus.dir = "./data")

stylo(corpus.dir = "./data", analysis.type = "BCT",
      mfw.min = 100, mfw.max = 1000, custom.graph.title = "PresConflict",
      write.png.file = TRUE, gui = FALSE)

stylo(corpus.dir = "./data", analysis.type = "CA",
      mfw.min = 500, mfw.max = 500, custom.graph.title = "PresConflict",
      write.png.file = TRUE, gui = FALSE)

zeta.results <- oppose(primary.corpus.dir = "Machen", secondary.corpus.dir = "Mencken")

#Machen preferred
zeta.results$words.preferred[0:20]
#Machen avoided
zeta.results$words.avoided[0:20]

classify(training.corpus.dir = "data", test.corpus.dir = "corpus")
