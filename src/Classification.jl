module Classification

##
# Naive Bayes
# - Working in log-space, since we'll be multiplying many small probabilities with one another
# i.e. P(w1 | c) * P(w2 | c) ... * P(wn | c), where w = word, c = class
# - We'll be using a bag of words representation, for convenience (not realistic)
# - We'll consider that every word is independent of each other (not realistic e.g. "thank you" is not the same as "thank" and "you")
# - We'll use a hyperparameter, α, called "smoothing parameter" to avoid multiplying probabilities
# with zero in case some word(s) do not exist in the document
# - We'll use a confusion matrix (precision / recall), since accuracy breaks down when trying
# to discover rare or imbalanced classes
# - Precision / Recall trade-off: When one goes up the other goes down. Which one you favour
# depends on what kind of error is less costly for your goal
# - F1 score: harmonic mean of precision and recall. Drawback: it favours classifiers where
# precision and recall have a similar value, which may not apply in our situation

end
