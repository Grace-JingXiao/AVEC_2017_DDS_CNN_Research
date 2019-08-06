from jiwer import wer

ground_truth = "hello ducek world"
hypothesis = "hello world duck"

error = wer(ground_truth, hypothesis)
print(error)
