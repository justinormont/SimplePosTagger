# SimplePosTagger
Sample part-of-speech tagger in ML․NET

### Metrics
When using AveragedPerceptron as the trainer, the testset metrics are as follows.

* Accuracy (micro-avg):              `0.9154`   # 0..1, higher is better
* Accuracy (macro):                  `0.6963`   # 0..1, higher is better
* Top-K accuracy:                    `[0.9154, 0.9636, 0.9785, 0.9846, 0.9879, 0.9898, 0.9907, 0.9915, 0.9921, 0.9926, 0.9929, 0.9933, 0.9938, 0.9941, 0.9946, 0.9949, 0.9953, 0.9958, 0.9959, 0.9961, 0.9964, 0.9966, 0.9969, 0.9971, 0.9972, 0.9974, 0.9975, 0.9976, 0.9977, 0.9980, 0.9981, 0.9983, 0.9985, 0.9986, 0.9986, 0.9987, 0.9988, 0.9988, 0.9989, 0.9989, 0.9990, 0.9991, 0.9993, 0.9993, 0.9995, 0.9995, 0.9996, 0.9997, 0.9997, 0.9997, 0.9998, 0.9999, 0.9999, 1.0000, 1.0000]`   # 0..1, higher is better
* Log-loss reduction:                `0.8497`   # -Inf..1, higher is better
* Log-loss:                          `0.4515`   # 0..Inf, lower is better
