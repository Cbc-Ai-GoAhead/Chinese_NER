training data size : 33887

Loading Evaluation data
100%|███████████████████████████████████████████████████████████████████████████████| 485541/485541 [00:09<00:00, 52239.20it/s]
Eval
tag2id: {'O': 0, 'B-LOC': 1, 'I-LOC': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-PER': 5, 'I-PER': 6}
id2tag: {0: 'O', 1: 'B-LOC', 2: 'I-LOC', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-PER', 6: 'I-PER'}
tag number: 7 

evaluation data size : 9671 


Epoch: 0=====================================================================================
---------------------------Training---------------------------
Train:   0%|                                                                                           | 0/132 [00:00<?, ?it/s]/home/cbc/deep_learning_course/final_project/AITutorial-2022-ChineseNER/package/dataset.py:98: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  embedding_tensor=torch.tensor(embedding)
/home/cbc/deep_learning_course/final_project/AITutorial-2022-ChineseNER/package/nn.py:171: UserWarning: where received a uint8 condition tensor. This behavior is deprecated and will be removed in a future version of PyTorch. Use a boolean condition instead. (Triggered internally at ../aten/src/ATen/native/TensorCompare.cpp:493.)
  score = torch.where(mask[i].unsqueeze(1), next_score, score)
Train: 100%|███████████████████████████████████████████████████████████████████| 132/132 [40:32<00:00, 18.43s/it, loss=1.12e+3]
--------------------------Evaluation--------------------------
Eval:  95%|███████████████████████████████████████████████████████████████████████████████▌    | 36/38 [10:14<00:34, 17.42s/it]
Eval: 100%|████████████████████████████████████████████████████████████████████████████████████| 38/38 [10:45<00:00, 17.00s/it]
processed 475870 tokens with 13274 phrases; found: 12286 phrases; correct: 7120.
accuracy:  64.38%; (non-O)
accuracy:  95.55%; precision:  57.95%; recall:  53.64%; FB1:  55.71
              LOC: precision:  60.92%; recall:  53.20%; FB1:  56.80  5051
              ORG: precision:  58.07%; recall:  43.35%; FB1:  49.64  2919
              PER: precision:  54.40%; recall:  65.59%; FB1:  59.47  4316
f1: 55.712050078247266
recall: 53.6386921802019
precision: 57.952140647891916
Training Finish
