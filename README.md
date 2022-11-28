# ExMed-BERT: A Transformer-Based Model Trained on Large Scale Claims Data for Prediction of Severe COVID Disease Progression

Code for ExMed-BERT training and fine-tuning.

## Pre-trained model

Our pre-trained model is available at https://doi.org/10.5281/zenodo.7324178.

## Relevant files

- **scripts/calculate\_iptw\_scores.py**: Script to calculate IPTWs
- **scripts/pretrain-exmed-bert.py**: Script to pre-train ExMed-BERT from scratch
- **scripts/finetune-exmed-bert.py**: Script to fine-tune ExMed-BERT for a classification task
- **scripts/train-rf.py**: Script to train an RF model for a classification task
- **scripts/train-xgboost.py**: Script to train an XGBoost model for a classification task

## Contact

Please post a GitHub issue or write an e-mail to manuel.lentzen@scai.fraunhofer.de if you have any questions.
