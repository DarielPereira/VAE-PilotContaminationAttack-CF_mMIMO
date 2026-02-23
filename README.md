# VAE-PilotContaminationAttack-CF_mMIMO
Using variational autoencoders (VAEs) to detect low-power, multi-adversary pilot contamination attacks in CF-mMIMO 
communication systems.

This code corresponds to the paper "A VAE Approach to Low-Power, Multi-Adversary Pilot Contamination Attacks in CF-mMIMO" 
by Dariel Pereira Ruisánchez, Óscar Fresnedo, Darian Pérez Adán, and Luis Castedo, submitted to EUSIPCO 2026 and available 
on TechRxiv: https://www.techrxiv.org/users/683174/articles/1389206 and 
   ResearchGate: https://www.researchgate.net/publication/400819465.

Note: Differences between the paper and the code may exist due to ongoing development and improvements. 
The code is provided as-is for reproducibility and further research.

## File structure
- `README.md` - Central project document (this file).
- `Graphs/` - It contains the figures generated for the results' publication.
- `Models/` - It contains the trained models ('_Large' model refers to the model trained with samples from
              large, practical configurations). 
- `TrainingData/` - It contains the datasets generated from the runSampleGeneration.py script.
- `functionsAllocation` - It implements the functions for pilot allocation and AP cooperation cluster formation.
- `functionsAttack` - It implements the function that emulates different attack strategies.
- `functionsAttackDetection.py` - It implements the functions that determine the attack scores and probabilities
                                  for the VAE-based and Norm-based approaches.
- `functionsChannelEstimates.py` - It computes the channel estimates and the channel statistics matrices for clean and 
                                   contaminated scenarios.
- `functionsComputeNMSE_uplink.py` - It computes NMSE values for the uplink channel estimation.
- `functionsComputeSE_uplink.py` - It computes SE values for the uplink data transmission.
- `functionscVAE.py` - It implements the class the rules the architecture and training of the VAE model.
- `functionsDataProcessing.py` - It implements the functions for managing the datasets, and some additional functions
                                 for data processing.
- `functionsGraphs.py` - It implements the functions for creating all the relevant graphs.
- `functionsSetup.py` - It implements the functions for creating the random scenarios, and for setting up the system 
                        parameters.
- `functionsUtils.py` - It implements some useful functions for this or other project.
- `runSampleGeneration.py` - It generates the datasets for training and testing the VAE model.
- `runTrainingVAE.py` - It trains the VAE model with the generated datasets.
- `runSimulations_CDF.py`, `runSimulations_KLScore_Norm_AttackProb.py`, and `runSimulations_ProbVsPower.py` run the 
simulations for the results presented in the paper, and generate the corresponding graphs.

## Credits and contact
- Authors: Dariel Pereira Ruisánchez, Óscar Fresnedo, Darian Pérez Adán, Luis Castedo.
- Contact: d.ruisanchez@udc.es 
