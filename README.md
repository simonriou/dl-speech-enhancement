# Model training schedule
- Model 1: batch_size = 8, epochs = 15, no batch norm, no validation split, trained on babble_16k only
- Model 2: batch_size = 8, epochs = 15, with batch norm, no validation split, trained on babble_16k only
- Model 3: batch_size = 8, epochs = 15, with batch norm, validation split = 0.2, trained on multiple babble noises
- Model 4: batch_size = 32, epochs = 15, with batch norm, validation split = 0.2, trained on multiple babble noises