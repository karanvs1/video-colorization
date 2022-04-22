## Tasks
1. Loss function update- Loss function from CIC with class rebalancing. X
2. Add CNN-LSTM: Maybe implement using LSTM-cell X
3. Verify output of trained model 
4. Fix dataloader issue - load frames in init, not getItems X
5. 
6. 


## TODOS:
1. Continue using MSE Loss (Use Cross-Entropy if there is time)
2. Change preprocessing network to pass into the conv1 layer and freeze weights from conv2 onwards
3. Modify network to run frame wise colorization (Share with Aparajith)
4. Separate branches for experiments 2 and 3
5. Write script for downloading dataset - K/K
6. Training loop interface improve: Add number of epochs, fix tqdm bar "disappearing" issue


## Experiments: [Sample video](https://www.youtube.com/watch?v=4iDx-ctQkiQ)


2. Context based frame wise colorization video (A + C)

3. Attention + Context based frame wise colorization video (K + K)

4. Concatenate the output from the previous frame into the input (Stretch)


## Tasks:
### (in chronological order, no. of * indicates priority)
1. Improve Training loop interface - Arj*
2. Write script for downloading complete dataset to AWS - C*
3. Modify network to run just CIC based frame wise colorization (Share script with Aparajith) - K/K**
4. Change preprocessing network to pass into the conv1 layer and freeze weights from conv2 onwards - K/K**
5. Make separate branches

### Once done with above tasks:
6. Context based frame wise colorization video (Arj + C)***
7. Attention + Context based frame wise colorization video (K + K)***
0. Ground truth sample video (A) and Frame wise colorization video (A)