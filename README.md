# TO Run training and results
1. You need to run in T4 GPU in colab to run the Code_Final_training.ipynb training part

2. Run the code untill extracting zip file

3. Change data.yaml after extracting the zip file to the following

path: /content/dataset

train: train/images
val: valid/images
test: test/images

names: 
  0: crack
  1: hole
  2: spalling

4. Run the training (it will take an hour and a half)

5. Change the model weights line in the produce results section in line 166 to the latest train file in runs/detect/

MODEL_WEIGHTS = 'runs/detect/train5/weights/best.pt'

6. Check the latest produced PDF to see the results in runs/detect



# TO Produce results only run from the produce results only section

1. You need to run in T4 GPU in coab to run the Code_Final_training.ipynb

2. Run the code untill extracting zip file

3. Run the rest of the code from the Produce results section (check the latest produced PDF to see the results in runs/detect) 