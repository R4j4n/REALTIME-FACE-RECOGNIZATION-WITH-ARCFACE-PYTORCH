## REALTIME FACE RECOGNIZATION WITH ARCFACE PYTORCH


### STEP 1: Run ``` DATA_COLLECT.py``` to collect your face dataset. <br>
### STEP 2: Run ``` GENERATE_EMBEDDINGS.py ``` to generate facial embeddings. <br>
#### STEP 3: Then run ``` VALIDATE.py ``` for realtime face recognization. <br>



 Dowload pre-trained weight from [Here.👈](https://drive.google.com/file/d/1Fa5WrlJm7CWZQl2j7Xpi0RzU1Sih7g1g/view?usp=sharing) <br>
 And place it in the directory: 
 ```
 libs\arcface\weight\model_final.pth
 ```



![Alt Text](Media\chunea.gif) <br>

![Alt Text](Media\em.gif) <br>






# Dependencies

```
torch == 1.7.1
pencv-python == 4.5.1.48
numpy
scikit-learn
scipy
```