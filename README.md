# Description

Begin with this Tutorial: [A Deep Learning Approach](https://zindi.africa/learn/spot-the-mask-challenge-tutorial-a-deep-learning-approach) then proceed to the Climbing the Ladder: Image Recognition for ML Competitions (Tutorial) [here](https://zindi.africa/competitions/spot-the-mask)

Face masks have become a common public sight in the last few months. The Centers for Disease Control (CDC) recently advised the use of simple cloth face coverings to slow the spread of the virus and help people who may have the virus and do not know it from transmitting it to others. Wearing masks is broadly recognised as critical to reducing community transmission and limiting touching of the face.

In a time of concern about slowing the transmission of COVID-19, increased surveillance combined with AI solutions can improve monitoring and reduce the human effort needed to limit the spread of this disease. The objective of this challenge is to create an image classification machine learning model to accurately predict the likelihood that an image contains a person wearing a face mask, or not. The total dataset contains 1,800+ images of people either wearing masks or not.

Your machine learning solution will help policymakers, law enforcement, hospitals, and even commercial businesses ensure that masks are being worn appropriately in public. These solutions can help in the battle to reduce community transmission of COVID-19.

## Evaluation
The evaluation metric for this challenge is Area Under the Curve.

Your submission file should look like:
```
id                                    label
ttuqxjhrmdqhppfxrbzgyciipwdxcf.jpg     0.99
qmltykiislwklsklnzhcsrfsqwmaun.jpg     0.23
lkzeblenqbovljxpucpsufmprjxxqn.jpg     0.67
```

TODO:
1. -- Predict one photo in the detector nb
2. Convert the pipeline into .py file or download the model
3. Make a streamlit file where you can take pics or upload them and predict
