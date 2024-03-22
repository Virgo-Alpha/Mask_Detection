from fastai.learner import load_learner

# Load the model
model_path = 'study_nbs/mask_classifier_model.pkl'
# learn = load_learner(model_path)

if __name__ == '__main__':
    learn = load_learner(model_path, cpu=True)

def predict_mask(image):
    dl = learn.dls.test_dl([image])
    test_pred, _ = learn.get_preds(dl=dl)
    predicted_class_idx = test_pred.numpy()[0].argmax()
    return "masked" if predicted_class_idx == 1 else "not masked"
