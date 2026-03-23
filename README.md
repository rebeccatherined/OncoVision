# BreastAI Project Structure

## Folder Setup
```
breast_cancer_detection/
├── app.py
├── utils.py
├── explainability.py
├── requirements.txt
├── ensemble_meta.pkl          <- ROOT (not in models/)
├── ultrasound_meta.pkl        <- ROOT
├── density_meta.pkl           <- ROOT
├── class_indices_mammogram.json  <- ROOT
├── class_indices_ultrasound.json <- ROOT
└── models/
    ├── model_custom.weights.h5
    ├── model_vgg_feature.weights.h5
    ├── model_vgg_finetune.weights.h5
    ├── model_densenet.weights.h5
    ├── model_mobilenet.weights.h5
    ├── model_ultrasound_custom.weights.h5
    ├── model_ultrasound_densenet.weights.h5
    ├── model_ultrasound_mobilenet.weights.h5
    ├── model_ultrasound_inceptionv3.weights.h5
    └── model_density.weights.h5
```

## Run
pip install -r requirements.txt
streamlit run app.py