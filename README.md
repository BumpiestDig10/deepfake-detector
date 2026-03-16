# DEEPFAKE DETECTION

Documentation is loosely updated.
```bash
$ git clone https://github.com/BumpiestDig10/deepfake-detector.git
$ python -m venv venv
$ source venv/bin/activate  # Linux
$ venv/Scripts/activate     # Windows
$ pip install -r allRequirements.txt
```

-------

**To run the Deepfake Training Orchestrator** (Dashboard with all the tools)
- On the dashboard, paramaeters for all tools will have "Browse Files" and "Browse Folders" buttons, be smart about what you should actually input.
```bash
python -m ui.dashboard
```

**To calculate SHA256 hash of a file**
```bash
python -m utils.filehash --input "path/to/inputFile"
```

**To run the InceptionV3 Feature Extractor**
```bash
python -m utils.featureExtractor.InceptionV3_image_feature_extractor --input "relativePath/to/input_directory" --output "(OPTIONAL) relativePath/to/output_file" --weights "(OPTIONAL) imagenet"
```

**To run the ResNet50 Feature Extractor**
```bash
python -m utils.featureExtractor.ResNet50_image_feature_extractor --input "relativePath/to/input_directory" --output "(OPTIONAL) relativePath/to/output_file" --weights "(OPTIONAL) imagenet"
```

**To run the Metadata Parser**
```bash
python -m utils.metadata.metadata_parser --input "relativePath/to/input_directory" --output "(OPTIONAL) relativePath/to/output_file"
```

**To download image datasets from Hugging Face**
```bash
python -m utils.preprocessor.hf_to_image --dataset "huggingFace/Dataset" --split "(OPTIONAL) train" --output "(OPTIONAL) relativePath/to/output_directory" --token "(OPTIONAL) huggingFaceAccessToken"
```

**To run the Instagram Profile Downloader** (no private accounts)
- Change the username and password in the [script](utils/preprocessor/insta_profile_download.py) if needed. The one mentioned is a burner and may or may not work for you.
- Create `instaProfile/usernames.txt` to sequentially download for each username mentioned. Not required for single profile.
```bash
python -m utils.preprocessor.insta_profile_download
```

**To run the Reddit Downloader**
- This tool works as a Chrome extension to bypass Reddit login issues.
- Downloads everything to the `Downloads/` folder.
- Logs can be checked using Chrome's DevTools Console.
```txt
- Navigate to chrome://extensions/ using Google Chrome.
- Enable Developer Mode.
- Select "Load Unpacked".
- Select the folder with all files of the extension. Default should be utils/preprocessor/reddit_downloader.
```

**To find and delete duplicate files in a particular directory** (Windows-only)
- Update the target folder in [duplicate_finder.ps1](utils/preprocessor/duplicate_finder.ps1).
- Execute the script in powershell.
    ```bash
    $ ./utils/preprocessor/duplicate_finder.ps1 # From project root directory or
    $ ./duplicate_finder.ps1                    # if CWD = utils/preprocessor/
    ```

**To Train a Random Forest Model**
```bash
python -m notebooks.RandomForestTrainer --input "path/to/features.csv" --output "(OPTIONAL) path/to/outputDirectory" --test_size (OPTIONAL) 0.2 --random_state (OPTIONAL) 420
```
-------

### TODO: (for images branch)
- [ ] Copy ToDo to GitHub Projects
- [ ] Convert hard-coded or input based to args (for UI)
    - [ ] [insta_profile_download](utils/preprocessor/insta_profile_download.py)
        - [ ] Target username (single and file)
        - [ ] Output directory
- [ ] Create a pipeline to generate deepfakes.
- [ ] Start working on the Detector
    - [ ] Feature Extractors
        - [ ] ResNet50
            - [x] imagenet
            - [ ] Open Images Dataset (by Google)
            - [ ] COCO (Common Objects in Context)
        - [ ] InceptionV3 (different layers)
            - [x] imagenet
            - [ ] Open Images Dataset (by Google)
            - [ ] COCO (Common Objects in Context)
        - [ ] [VGG16](https://stackoverflow.com/questions/56876348/how-many-features-is-vgg16-supposed-to-extract-when-used-as-a-pre-trained-featur)
            - [ ] imagenet
            - [ ] Open Images Dataset (by Google)
            - [ ] COCO (Common Objects in Context)
    - [ ] Feature Classifiers
        - [ ] Random Forest
        - [ ] XGBoost
        - [ ] LightGBM
        - [ ] Linear SVM
        - [ ] Regularized Logistic Regression
        - [ ] Custom Neural Network
    - [ ] Feature Clustering (with Principal Component Analysis)
        - [ ] K Means Clustering
        - [ ] Gaussian Mixture Models
        - [ ] Hierarchical Clustering
        - [ ] DBSCAN
    - [ ] Image Classifiers
        - [ ] [Custom CNN](https://www.analyticsvidhya.com/blog/2020/02/learn-image-classification-cnn-convolutional-neural-networks-3-datasets/#h-steps-to-build-an-image-classification-model-using-cnn)
- [ ] Add time and resource consumption modules.
- [ ] Add support for scalpel. If embedded files steganography found, this will be used to extract all files.
- [ ] Fix all README.

> [!NOTE]
> **Args**:
> - [filehash.py](utils/filehash.py): input
> - /utils/featureExtractor/
>   - [InceptionV3_image_feature_extractor.py](/utils/featureExtractor/InceptionV3_image_feature_extractor.py): input, output (optional), weights (optional)
>   - [ResNet50_image_feature_extractor.py](/utils/featureExtractor/ResNet50_image_feature_extractor.py): input, output (optional), weights (optional)
> - /utils/metadata/
>   - [metadata_parser.py](/utils/metadata/metadata_parser.py): input, output (optional)
> - /utils/preprocessor/
>   - [csv_mapNmerge.py](/utils/preprocessor/csv_mapNmerge.py): base, label
>   - [hf_to_image.py](/utils/preprocessor/hf_to_image.py): dataset, split (optional), output (optional), token (optional)
>   - [real_fake_csv_merger.py](/utils/preprocessor/real_fake_csv_merger.py): real, fake, output (optional)
> - /notebooks/
>   - [RandomForestTrainer.py](/notebooks/RandomForestTrainer.py): input, output (optional), test_size (optional), random_state (optional)
>
> **Labels**:
> - Real = 1
> - Fake = 0

## RESULTS

### Model 1
- Model Type: Random Forest
- Dataset Type: Images
- Dataset Size: 31,762
    - Real: 15,364
    - Fake: 16,398
- Feature Extractor: ResNet50 (imagenet)
- Data Split:
    - Train: 80% (25409 images)
    - Test: 20% (6353 images)

- **[Best Model](results/imageModels/ResNet50_imagenet/32kModel/best_random_forest_model.joblib)**
    - n_estimators: 150 | max_depth: null | min_samples_split: 5 | min_samples_leaf: 1 | max_features: 0.2 | bootstrap: false
    - [Report](results/imageModels/ResNet50_imagenet/32kModel/classification_report.txt)
        - Accuracy: 0.854
        - Precision: 0.854
        - F1 Score: 0.854
        - Matthews Correlation Coefficient: 0.7070566271285679
        - Cohen's Kappa: 0.7070160654228586
        - Balanced Accuracy: 0.8536314517473194
        - ROC-AUC (weighted ovr): N/A

**Datasets Used:**
- [JamieWithofs/Deepfake-and-real-images-4](https://huggingface.co/datasets/JamieWithofs/Deepfake-and-real-images-4)
- [StyleGan-StyleGan2 Deepfake Face Images](https://www.kaggle.com/datasets/kshitizbhargava/deepfake-face-images)
- [Fake-Vs-Real-Faces (Hard)](https://www.kaggle.com/datasets/hamzaboulahia/hardfakevsrealfaces)
- Images scraped from Instagram and Reddit

### Model 2
- Model Type: 
- Dataset Type: Images
- Dataset Size: 
    - Real: 
    - Fake: 
- Feature Extractor: 
- Data Split:
    - Train: 
    - Test: 

- **[Best Model] < link >**
    - n_estimators:  | max_depth:  | min_samples_split:  | min_samples_leaf:  | max_features:  | bootstrap: 
    - [Report] < link >
        - Accuracy: 
        - Precision: 
        - F1 Score: 
        - Matthews Correlation Coefficient: 
        - Cohen's Kappa: 
        - Balanced Accuracy: 
        - ROC-AUC (weighted ovr): 

**Datasets Used**
- [Stable Diffusion Face Dataset](https://www.kaggle.com/datasets/mohannadaymansalah/stable-diffusion-dataaaaaaaaa?resource=download)
- [metfaces-dataset](https://github.com/NVlabs/metfaces-dataset)
- [Human Images Dataset - Men and Women](https://www.kaggle.com/datasets/snmahsa/human-images-dataset-men-and-women)
- [A Generated Face Dataset: AGFD-20K](https://github.com/Robin-WZQ/AGFD-20K)
- [Academic Dataset by Generated Photos](https://generated.photos/datasets/academic)
- [Celebrity Face Image Dataset](https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset)
- [CelebaHQ](https://github.com/tkarras/progressive_growing_of_gans)
- [SSHQ-1.0](https://github.com/stylegan-human/StyleGAN-Human) Password: StylisH-HumanS-hq_1.0
- [SFHQ-T2I: Synthetic Faces from Text 2 Image models](https://www.kaggle.com/datasets/selfishgene/sfhq-t2i-synthetic-faces-from-text-2-image-models)

- [ffhq-dataset](https://github.com/NVlabs/ffhq-dataset) Thumbnails Only