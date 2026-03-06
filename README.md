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
```bash
python -m ui.dashboard
```

**To run the InceptionV3 Feature Extractor**
```bash
python -m utils.featureExtractor.InceptionV3_GlobalAvgPoolLayer_image_feature_extractor --input "relativePath/to/input_directory" --output "(OPTIONAL) relativePath/to/output_file"
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
- Update the target folder in [duplicate_finder.ps1](utils\preprocessor\duplicate_finder.ps1).
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
- [ ] Integrate Central Logging
    - [x] feature-extractor
        - [x] InceptionV3_GlobalAvgPoolLayer_image_feature_extractor
        - [x] ResNet50_image_feature_extractor
    - [x] metadata
        - [x] fileTypeIdentifier
        - [x] metadata_parser
    - [ ] preprocessor
        - [ ] csv_mapNmerge
        - [x] hf_to_image
        - [x] insta_profile_download
        - [ ] real_fake_csv_merger
- [ ] Convert hard-coded or input based to args (for UI)
    - [x] hf_to_image
    - [ ] insta_profile_download
- [x] Add `TARGET_USERNAME` file parser for [insta_profile_download](utils/preprocessor/insta_profile_download.py).
- [x] Merge metadata extractor, feature extractor, and preprocessors as a single unified tool.
- [ ] Create a pipeline to generate deepfakes.
- [ ] Issues:
    - [ ] Feature Extractor in Kali Linux is ending up in a dependency issue - tensorflow.
    - [ ] insta_profile_download creates a new folder in root for saving photos and videos instead of putting them in `instaProfiles/photos` or `instaProfiles/videos`.
- [ ] Start working on the Detector
    - [ ] Image Classifiers
        - [ ] ResNet50
            - [ ] imagenet
            - [ ] Open Images Dataset (by Google)
            - [ ] COCO (Common Objects in Context)
        - [ ] InceptionV3 (different layers)
            - [ ] imagenet
            - [ ] Open Images Dataset (by Google)
            - [ ] COCO (Common Objects in Context)
        - [ ] VGG16
            - [ ] imagenet
            - [ ] Open Images Dataset (by Google)
            - [ ] COCO (Common Objects in Context)
        - [ ] [Custom CNN](https://www.analyticsvidhya.com/blog/2020/02/learn-image-classification-cnn-convolutional-neural-networks-3-datasets/#h-steps-to-build-an-image-classification-model-using-cnn)
    - [ ] Feature Extractors
        - [ ] ResNet50
        - [ ] InceptionV3 (different layers)
        - [ ] [VGG16](https://stackoverflow.com/questions/56876348/how-many-features-is-vgg16-supposed-to-extract-when-used-as-a-pre-trained-featur)
        - [ ] Custom CNN
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
- [ ] Add time and resource consumption modules.
- [ ] Add support for scalpel. If embedded files steganography found, this will be used to extract all files.
- [ ] Fix all README.

> [!NOTE]
> **Args**:
> - /utils/featureExtractor/
>   - [InceptionV3_GlobalAvgPoolLayer_image_feature_extractor.py](/utils/featureExtractor/InceptionV3_GlobalAvgPoolLayer_image_feature_extractor.py): input, output (optional)
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
- Feature Extractor: ResNet50
- Data Split:
    - Train: 80%
    - Test: 20%

- **[Best Model](results/imageModels/ResNet50/32kModel/best_random_forest_model.joblib):**
    - n_estimators: 200 | max_depth: 20 | min_samples_split: 5 | min_samples_leaf: 1 | max_features: sqrt | bootstrap: false
    - [Report]()
        - **Accuracy:** 0.84
        - **Precision:** 0.84
        - **F1 Score:** 0.84
        - **MCC:** 0.6849507524194016
        - **Cohen's Kappa:** 0.6849311698838436
        - **Balanced Accuracy:** 0.8425499829355598
        - **ROC-AUC (weighted ovr):** N/A

**References:**
- [JamieWithofs/Deepfake-and-real-images-4](https://huggingface.co/datasets/JamieWithofs/Deepfake-and-real-images-4)
- [StyleGan-StyleGan2 Deepfake Face Images](https://www.kaggle.com/datasets/kshitizbhargava/deepfake-face-images)
- [Fake-Vs-Real-Faces (Hard)](https://www.kaggle.com/datasets/hamzaboulahia/hardfakevsrealfaces)
- Images scraped from Instagram and Reddit