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

$~$

**To run the InceptionV3 Feature Extractor**
```bash
python -m utils.featureExtractor.InceptionV3_GlobalAvgPoolLayer_image_feature_extractor --input "relativePath/to/input_directory" --output "(OPTIONAL) relativePath/to/output_file"
```

**To run the ResNet50 Feature Extractor**
```bash
python -m utils.featureExtractor.ResNet50_image_feature_extractor --input "relativePath/to/input_directory" --output "(OPTIONAL) relativePath/to/output_file"
```

$~$

**To run the Metadata Parser**
```bash
python -m utils.metadata.metadata_parser --input "relativePath/to/input_directory" --output "(OPTIONAL) relativePath/to/output_file"
```

$~$

**To download image datasets from Hugging Face**
- Make sure to change the name and split of the dataset before running.
```bash
python -m utils.preprocessor.hf_to_image
```

**To run the Instagram Profile Downloader** (no private accounts)
- Change the username and password in the [script](utils/preprocessor/insta_profile_download.py) if needed. The one mentioned is a burner and may or may not work for you.
- Create `instaProfile/usernames.txt` to sequentially download for each username mentioned. Not required for single profile.
```bash
python -m utils.preprocessor.insta_profile_download
```

$~$

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
    - [ ] hf_to_image
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
        - [ ] InceptionV3 (different layers)
        - [ ] VGG16
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

$~$

> [!NOTE]
> **Args**:
> - /utils/featureExtractor/
>   - [InceptionV3_GlobalAvgPoolLayer_image_feature_extractor.py](/utils/featureExtractor/InceptionV3_GlobalAvgPoolLayer_image_feature_extractor.py): input, output (optional)
>   - [ResNet50_image_feature_extractor.py](/utils/featureExtractor/ResNet50_image_feature_extractor.py): input, output (optional)
> - /utils/metadata/
>   - [metadata_parser.py](/utils/metadata/metadata_parser.py): input, output (optional)
> - /utils/preprocessor/
>   - [csv_mapNmerge.py](/utils/preprocessor/csv_mapNmerge.py): base, label
>   - [real_fake_csv_merger.py](/utils/preprocessor/real_fake_csv_merger.py): real, fake, output (optional)
>
> **Labels**:
> - Real = 1
> - Fake = 0