# DEEPFAKE DETECTION

Documentation is loosely updated.
```bash
$ git clone https://github.com/BumpiestDig10/deepfake-detector.git
$ python -m venv venv
$ source venv/bin/activate  # Linux
$ venv/Scripts/activate     # Windows
$ pip install -r allRequirements.txt
```

$~$

**To run the InceptionV3 Feature Extractor**
```bash
python -m utils.featureExtractor.InceptionV3_GlobalAvgPoolLayer_image_feature_extractor --input "relativePath/to/input_directory" --output "(OPTIONAL) relativePath/to/output_file"
```

**To run the Metadata Parser**
```bash
python -m utils.metadata.metadata_parser --input "relativePath/to/input_directory" --output "(OPTIONAL) relativePath/to/output_file"
```

**To run the Instagram Profile Downloader**
- Change the username and password in the [script](utils/preprocessor/insta_profile_download.py) if needed. The one mentioned is a burner and may or may not work for you.
```bash
python -m utils.preprocessor.insta_profile_download
```

$~$

### TODO: (for images branch)
- [ ] Integrate Central Logging
    - [ ] data-preprocessor
    - [ ] feature-extractor
        - [x] InceptionV3_GlobalAvgPoolLayer_image_feature_extractor
        - [ ] ResNet50_image_feature_extractor
    - [x] metadata
        - [x] fileTypeIdentifier
        - [x] metadata_parser
    - [ ] preprocessor
        - [ ] csv_mapNmerge
        - [ ] hf_to_image
        - [x] insta_profile_download
        - [ ] real_fake_csv_merger
- [ ] Add `TARGET_USERNAME` file parser for [insta_profile_download](utils/preprocessor/insta_profile_download.py).
- [ ] Merge metadata extractor and feature extractor as a single unified tool.
- [ ] Create a pipeline to generate deepfakes.
- [ ] Fix:
    - [ ] Feature Extractor in Kali Linux is ending up in a dependency issue - tensorflow.
- [ ] Start working on the Detector itself.
- [ ] Add time and resource consumption modules.
- [ ] Add support for scalpel. If embedded files steganography found, this will be used to extract all files.
- [ ] Fix all README.

$~$

> [!NOTE]
> Args:
> - /utils/featureExtractor/
>   - [InceptionV3_GlobalAvgPoolLayer_image_feature_extractor.py](/utils/featureExtractor/InceptionV3_GlobalAvgPoolLayer_image_feature_extractor.py): input, output (optional)
>   - [ResNet50_image_feature_extractor.py](/utils/featureExtractor/ResNet50_image_feature_extractor.py): input, output (optional)
> - /utils/metadata/
>   - [metadata_parser.py](/utils/metadata/metadata_parser.py): input, output (optional)
> - /utils/preprocessor/
>   - [csv_mapNmerge.py](/utils/preprocessor/csv_mapNmerge.py): base, label
>   - [real_fake_csv_merger.py](/utils/preprocessor/real_fake_csv_merger.py): real, fake, output (optional)