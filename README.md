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
To run the Metadata Parser:
```bash
python -m utils.metadata.metadata_parser --input "path/to/input_directory" --output "(OPTIONAL) path/to/output_file"
```

$~$

### TODO: (for images branch)
- [ ] Fix args to be consistent throughout codebase (--input/-i, --output/-o, --help/-h).
- [ ] Integrate Central Logging
    - [ ] data-preprocessor
    - [ ] feature-extractor
    - [x] metadata
- [ ] Merge metadata extractor and feature extractor as a single unified tool.
- [ ] Create a pipeline to generate deepfakes.
- [ ] Fix:
    - [ ] Feature Extractor in Kali Linux is ending up in a dependency issue - tensorflow
- [ ] Start working on the Detector itself.
- [ ] Add time and resource consumption modules.
- [ ] Add support for scalpel. If embedded files steganography found, this will be used to extract all files.
- [ ] Fix all README.