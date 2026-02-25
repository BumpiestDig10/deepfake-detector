# DEEPFAKE DETECTION

Documentation is loosely updated.
```bash
$ git clone https://github.com/BumpiestDig10/deepfake-detector.git
$ python -m venv venv
$ source venv/bin/activate  # Linux
$ venv/Scripts/activate     # Windows
$ pip install -r allRequirements.txt
```

To run the Metadata Parser
```bash
python -m utils.metadata.src.metadata_parser --input "path/to/input_files" --output "(OPTIONAL) path/to/output_files"
```

$~$

### TODO: (for images branch)
1. Integrate Central Logging.
2. Merge metadata extractor and feature extractor as one single unified tool.
3. Create a pipeline to generate deepfakes.
4. Fix:
4.1. Feature Extractor in Kali Linux is ending up in a dependency issue - tensorflow
5. Start working on the Detector itself.
6. Add time and resource consumption modules.