# DEEPFAKE DETECTION

```bash
$ git clone https://github.com/BumpiestDig10/deepfake-detector.git
$ python -m venv venv
$ source venv/bin/activate  # Linux
$ venv/Scripts/activate     # Windows
$ pip install -r allRequirements.txt
```

$~$

> $~$
> **_NOTE 1:_**
> You do not need to run the following commands for each sub-project individually if you run those listed above.
> ```bash
> $ python -m venv venv
> $ pip install -r requirements.txt
> ```
> $~$

> $~$
> **_NOTE 2:_**
> 1. Metadata Extractor/Parser works well in Kali Linux (at least).
> 2. Feature Extractor works in Windows 11 (at least) - some edits to be made (check _TODO_).
> $~$


_TODO:_ (No particular order)
1. Create Feature Extractor for other file types and integrate everything as one single unified tool.
2. _Fix:_
2.1. Feature Extractor in Kali Linux is ending up in a dependency issue - tensorflow
3. Start working on the Detector itself.