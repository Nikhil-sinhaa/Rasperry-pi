## Authentication

Install the download scripts' dependency:

```bash
python -m pip install -r data/requirements.txt
```

Before downloading the dataset, authenticate with Hugging Face:

```bash
hf auth login
```

The Hugging Face CLI will prompt for your access token. Then run the relevant script:

```bash
python data/open-india-legislation.py
python data/open-india-regulations.py

```