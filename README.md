## Introduction

This repo is for a project aimed to build a computer vision and deep learning based software, which provides aid to those with bad vision/memory, mostly aimed towards (but not limited to) the elderly, and visually impared. 

The software accepts input in forms of Live Video (via an in-built webcam or an external webcam), a pre-recorded Video, or an Image, you can then question the bot with what you may want to know regarding the provided media.

For installation plese follow the steps given below:

## Installation

### Manual Installation

- **Requirements**

There are some python packages and libraries, you'll need to install, these are given in the requirements.txt, please make sure you install these according to your preference, using

```python
pip install ddgs pillow requests tqdm
```
*Other requirements should be pre-installed*

*However, if you encounter any other errors, please look in the requirements.txt and install any of the packages not yet installed.*

*to check which packages you already have installed, use:*

```pip freeze```

- **Dataset**

Please refer to Dataset/README.md for more info. 

### Script Installation 

- **Windows** 

Download the zip from browser or run the following command on terminal
```cmd
git clone https://github.com/<username>/VisualRAG.git
cd VisualRAG
```

```
setup.bat
run.bat
```

- **Linux** 

Run the following command on your console
```bash
git clone https://github.com/<username>/VisualRAG.git
cd VisualRAG

chmod +x setup.sh run.sh

./setup.sh
./run.sh
```

---
## Script Information [?]

*This will be updated soon*

---
## Development

Frontend:

```bash
cd frontend
npm run dev
```

Tauri:

```bash
cd frontend
npm run tauri dev
```

---
