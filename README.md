# NLU_projects (p1 + p2)

This repository has two independent modules:
- `p1`: custom Word2Vec (CBOW / SGNS) from scratch + gensim + analysis
- `p2`: character-level name generation (RNN / BLSTM / Attention)

> Run all commands from the repository root:
> `e:\AdityaPratapSingh\adi\Projects\NLU_projects`

---

## 🧩 Global setup

1. Create/activate Python environment:
   - `python -m venv .venv`
   - Windows: `.venv\Scripts\activate`
2. Install dependencies:
   - `pip install torch torchvision torchaudio`
   - `pip install numpy scipy scikit-learn matplotlib gensim requests beautifulsoup4 wordcloud`
   - `pip install pytest` (optional)
3. Optional GPU support: install correct `torch` package from https://pytorch.org.

---

## 📁 Problem 1 (`p1`) – Word2Vec pipeline

### 1. Collect raw text
- `python p1/collect_data.py`
- Output: `raw_corpus.txt`, plus `p1/logs/collect_data.log`.

### 2. Preprocess
- `python p1/preprocess.py`
- Output: `cleaned_corpus.txt`, `wordcloud.png`, plus `p1/logs/preprocess.log`.

### 3. Train Word2Vec
- `python p1/train_word2vec.py`
- Output:
  - `p1/models/` (gensim models like `cbow_..._lib.model`, `sgns_..._lib.model`)
  - `p1/models_scratch/` (`*.pkl`)
  - `experiment_results.json`
  - `p1/logs/train.log`
- Configurable in `p1/train_word2vec.py`:
  - `EXPERIMENT_CONFIGS`, `EPOCHS_SCRATCH`, `EPOCHS_GENSIM`, etc.

### 4. Analyze + visualize
- `python p1/analyze_and_visualize.py`
- Output:
  - `visualization_pca_*.png`
  - `visualization_tsne_*.png`
  - `p1/logs/analyze.log`
  - nearest neighbours + analogies printed in log

### Optional checks
- Verify `p1/models/` contains files such as:
  - `cbow_d100_w5_n5_lib.model`
  - `sgns_d100_w5_n5_lib.model`
- Verify `p1/models_scratch/` contains `*.pkl` model checkpoints.

---

## 📁 Problem 2 (`p2`) – char-level name generation

### 0. Data requirement
- Required: `p2/TrainingNames.txt` (already present)
- If not present, create one name per line manually.
- (README mentions `p2/generate_dataset.py`, but in this repository there is no file; use manual `TrainingNames.txt`)

### 1. Train models
- `python p2/train.py`
- It trains models:
  - `rnn`, `blstm`, `attention`
- Outputs:
  - `p2/checkpoints/{rnn,blstm,attention}_best.pt`
  - `p2/logs/train_*.log`
- Config:
  - `p2/train.py -> CONFIGS` (epochs, batch_size, lr, etc)
  - change `models_to_train` if you want a subset.

### 2. Evaluate + generate names
- `python p2/evaluate.py`
- Outputs:
  - `p2/generated_names.txt`
  - evaluation metrics table in log
  - `p2/logs/evaluate.log`

### 3. (Optional) inspect models
- `p2/data_utils.py` for `Vocabulary`, `build_dataloader`
- `p2/model_rnn.py`, `p2/model_blstm.py`, `p2/model_attention_rnn.py` for architecture

---

## 📝 Quick run commands (all together)

```powershell
cd e:\AdityaPratapSingh\adi\Projects\NLU_projects

# p1 workflow
python collect_data.py
python preprocess.py
python train_word2vec.py
python analyze_and_visualize.py

# p2 workflow
python train.py
python evaluate.py