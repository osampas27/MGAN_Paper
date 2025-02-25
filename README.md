# MGAN_Paper
MGAN is a novel deep learning framework designed for malicious traffic detection in network security. It combines Graph Attention Networks (GATs) for capturing spatial dependencies and Transformer-based sequence modeling for identifying temporal attack behaviors. 
# Installation instruction
pip install -r requirements.txt

## 🚀 Implementation

Here we provide the implementation of **MGAN (Multi-View Graph Attention Network)** in **TensorFlow/PyTorch**, along with a **minimal execution example** on the **CICIDS2017 dataset**.

### 📁 Repository Structure
The repository is organized as follows:

- **`dataset/`** – Contains the necessary dataset files for **CICIDS2017**.
- **`models/`** – Contains the full implementation of **MGAN**, including:
  - **`mgan_model.py`** – The main MGAN architecture combining Graph Attention Networks (GATs) and Transformers.
  - **`graph_attention.py`** – Implements the multi-head **Graph Attention layer**.
  - **`transformer_encoder.py`** – Defines the **Transformer-based sequence encoder** for temporal modeling.
  - **`loss_functions.py`** – Includes **contrastive loss** and **cross-entropy loss**.
- **`layers.py`** – Implements additional **custom layers** used in MGAN.
- **`train.py`** – Combines all the components to execute a **full training run** on **CICIDS2017**.

---

### 🔧 **Usage**
#### **1️⃣ Dataset Preparation**
Ensure the **CICIDS2017 dataset** is placed inside the `dataset/` folder. Then, preprocess it:
```bash
python dataset/preprocessing.py --input dataset/cicids2017_raw.csv --output dataset/cicids2017_processed.csv --normalize --encode_labels


python evaluate.py --model_path results/mgan_model.pth
