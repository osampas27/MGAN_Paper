# MGAN_Paper
is a novel deep learning framework designed for malicious traffic detection in network security. It combines Graph Attention Networks (GATs) for capturing spatial dependencies and Transformer-based sequence modeling for identifying temporal attack behaviors. 
# Installation instruction
pip install -r requirements.txt

## 🚀 Implementation

Here we provide the implementation of an **MGAN layer** in **TensorFlow**, along with a **minimal execution example** on the **CICIDS2017 dataset**.

### 📁 Repository Structure
The repository is organized as follows:

- **`dataset/`** – Contains the necessary dataset files for **CICIDS2017**.
- **`models/models.py`** – Implements the **MGAN model**.
- **`layers.py`** – Defines the **MultiGraphConvolution layer**.
- **`train.py`** – Combines all the components to execute a **full training run on CICIDS2017**.

---

### 🔧 **Usage**
To train MGAN on CICIDS2017, run:
```bash
python train.py --dataset dataset/cicids2017.csv --epochs 50
