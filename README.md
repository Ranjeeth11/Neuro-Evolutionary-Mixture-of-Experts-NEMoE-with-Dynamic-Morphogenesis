# Neuro-Evolutionary Mixture-of-Experts (NEMoE) with Dynamic Morphogenesis

This repository contains the implementation, training logs, and experiments for **NEMoE**: a Transformer-based architecture augmented with **Neuro-Evolutionary Morphogenesis** — enabling experts to **split, prune, and evolve dynamically** during training.  
Designed to fit **Google Colab T4 GPU constraints**, this work demonstrates how adaptive expert growth can improve efficiency and representation power under limited compute budgets.

---

## 🔬 Research Motivation

Traditional Transformers and Mixture-of-Experts (MoE) architectures require **fixed expert counts** and suffer from either under-utilisation or over-parameterisation.  
Our approach introduces **evolutionary morphogenesis**:

- **Split**: Clone & mutate high-usage experts.  
- **Prune**: Remove under-utilised experts.  
- **Crossover**: Blend parameters between experts (genetic operator).  
- **Fitness tracking**: Age, probation, and usage-based fitness guide expert survival.  

This allows the model to **self-organise its capacity** during training, achieving a balance between efficiency and expressivity.

---

## ⚙️ Model Configuration

Final stable training run (Colab T4-optimised):

- **Model dim**: 384  
- **Layers**: 6  
- **Heads**: 6  
- **Experts per layer**: 2 → 4 (dynamic)  
- **Sequence length**: 64  
- **Batch size**: 1  
- **Gradient accumulation**: 16  
- **Learning rate**: 2e-4  
- **Total tokens target**: ~10M  
- **Morph interval**: every 1000 steps  
- **Datasets**:  
  - [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (60%)  
  - [Alpaca Cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) (20%)  
  - [WikiText-103](https://huggingface.co/datasets/wikitext) (20%)  

---

## 📊 Training Logs

Detailed logs are stored in:
- `training_log.txt` → human-readable event stream  
- `training_log.csv` → structured data for plots  

Example log snippet:

- [2025-09-13 20:37:59] Step 2000 | Tokens 126,976 | LossCE 3.0404 | LossLB 0.0486 | LR 2.00e-04 | Event evolve_split | Layer 0 | Action split | Expert 2 | Usage 0.3568 | Thresh 0.35 | 3→4
- [2025-09-13 20:37:59] Step 2000 | Tokens 126,976 | LossCE 3.0404 | LossLB 0.0486 | LR 2.00e-04 | Event evolve_prune | Layer 0 | Action prune | Expert 1 | Usage 0.2902 | Thresh 0.02 | 4→3


**Split triggered** when expert usage > 0.35  
**Prune triggered** when usage < 0.02  
**Checkpointing** every 2000 steps  
**Expert count per layer** tracked continuously in CSV  

---

## 📈 Results

- **Final Perplexity**: ~21.67  
- **Sample Generation (step 25k):**
  - Prompt: The quick brown fox
  - Output: The quick brown fox perennlington028 aren protagonistrelevantboarding Chiefviste storage Authorskef...

- Model demonstrates **novel text generation** but limited coherence (expected given compute + dataset scale).  
- Crucially, **training stability was preserved** despite evolutionary changes.  

---

## 🏗️ How to Reproduce

1. Clone repo & install requirements:
   ```bash
   git clone https://github.com/<your-username>/nemo-evolutionary-transformer.git
   cd nemo-evolutionary-transformer
   pip install -r requirements.txt
   ```
2. Run training in Colab / local:
   ```bash
   python train_nemoe.py
   ```
3. Logs & checkpoints will be stored in:
   ```bash
   /content/drive/MyDrive/NEMoE/
   ```
4. Resume training automatically from latest checkpoint:
   ```bash
   python train_nemoe.py --resume
   ```

---

## 📂 Repository Structure

```bash
├── train_nemoe.py        # Main training script
├── training_log.txt      # Human-readable logs
├── training_log.csv      # Structured logs for plotting
├── checkpoints/          # Model checkpoints (nemo_ckpt_stepXXXX.pt)
├── README.md             # This file
```

---

## 📑 Citation

If you use this work, please cite:
```bash
@article{narayanasamy2025nemo,
  title={Neuro-Evolutionary Mixture-of-Experts with Dynamic Morphogenesis under Compute Constraints},
  author={Ranjeeth Narayanasamy},
  year={2025},
  note={Work-in-progress research on adaptive expert growth in Transformers.}
}
```

---

## 🚀 Future Work

- Scale to longer sequences and larger datasets (Pile, C4).
- Compare with fixed MoE baselines.
- Explore reinforcement-guided expert evolution.
- Deploy smaller checkpoints as lightweight LMs for downstream NLP tasks.

## 👤 Author
Ranjeeth Narayanasamy
Early-career AI/ML researcher exploring resource-efficient Transformers and evolutionary learning methods.
