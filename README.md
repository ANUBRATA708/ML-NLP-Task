🤖 ML/NLP Engineer Intern Challenge  
🎯 Objective  
Build a complete text classification pipeline using Hugging Face Transformers. Demonstrate your skills in NLP preprocessing, model fine-tuning, and evaluation.

📋 Task Overview  
- Selected the **Amazon Polarity** dataset (1M+ labeled samples for binary sentiment classification)  
- Preprocessed and tokenized using **Hugging Face AutoTokenizer**  
- Fine-tuned **DistilBERT (distilbert-base-uncased)** for binary classification  
- Evaluated using **F1 score, Precision, Recall, Accuracy**  
- Documented insights and improvement ideas  
- Bonus planned: Extend to multilingual use case using `xlm-roberta-base`

📁 Project Structure  
/notebooks/  
- **data_exploration.ipynb** - Dataset analysis, class distribution, sample exploration  
- **model_training.ipynb** - Interactive model training and experimentation  
- **evaluation_analysis.ipynb** - Results analysis, error analysis, visualizations  

/src/  
- **train_model.py** - Main training script with Hugging Face Trainer  
- **data_preprocessing.py** - Text cleaning, tokenization, dataset preparation  
- **model_utils.py** - Model loading, saving, prediction utilities  
- **config.py** - Training hyperparameters and model configurations  

/models/  
- **trained_model/** - Fine-tuned model weights and configuration  
- **tokenizer/** - Trained tokenizer files  
- **.gitkeep** - Maintains directory structure  

/reports/  
- **model_report.md** - Model architecture decisions, training insights, improvements  
- **evaluation_metrics.json** - Detailed metrics (F1, precision, recall, accuracy)  
- **confusion_matrix.png** - Classification results visualization  

**Root Files**  
- **requirements.txt** - Python dependencies (transformers, torch, datasets, etc.)  
- **README.md** - Project documentation (this file)  
- **submission.md** - Your approach, model decisions, and key learnings  
- **train.py** - Simple training script entry point  
- **.gitignore** - Files to exclude from git (models/, pycache, etc.)

🚀 Getting Started  
**Setup Environment**

```bash
python -m venv venv  
source venv/bin/activate  # Windows: venv\Scripts\activate  
pip install -r requirements.txt  
```

**Run Data Exploration**

```bash
jupyter notebook notebooks/data_exploration.ipynb  
```

**Train Model**

```bash
python train.py  
# or  
python src/train_model.py  
```

**Evaluate Results**

```bash
jupyter notebook notebooks/evaluation_analysis.ipynb  
```

📊 Dataset Requirements  
This project uses the **Amazon Polarity** dataset, which meets all requirements:  

- ✅ 1M+ labeled samples  
- ✅ Binary classification (positive vs. negative reviews)  
- ✅ English text  

Suggested alternatives if needed:
- IMDB Movie Reviews  
- AG News  
- Yelp Reviews  
- Twitter Sentiment datasets  

✅ Expected Deliverables  
- ✅ Working fine-tuned DistilBERT model with saved weights  
- ✅ Complete training pipeline using Hugging Face Transformers  
- ✅ Evaluation metrics in JSON format (F1, precision, recall, accuracy)  
- ✅ Model report with decisions and insights  
- ✅ Clear notebooks for exploration and analysis  
- ✅ Updated submission.md with approach and learnings  

🎯 Evaluation Focus  
- Quality of model selection and fine-tuning  
- Effectiveness of preprocessing/tokenization  
- Evaluation methodology and correct metrics  
- Clean, modular code and reproducibility  
- Depth of insights and learnings in documentation  

💡 Bonus Points  
- Planned multilingual extension using XLM-R  
- Advanced evaluation (e.g., ROC curves, class-wise F1)  
- Comparison with other models (e.g., BERT vs. DistilBERT)  
- Deployable inference-ready pipeline  
- Error analysis and failure case breakdown  

🔧 Key Technologies  
- **Hugging Face Transformers** - For model training and inference  
- **Datasets library** - For loading and managing text datasets  
- **PyTorch** - Deep learning training backend  
- **Scikit-learn** - Evaluation metrics and confusion matrix  
- **Matplotlib/Seaborn** - Visualizations and analysis