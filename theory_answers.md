
# 🧠 Part 1: Theoretical Understanding

## Q1: Explain the primary differences between TensorFlow and PyTorch. When would you choose one over the other?

**TensorFlow** and **PyTorch** are both powerful deep learning frameworks, but they differ in several ways:

| Feature         | TensorFlow                          | PyTorch                             |
|----------------|-------------------------------------|-------------------------------------|
| Execution Mode | Static computation graph (TF 1.x); Eager mode added in TF 2.x | Dynamic computation graph (eager by default) |
| Syntax         | More complex; uses more abstractions | More Pythonic and intuitive        |
| Debugging      | Harder to debug (especially TF 1.x) | Easier to debug line-by-line       |
| Deployment     | Excellent deployment tools (TF Lite, TF Serving) | Less mature deployment pipeline   |
| Community      | Backed by Google; very popular in production | Popular in research; backed by Meta |

**When to use:**
- Choose **TensorFlow** for: Production systems, deployment to mobile/web, or scalability
- Choose **PyTorch** for: Research, experimentation, academic work, or faster prototyping

---

## Q2: Describe two use cases for Jupyter Notebooks in AI development.

1. **Interactive Model Development:**  
   Jupyter allows AI developers to write, test, and visualize code in blocks, which is ideal for building and tuning machine learning models step by step.

2. **Data Exploration and Visualization:**  
   Jupyter integrates well with tools like Pandas, Matplotlib, and Seaborn to analyze datasets, visualize trends, and display results interactively — great for EDA (Exploratory Data Analysis).

---

## Q3: How does spaCy enhance NLP tasks compared to basic Python string operations?

**spaCy** is a modern NLP library that outperforms traditional string operations by:

- Offering **pre-trained models** for tokenization, POS tagging, dependency parsing, and named entity recognition (NER)
- Processing text **faster** and more **accurately**
- Understanding linguistic context (e.g., knowing that “Apple” can be a brand or a fruit)
- Supporting **custom pipelines** and rule-based matching

In contrast, Python string operations like `.split()` or `.replace()` lack context, are language-agnostic, and are prone to errors when processing complex sentences.

---

## Q4: Comparative Analysis – Scikit-learn vs TensorFlow

| Criteria           | Scikit-learn                               | TensorFlow                           |
|--------------------|---------------------------------------------|--------------------------------------|
| **Target Applications** | Classical ML (SVM, Decision Trees, etc.) | Deep Learning (CNNs, RNNs, etc.)     |
| **Ease of Use**        | Very beginner-friendly                   | Slightly complex (especially for deep learning) |
| **Community Support**  | Strong in academia and production        | Backed by Google, very large ecosystem |
