# 📊 Employee Attrition Analysis

A complete data science project analyzing employee turnover using the IBM HR Analytics dataset. This project includes data preprocessing, exploratory data analysis (EDA), statistical testing, predictive modeling, and a professional PDF report generated via LaTeX.

---
## 🎯 Problem Statement

Employee attrition, or the voluntary departure of employees, poses a significant challenge for organizations, leading to increased recruitment costs, loss of expertise, and reduced productivity. The Employee Attrition Analysis project aims to address this issue by analyzing the IBM HR Analytics Employee Attrition dataset to identify key factors driving employee turnover. The project seeks to answer: *What are the primary drivers of attrition, and how can organizations mitigate them?*

Using data preprocessing, exploratory data analysis (EDA), statistical testing, and predictive modeling, the project uncovers patterns and significant factors influencing attrition. It generates 14 visualizations and employs models like Logistic Regression, Random Forest, and Decision Tree to predict turnover. The resulting insights are compiled into a professional LaTeX report with actionable recommendations, such as increasing salaries for low-income employees or reducing frequent business travel, to help HR professionals develop targeted retention strategies.

## 📌 Project Highlights

- 🔍 **Data Cleaning**: Removes redundant columns, handles missing values, and encodes categorical variables.
- 📈 **EDA Visualizations**: Generates 14 insightful plots including bar plots, boxplots, countplots with counts, pairplots, histograms, and a correlation heatmap.
- 🧪 **Statistical Testing**: Uses Shapiro-Wilk, Chi-square, t-tests, and Mann-Whitney U to discover significant attrition factors.
- 🤖 **Predictive Modeling**: Builds Logistic Regression, Random Forest, and Decision Tree models (~85% accuracy).
- 📄 **LaTeX Report**: Generates a professional-quality `attrition_report.pdf` with plots, insights, and HR-focused recommendations.

---

## 🛠️ Prerequisites

### Python & Libraries

Install Python 3.8+ and required packages:

```bash
pip install pandas matplotlib seaborn scipy numpy scikit-learn statsmodels
```
### Tools
- **Graphviz**: For converting the decision tree DOT file to PNG
- **LaTeX Distribution**: MiKTeX (Windows) or TeX Live (macOS/Linux) for compiling the report
- **VS Code (optional)**: With LaTeX Workshop extension for LaTeX compilation
- **Dataset**: Attrition Rate Analysis.csv (IBM HR Analytics dataset)

## 🚀 Setup Instructions
**1. Clone the Repository**

```bash
git clone https://github.com/SaiRakshithaa/Employee-Attrition-Analysis.git
cd Employee-Attrition-Analysis
```

**2. Install Python Dependencies**

- Install Python 3.8+.
- Install required libraries:
```bash
pip install pandas matplotlib seaborn scipy numpy scikit-learn statsmodels
```

**3. Install Graphviz**

- **Windows:**
  - Download and install Graphviz.
  - Add to PATH (e.g., `C:\Program Files\Graphviz\bin`):
    - Right-click “This PC” > Properties > Advanced system settings > Environment Variables > System PATH > Edit.


  - Verify:
  ```bash
  dot --version
  ```

- macOS/Linux:
```bash
sudo apt-get install graphviz  # Ubuntu
brew install graphviz         # macOS
dot --version
```


**4. Install LaTeX Distribution**

- **Windows (MiKTeX)**:
 - Install MiKTeX.
 - Add to PATH (e.g., `C:\Program Files\MiKTeX\bin\x64`).
 - Update MiKTeX in MiKTeX Console (Updates > Check for Updates > Update Now).
 - Verify:
  ```bash
  pdflatex --version
  ```




 - macOS (MacTeX):
   
```bash
brew install mactex
pdflatex --version
```


 - Linux (TeX Live):

```bash
sudo apt-get install texlive-full
pdflatex --version
```



## 📖 Usage

**Prepare the Dataset**:

- Place `Attrition Rate Analysis.csv` in the project root (`HR_analytics_project/`).
- Update the path in `main.py` (line ~20):
```python
dataset = pd.read_csv('Attrition Rate Analysis.csv')
```

**Run the Python Script**:
```bash
python attrition_analysis.py
```

 - Outputs:

   - `plots/` folder with 14 PNG visualizations (e.g., `attrition_department_countplot.png` with counts).
   - `EmployeeAttrition.dot` (decision tree).
   - `significant_factors_attrition.txt` (significant factors and recommendations).
   - `inferences.txt` (visualization inferences with counts/percentages).


**Generate Decision Tree Image**:
```bash
dot -Tpng EmployeeAttrition.dot -o plots/decision_tree.png
```


**Compile the LaTeX Report**:

 - **VS Code**:
   - Open `attrition_report.tex`.
   - Compile: `Ctrl+Alt+B` or save (`Ctrl+S`) with “onSave” enabled.
   - View: `Ctrl+Alt+V`.


 - **Command Line**:
```bash
cd Employee-Attrition-Analysis
pdflatex attrition_report.tex
pdflatex attrition_report.tex
```

**Verify Outputs**:

 - Check `attrition_report.pdf` for visualizations (e.g., Section 3.6 for Department countplot with counts).
 - Verify counts in `inferences.txt` or run:
```python
print(old.groupby(['Department', 'Attrition']).size())
```


## 📂 File Structure
```plain
Employee-Attrition-Analysis/
├── Attrition Rate Analysis.csv     # Input dataset
├── main.py           # Python script for analysis
├── attrition_report.tex            # LaTeX report file
├── plots/                          # Generated PNG visualizations
│   ├── attrition_bar.png
│   ├── attrition_department_countplot.png
│   ├── decision_tree.png
│   └── ...
├── EmployeeAttrition.dot           # Decision tree DOT file
├── significant_factors_attrition.txt # Significant factors
├── inferences.txt                  # Visualization inferences
└── README.md                       # This file
```

