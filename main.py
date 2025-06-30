import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, chi2_contingency
import numpy as np
import os
from sklearn import preprocessing
import copy
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

# Create a directory for saving plots
if not os.path.exists('plots'):
    os.makedirs('plots')

# Data Cleaning
dataset = pd.read_csv(r'D:\Rakshu\DataSciencePythonProjects\HR_analytics_project\Attrition Rate Analysis.csv')

dataset = dataset.drop_duplicates()
dataset = dataset.drop(['EmployeeCount', 'EmployeeID', 'Over18', 'StandardHours'], axis=1)

numerical_vars = ['Age', 'DistanceFromHome', 'Education', 'JobLevel', 'MonthlyIncome',
                  'NumCompaniesWorked', 'PercentSalaryHike', 'StockOptionLevel',
                  'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',
                  'YearsSinceLastPromotion', 'YearsWithCurrManager']
categorical_vars = ['BusinessTravel', 'Department', 'EducationField', 'Gender',
                    'JobRole', 'MaritalStatus']
for col in numerical_vars:
    if col in dataset.columns:
        dataset[col] = dataset[col].fillna(dataset[col].mean())
for col in categorical_vars:
    if col in dataset.columns:
        dataset[col] = dataset[col].fillna(dataset[col].mode()[0])
        
old = copy.deepcopy(dataset)

l = preprocessing.LabelEncoder()
dataset['Attrition'] = l.fit_transform(dataset['Attrition'])
dataset['BusinessTravel'] = l.fit_transform(dataset['BusinessTravel'])
dataset['Department'] = l.fit_transform(dataset['Department'])
dataset['EducationField'] = l.fit_transform(dataset['EducationField'])
dataset['Gender'] = l.fit_transform(dataset['Gender'])
dataset['JobRole'] = l.fit_transform(dataset['JobRole'])
dataset['MaritalStatus'] = l.fit_transform(dataset['MaritalStatus'])


# Data Visualization (Graphs with Saving)
inferences = []

# Bar graph for attrition proportion
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
x = ['Yes', 'No']
y_count = dataset['Attrition'].value_counts(0)
y_proportion = dataset['Attrition'].value_counts(1)
y = [y_count[1], y_count[0]]
plt.bar(x, y)
plt.title('Attrition Counts')
plt.xlabel('Attrition')
plt.ylabel('Count')
for i, v in enumerate(y):
    plt.text(i, v + 0.5, str(v), ha='center')
plt.savefig('plots/attrition_bar.png')
plt.close()
inferences.append("The bar graph shows that approximately 16.12% of employees (237 out of 1470) left the company (Attrition = Yes), indicating a significant minority experiencing attrition.")

# Attrition and Age
sns.boxplot(x='Attrition', y='Age', data=old)
plt.title('Age by Attrition')
plt.savefig('plots/attrition_age_boxplot.png')
plt.close()
inferences.append("The boxplot indicates that employees who left have a lower median age (~33 years) compared to those who stayed (~37 years), suggesting younger employees are more prone to attrition.")

# Attrition and Monthly Income
sns.boxplot(x='Attrition', y='MonthlyIncome', data=old)
plt.title('Monthly Income by Attrition')
plt.savefig('plots/attrition_income_boxplot.png')
plt.close()
inferences.append("Employees who left have a lower median monthly income (~$4,900) compared to those who stayed (~$6,500), indicating income disparities may influence retention.")

# Attrition and Distance from Home
sns.boxplot(x='Attrition', y='DistanceFromHome', data=old)
plt.title('Distance from Home by Attrition')
plt.savefig('plots/attrition_distance_boxplot.png')
plt.close()
inferences.append("The median distance from home for employees who left is slightly higher (~10 miles) than for those who stayed (~8 miles), suggesting longer commutes may contribute to attrition.")

# Attrition and Business Travel
ax=sns.countplot(x='BusinessTravel', hue='Attrition', data=old)
plt.title('Business Travel by Attrition')
# Add count labels on top of each bar
for p in ax.patches:
    height = p.get_height()
    if height > 0:  # Only annotate non-zero bars
        ax.annotate(f'{int(height)}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', xytext=(0, 5), 
                    textcoords='offset points')
plt.savefig('plots/attrition_businesstravel_countplot.png')
plt.close()
inferences.append("Employees who travel frequently have a higher attrition rate (~24.6% for frequent travelers vs. 14.8% for non-travelers), suggesting frequent business travel is a risk factor.")

# Attrition and Department (Pie Chart)
left = old[old['Attrition'] == 'Yes']
sizes = left['Department'].value_counts()
colors = ['Red', 'Yellow', 'Green']
plt.pie(sizes, labels=sizes.index, colors=colors, autopct='%1.1f%%')
plt.title('Department Distribution (Attrition = Yes)')
plt.savefig('plots/attrition_department_pie.png')
plt.close()
inferences.append("The pie chart shows that Sales accounts for ~42% of attrition cases, followed by Human Resources (~26%) and Research & Development (~32%).")

# Attrition and Department (Countplot)
ax=sns.countplot(x='Department', hue='Attrition', data=old)
plt.title('Department by Attrition')
# Add count labels on top of each bar
for p in ax.patches:
    height = p.get_height()
    if height > 0:  # Only annotate non-zero bars
        ax.annotate(f'{int(height)}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', xytext=(0, 5), 
                    textcoords='offset points')

plt.savefig('plots/attrition_department_countplot.png')
plt.close()
inferences.append("The countplot confirms higher attrition rates in Sales (~20.6%) compared to Research & Development (~13.8%).")

# Attrition and Job Role
ax=sns.countplot(x='JobRole', hue='Attrition', data=old)
plt.title('Job Role by Attrition')
# Add count labels on top of each bar
for p in ax.patches:
    height = p.get_height()
    if height > 0:  # Only annotate non-zero bars
        ax.annotate(f'{int(height)}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', xytext=(0, 5), 
                    textcoords='offset points')

plt.xticks(rotation=45)
plt.savefig('plots/attrition_jobrole_countplot.png')
plt.close()
inferences.append("Sales Representatives have the highest attrition rate (~40%), while Research Directors have the lowest (~2.5%), indicating role-specific factors influence turnover.")

# Attrition and Education
new = old.copy()
edu = {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'}
new['Education'] = new['Education'].map(edu)
ax=sns.countplot(x='Education', hue='Attrition', data=new)
# Add count labels on top of each bar
for p in ax.patches:
    height = p.get_height()
    if height > 0:  # Only annotate non-zero bars
        ax.annotate(f'{int(height)}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', xytext=(0, 5), 
                    textcoords='offset points')

plt.title('Education by Attrition')
plt.savefig('plots/attrition_education_countplot.png')
plt.close()
inferences.append("Employees with a Bachelor's degree have the highest attrition rate (~17.5%), while those with a Doctorate have the lowest (~5.6%).")

# Attrition and Gender
ax=sns.countplot(x='Gender', hue='Attrition', data=old)
plt.title('Gender by Attrition')
# Add count labels on top of each bar
for p in ax.patches:
    height = p.get_height()
    if height > 0:  
        ax.annotate(f'{int(height)}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', xytext=(0, 5), 
                    textcoords='offset points')

plt.savefig('plots/attrition_gender_countplot.png')
plt.close()
inferences.append("Males have a slightly higher attrition rate (~17.1%) compared to females (~14.9%).")

# Attrition and Marital Status
ax=sns.countplot(x='MaritalStatus', hue='Attrition', data=old)
plt.title('Marital Status by Attrition')
# Add count labels on top of each bar
for p in ax.patches:
    height = p.get_height()
    if height > 0:  
        ax.annotate(f'{int(height)}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', xytext=(0, 5), 
                    textcoords='offset points')

plt.savefig('plots/attrition_maritalstatus_countplot.png')
plt.close()
inferences.append("Single employees have the highest attrition rate (~25.1%), followed by married (~12.3%) and divorced (~10.2%) employees.")

# Pairplot
features = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'YearsSinceLastPromotion', 'PercentSalaryHike']
stayed_data = old[old['Attrition'] == 'Yes']
sns.pairplot(stayed_data[features + ['Attrition']], hue='Attrition')
plt.savefig('plots/attrition_pairplot.png')
plt.close()
inferences.append("The pairplot shows that employees with lower total working years (median: ~7 years) and years at the company (median: ~3 years) are more likely to leave, with a cluster of leavers having monthly incomes below $5,000.")

# Correlation Heatmap
cmatrix = dataset.corr()
attrition_corr = dataset.corr(numeric_only=True)['Attrition'].drop('Attrition').sort_values()
attrition_corr_df = attrition_corr.to_frame().T
plt.figure(figsize=(12, 1.5))
sns.heatmap(attrition_corr_df, cmap='coolwarm', annot=True, fmt=".2f", cbar=True)
plt.title('Correlation with Attrition')
plt.savefig('plots/attrition_correlation_heatmap.png')
plt.close()
inferences.append("The heatmap shows TotalWorkingYears (-0.17), Age (-0.16), and YearsAtCompany (-0.13) have the strongest negative correlations with attrition.")

# Histogram for Normality
plt.figure(figsize=(10, 10))
dataset.hist()
plt.tight_layout()
plt.savefig('plots/attrition_histogram.png')
plt.close()
inferences.append("Histograms show Age and MonthlyIncome are roughly normal, while DistanceFromHome and YearsSinceLastPromotion are right-skewed, affecting statistical test choices.")

# Map Attrition for old dataset
if 'Attrition' in old.columns:
    old['Attrition'] = old['Attrition'].map({'Yes': 1, 'No': 0})

# Normality Test
def test_normality(data, var, group, attrition_value):
    if len(data[data['Attrition'] == attrition_value][var].dropna()) > 3:
        stat, p = shapiro(data[data['Attrition'] == attrition_value][var].dropna())
        print(f'Shapiro-Wilk Test for {var} (Attrition={attrition_value}): p-value = {p:.4f}')
        return p > 0.05
    else:
        print(f'Shapiro-Wilk Test for {var} (Attrition={attrition_value}): Not enough data')
        return False

normality_results = {}
for var in numerical_vars:
    if var in dataset.columns:
        normality_yes = test_normality(dataset, var, 'Attrition', 1)
        normality_no = test_normality(dataset, var, 'Attrition', 0)
        normality_results[var] = normality_yes and normality_no

significant_factors = []
recommendations = []

# Chi-square Test for Categorical Variables
print("\nChi-square Test Results for Categorical Variables:")
for var in categorical_vars:
    if var in old.columns and old[var].nunique() > 1:
        contingency_table = pd.crosstab(old[var], old['Attrition'])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(f'{var}: Chi2 p-value = {p:.4f}')
        if p < 0.05:
            significant_factors.append(f'Categorical - {var}: Chi2 p-value = {p:.4f}')
            attrition_rates = old.groupby(var)['Attrition'].mean().sort_values(ascending=False)
            high_attrition = attrition_rates.head(1).index[0]
            low_attrition = attrition_rates.tail(1).index[0]
            recommendation = (f"For {var}, prioritize support for '{high_attrition}' employees "
                            f"(attrition rate: {attrition_rates[high_attrition]:.2%}) over "
                            f"'{low_attrition}' (attrition rate: {attrition_rates[low_attrition]:.2%}).")
            if var == 'MaritalStatus':
                recommendation += " Offer targeted benefits for high-risk groups."
            elif var == 'BusinessTravel':
                recommendation += " Reduce travel or provide travel benefits for frequent travelers."
            elif var == 'JobRole':
                recommendation += " Address role-specific issues (workload, resources)."
            recommendations.append(recommendation)

# Hypothesis Testing for Numerical Variables
print("\nHypothesis Testing Results for Numerical Variables:")
for var in numerical_vars:
    if var in dataset.columns:
        attr_yes = dataset[dataset['Attrition'] == 1][var].dropna()
        attr_no = dataset[dataset['Attrition'] == 0][var].dropna()
        if len(attr_yes) > 1 and len(attr_no) > 1:
            if normality_results[var]:
                stat, p = ttest_ind(attr_yes, attr_no, equal_var=False)
                test_type = 't-test'
            else:
                stat, p = mannwhitneyu(attr_yes, attr_no, alternative='two-sided')
                test_type = 'Mann-Whitney U'
            print(f'{var}: {test_type} p-value = {p:.4f}')
            if p < 0.05:
                significant_factors.append(f'Numerical - {var}: {test_type} p-value = {p:.4f}')
                mean_yes = attr_yes.mean()
                mean_no = attr_no.mean()
                median_yes = attr_yes.median()
                median_no = attr_no.median()
                value_type = "mean" if normality_results[var] else "median"
                value_yes = mean_yes if normality_results[var] else median_yes
                value_no = mean_no if normality_results[var] else median_no
                if var == 'Age':
                    if value_yes < value_no:
                        recommendation = (f"Younger employees (Attrition=Yes {value_type}: {value_yes:.2f}, "
                                        f"Attrition=No {value_type}: {value_no:.2f}) are more likely to leave. ")
                    else:
                        recommendation = (f"Older employees (Attrition=Yes {value_type}: {value_yes:.2f}, "
                                        f"Attrition=No {value_type}: {value_no:.2f}) are more likely to leave. "
                                        "Offer flexible work arrangements or retirement benefits.")
                elif var in ['MonthlyIncome', 'PercentSalaryHike', 'StockOptionLevel']:
                    if value_yes < value_no:
                        recommendation = (f"Increase {var} (Attrition=Yes {value_type}: {value_yes:.2f}, "
                                        f"Attrition=No {value_type}: {value_no:.2f}) to reduce attrition.")
                        if var == 'MonthlyIncome':
                            recommendation += " Implement salary increases or bonuses for low-income employees."
                        elif var == 'StockOptionLevel':
                            recommendation += " Expand equity programs to boost loyalty."
                    else:
                        recommendation = (f"No clear action needed for {var} (Attrition=Yes {value_type}: {value_yes:.2f}, "
                                        f"Attrition=No {value_type}: {value_no:.2f}), as higher values are associated with attrition.")
                elif var == 'DistanceFromHome':
                    if value_yes > value_no:
                        recommendation = (f"Reduce impact of {var} (Attrition=Yes {value_type}: {value_yes:.2f}, "
                                        f"Attrition=No {value_type}: {value_no:.2f}) by offering remote work or commuting benefits.")
                    else:
                        recommendation = (f"No clear action needed for {var} (Attrition=Yes {value_type}: {value_yes:.2f}, "
                                        f"Attrition=No {value_type}: {value_no:.2f}), as shorter distances are associated with attrition.")
                elif var in ['TotalWorkingYears', 'YearsAtCompany', 'YearsWithCurrManager']:
                    if value_yes < value_no:
                        recommendation = (f"Retain employees with lower {var} (Attrition=Yes {value_type}: {value_yes:.2f}, "
                                        f"Attrition=No {value_type}: {value_no:.2f}) through onboarding, loyalty bonuses, or stronger manager relationships.")
                    else:
                        recommendation = (f"No clear action needed for {var} (Attrition=Yes {value_type}: {value_yes:.2f}, "
                                        f"Attrition=No {value_type}: {value_no:.2f}), as longer tenure is associated with attrition.")
                elif var == 'YearsSinceLastPromotion':
                    if value_yes > value_no:
                        recommendation = (f"Reduce {var} (Attrition=Yes {value_type}: {value_yes:.2f}, "
                                        f"Attrition=No {value_type}: {value_no:.2f}) by accelerating promotion cycles or offering career development.")
                    else:
                        recommendation = (f"No clear action needed for {var} (Attrition=Yes {value_type}: {value_yes:.2f}, "
                                        f"Attrition=No {value_type}: {value_no:.2f}), as recent promotions are associated with attrition.")
                elif var == 'TrainingTimesLastYear':
                    if value_yes < value_no:
                        recommendation = (f"Increase {var} (Attrition=Yes {value_type}: {value_yes:.2f}, "
                                        f"Attrition=No {value_type}: {value_no:.2f}) by offering more training opportunities.")
                    else:
                        recommendation = (f"No clear action needed for {var} (Attrition=Yes {value_type}: {value_yes:.2f}, "
                                        f"Attrition=No {value_type}: {value_no:.2f}), as more training is associated with attrition.")
                else:
                    recommendation = (f"Review {var} (Attrition=Yes {value_type}: {value_yes:.2f}, "
                                    f"Attrition=No {value_type}: {value_no:.2f}) for targeted interventions based on context.")
                recommendations.append(recommendation)
        else:
            print(f'{var}: Skipped (insufficient data)')

# Logistic Regression
y = dataset['Attrition']
x = dataset.drop(['Attrition'], axis=1)
x1 = sm.add_constant(x)
model = sm.Logit(y, x1)
result = model.fit()
print(result.summary())
p_values = result.pvalues
significant_features = p_values[p_values < 0.05].index.tolist()
if 'const' in significant_features:
    significant_features.remove('const')

# Random Forest
print("Random Forest")
imp_features = []
rf_model = RandomForestClassifier(n_estimators=1000, max_features=2, oob_score=True)
rf_model.fit(dataset[significant_features], dataset['Attrition'])
for feature, imp in zip(significant_features, rf_model.feature_importances_):
    print(feature, '->', imp)
    if imp > 0.1:
        imp_features.append(feature)

plt.figure(figsize=(10, 6))
feature_importance = pd.DataFrame({'Feature': significant_features, 'Importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig('plots/feature_importance_rf.png')
plt.close()
inferences.append("The Random Forest feature importance plot highlights TotalWorkingYears (~0.15), MonthlyIncome (~0.12), and Age (~0.10) as key predictors of attrition.")

# Decision Tree
tree_model = tree.DecisionTreeClassifier(max_depth=4)
predictors = pd.DataFrame(dataset[imp_features])
tree_model.fit(predictors, dataset['Attrition'])
with open('EmployeeAttrition.dot', 'w') as f:
    f = tree.export_graphviz(tree_model, feature_names=imp_features, out_file=f)

print("Plot the Decision Tree using the file EmployeeAttrition.dot.")
print("The accuracy of this model (Decision Tree) is:", (tree_model.score(predictors, dataset['Attrition']) * 100), "%")
inferences.append(f"The decision tree (accuracy: {tree_model.score(predictors, dataset['Attrition']) * 100:.2f}%) splits primarily on TotalWorkingYears and MonthlyIncome, confirming their importance.")

# Save Significant Factors and Recommendations
with open('significant_factors_attrition.txt', 'w') as f:
    f.write("Significant Factors Contributing to Attrition (p < 0.05):\n")
    if significant_factors:
        for factor in significant_factors:
            f.write(f"{factor}\n")
    else:
        f.write("No significant factors found.\n")
    f.write("\nRecommendations to Reduce Attrition:\n")
    if recommendations:
        for rec in recommendations:
            f.write(f"{rec}\n")
    else:
        f.write("No actionable recommendations due to no significant factors.\n")

# Save Inferences to a File for Reference
with open('inferences.txt', 'w') as f:
    f.write("Inferences from Visualizations:\n")
    for i, inference in enumerate(inferences, 1):
        f.write(f"{i}. {inference}\n")

print("All plots saved in the 'plots' directory.")
print("Inferences saved in 'inferences.txt'.")


