import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Data Cleaning
df = df.drop_duplicates()
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Visualization 1: Survival Count
sns.countplot(x='Survived', data=df)
plt.title("Passenger Survival Count")
plt.savefig("survival_count.png")
plt.close()

# Visualization 2: Gender vs Survival
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Gender vs Survival")
plt.savefig("gender_survival.png")
plt.close()

# Visualization 3: Passenger Class vs Survival
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Passenger Class vs Survival")
plt.savefig("class_survival.png")
plt.close()

print("Project completed successfully.")
