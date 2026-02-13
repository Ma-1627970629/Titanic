#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


# Data Reading
train_df = pd.read_csv('i239e_project_train.csv')
test_df = pd.read_csv('i239e_project_test.csv')


# In[6]:


# Count the number of missing values in each column
missing_values = train_df.isnull().sum()

missing_percentage = (train_df.isnull().sum() / len(train_df)) * 100

missing_data = pd.concat([missing_values, missing_percentage], axis=1, keys=['Number of Percentage(%)])

print("Training Set Missing Value Statistics：")
print(missing_data[missing_data['Number of Missing Values'] > 0]) 


# In[7]:


# combine test and train as single to apply some function
all_data=[train_df,test_df] 


# In[8]:


# Delete Feature #9 with the most missing values
train_df.drop(columns=['Feature#9'], inplace=True)
test_df.drop(columns=['Feature#9'], inplace=True)


# In[10]:


target_features = ['Feature#1', 'Feature#2', 'Feature#3', 'Feature#4', 'Feature#5', 'Feature#6', 'Feature#7', 'Feature#8', 'Feature#10']

def analyze_feature_type(df, features):
    analysis_results = []
    
    for col in features:
        unique_count = df[col].nunique()
        data_type = df[col].dtype
        
        # If the number of unique values is small, then regarded as discrete or categorical.
        if unique_count < 10:
            feat_type = "Discrete/Categorical"
        else:
            feat_type = "Continuous/Numeric"
            
        analysis_results.append({
            "Characteristics": col,
            "Number of unique values": unique_count,
            "Data Type": data_type,
            "Preliminary determination": feat_type
        })
    
    return pd.DataFrame(analysis_results)

print(analyze_feature_type(train_df, target_features))

# Drawing Analysis
plt.figure(figsize=(16, 12))

for i, col in enumerate(target_features, 1):
    plt.subplot(3, 3, i)
    
    # Determine the plotting method based on the number of unique values
    if train_df[col].nunique() < 15:
        # Discrete type: Use a count plot and observe its relationship with survival rates
        sns.countplot(data=train_df, x=col, hue='Survived', palette='viridis')
        plt.title(f'{col} (Discrete) vs Survived')
    else:
        # Continuous type: Use histogram/density plot
        sns.histplot(train_df[col].dropna(), kde=True, color='skyblue')
        plt.title(f'{col} (Continuous) Distribution')

plt.tight_layout()
plt.show()


# In[11]:


import re

# Define title extraction function: Use regular expressions to separate titles from name strings
def get_title(name):
    if isinstance(name, str):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        if title_search:
            return title_search.group(1)
    return ""

# Title column added.
for dataset in all_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
    
    # Classified Rare Titles
    rare_titles = ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    dataset['Title'] = dataset['Title'].replace(rare_titles, 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# Mapping Based on Actual Survival Rates
# Calculate the mean survival rate of the training set
title_means = train_df.groupby('Title')['Survived'].mean().sort_values()
print("Survival Rate by Identity Reference：\n", title_means)

# Definition Mapping (Based on Survival Rate from Lowest to Highest: Mr < Rare < Master < Miss < Mrs)
title_mapping = {"Mr": 1, "Rare": 2, "Master": 3, "Miss": 4, "Mrs": 5}

# Iterate again, applying the numerical mapping.
for dataset in all_data:
    dataset['Title_Numeric'] = dataset['Title'].map(title_mapping).fillna(1)


# In[12]:


# Calculate the median age for each title in the training set.
# Title -> Median Age
title_age_map = train_df.groupby('Title')['Feature#4'].median()

for dataset in all_data:
    # Use the map function to precisely populate
    dataset['Feature#4'] = dataset['Feature#4'].fillna(dataset['Title'].map(title_age_map))
    
    # Fallback Strategy: If a Title is not present in the training set, use the global median.
    dataset['Feature#4'] = dataset['Feature#4'].fillna(train_df['Feature#4'].median())


# In[13]:


for dataset in all_data:
    # Create new feature: Merge Feature#5 and Feature#6
    dataset['Feature#56'] = dataset['Feature#5'] + dataset['Feature#6']


# In[19]:


heatmap_features = [
    'Survived', 'Feature#2', 'Feature#3', 'Feature#56',
    'Feature#10', 'Title_Numeric', 
    'Feature#4', 'Feature#8'
]

plt.figure(figsize=(12, 10))
# Ensure data is numeric
corr_matrix = train_df[heatmap_features].apply(pd.to_numeric).corr()

sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt=".2f")
plt.title("Fixed Correlation Heatmap (Addressing Bin Edge Issue)")
plt.show()


# In[20]:


# Define the 5 high-value attribute columns being selected.
cols = ["Feature#2", "Feature#3", "Title_Numeric", "Feature#8", "Feature#10"]

# Extract the training set from all_data[0] (i.e., train_df)
x_train = train_df[cols].values
y_train = train_df['Survived'].values

x_test = test_df[cols].values


# In[21]:


X_train = torch.tensor(x_train, dtype=torch.float32)
Y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(x_test, dtype=torch.float32)


# In[32]:


# Redefine the empty shell class
class PretrainedNetwork(nn.Module):
    def __init__(self):
        super(PretrainedNetwork, self).__init__()

    def forward(self, x):
        # Iterate through all sublayers in the model and execute them sequentially.
        for module in self.children():
            x = module(x)
        return x

# load the pre-trained model
frozen_model = torch.load("pretrained_network.pth", weights_only=False)


# In[36]:


import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split

# Define an enhanced residual model 
class ResidualTitanicModel(nn.Module):
    def __init__(self, frozen_model):
        super(ResidualTitanicModel, self).__init__()
        self.frozen = frozen_model
        
        # Freeze the pre-training component
        for param in self.frozen.parameters():
            param.requires_grad = False
            
        # Classifiers for Small Sample Designs (Input 8 = 3 (frozen output) + 5 (original input X_train dimension))
        self.refiner = nn.Sequential(
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),       
            
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        with torch.no_grad():
            feat_frozen = self.frozen(x)
        # Concatenate pre-trained features with original X_train features
        combined = torch.cat((feat_frozen, x), dim=1)
        return self.refiner(combined)

# Split the dataset 
dataset = TensorDataset(X_train, Y_train.view(-1, 1).float())
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

# Initialization
model = ResidualTitanicModel(frozen_model)
criterion = nn.BCELoss()
# Using L2 regularization (weight_decay)
optimizer = optim.Adam(model.refiner.parameters(), lr=0.005, weight_decay=1e-2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)


epochs = 100
best_val_acc = 0.0          
early_stop_counter = 0      
early_stop_patience = 20

for epoch in range(epochs):
    # Training Part
    model.train()
    total_train_loss = 0
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    # Evalution Part
    model.eval()
    val_correct = 0
    val_total = 0
    total_val_loss = 0
    with torch.no_grad():
        for v_x, v_y in val_loader:
            v_outputs = model(v_x)
            v_preds = (v_outputs > 0.5).float()
            val_correct += (v_preds == v_y).sum().item()
            val_total += v_y.size(0)

    val_acc = val_correct / val_total
    avg_train_loss = total_train_loss / len(train_loader)
    
    # Dynamically adjust the learning rate
    scheduler.step(avg_train_loss)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.refiner.state_dict(), 'best_refiner_weights.pth')
        early_stop_counter = 0
        print(f"Epoch {epoch+1}: Detected better Acc: {val_acc:.2%}, The model has been saved.")
    else:
        # If accuracy does not improve, increment the counter by 1.
        early_stop_counter += 1
        
    # Print Processing
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Currently Acc: {val_acc:.2%} (All-time high: {best_val_acc:.2%})")

    # Early Stop Trigger
    if early_stop_counter >= early_stop_patience:
        print(f"Stop training: Accuracy has been consistently {early_stop_patience} iters has not improved, indicating that a bottleneck has been reached.")
        break

print(f"Training has finally concluded! Highest validation set accuracy: {best_val_acc:.2%}")


# In[37]:


# Reload the model and apply the optimal weights.
inference_model = ResidualTitanicModel(frozen_model)
inference_model.refiner.load_state_dict(torch.load('best_refiner_weights.pth'))
inference_model.eval()  # Switch to evaluation mode

# Conduct reasoning
with torch.no_grad():
    # Obtain the model output (probability after Sigmoid)
    test_outputs = inference_model(X_test)
    # Convert the probability to 0 or 1 
    test_preds = (test_outputs > 0.5).int().numpy()

submission_df = pd.read_csv('i239e_project_test.csv')
submission_df['Survived'] = test_preds.flatten()
submission_df.to_csv('outcome.csv', index=False)

