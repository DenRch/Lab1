import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("D:/ИИиМО/pandemia/dataset/Historical_Pandemic_Epidemic_Dataset.csv")

#print(df.head())
print(df.isnull().sum())
print(df.dtypes)

mode_value = df['Containment_Method'].mode()[0]
df['Containment_Method'].fillna(mode_value, inplace = True)
#print(df.isnull().sum())

scaler = MinMaxScaler()
df['Estimated_Cases'] = scaler.fit_transform(df[['Estimated_Cases']])
#print(df[['Estimated_Cases']].head(20))

df = pd.get_dummies(df, columns = ['Pathogen_Type'], drop_first = True)
print(df.head(10))

df.to_csv("HPED.csv", index = False)
