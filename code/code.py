import libraries as lb



#load and check data
df = lb.pd.read_csv("C:/Users/Alper/Documents/GitHub/LogisticRegressionWithBreastCancerDataset/dataset/breast-cancer.csv")

#info
print(df.info())
# change diagnosis contents (M = 1 , B = 0) , and delete id column
df.diagnosis = [1  if each == "M" else 0 for each in df.diagnosis]
print(df.diagnosis)

