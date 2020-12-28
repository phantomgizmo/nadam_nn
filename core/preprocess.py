import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess(dataset):
	query = 'Glucose > 0 & BloodPressure > 0 & SkinThickness > 0 & Insulin > 0 & BMI > 0 & DiabetesPedigreeFunction > 0 & Age > 0'
	result = dataset.query(query)

	scaler = MinMaxScaler()
	scaler.fit(result)
	result = scaler.transform(result)

	x = []
	y = []

	for data in result:
		x.append(data[:-1])
		y.append(data[-1:])

	return (x, y)

