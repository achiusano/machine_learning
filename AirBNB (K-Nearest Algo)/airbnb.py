import pandas as pd
import numpy as np

dc_listings = pd.read_csv(r'listings.csv')
#print(dc_listings)

point1 = 3
point2 = dc_listings['accommodates'].iloc[0]
first_distance = np.abs(point1 - point2)

#print(first_distance)
new_listing = 3
dc_listings['distance'] = dc_listings['accommodates'].apply(lambda x: np.abs(x - new_listing))
print(dc_listings['distance'].value_counts())
np.random.seed(1)
dc_listings = dc_listings.loc[np.random.permutation(len(dc_listings))]
dc_listings = dc_listings.sort_values('distance')
#print(dc_listings.iloc[0:10]['price'])

stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype(float)
mean_price = dc_listings['price'].iloc[0:5].mean()
#print(mean_price)
def predict_price(new_listing):
    temp_df = dc_listings.copy()
    temp_df['distance'] = temp_df['accommodates'].apply(lambda x: np.abs(x - new_listing))
    temp_df = temp_df.sort_values('distance')
    nearest_neighbors = temp_df.iloc[0:5]['price']
    predicted_price = nearest_neighbors.mean()
    return(predicted_price)
  
acc_one = predict_price(1)
acc_four = predict_price(4)
print(acc_one)
print(acc_four)
