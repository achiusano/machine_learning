import pandas as pd
import numpy as np
np.random.seed(1)

dc_listings = pd.read_csv(r'listings.csv')
dc_listings = dc_listings.loc[np.random.permutation(len(dc_listings))]
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')

drop_columns = ['room_type', 'city', 'state', 'latitude', 'longitude', 'zipcode', 'host_response_rate', 'host_acceptance_rate', 'host_listings_count']
dc_listings = dc_listings.drop(drop_columns, axis = 1)
#print(dc_listings.isnull().sum())

dc_listings = dc_listings.drop(['cleaning_fee', 'security_deposit'], axis = 1)
dc_listings = dc_listings.dropna(axis = 0)
#print(dc_listings.isnull().sum())

normalized_listings = (dc_listings - dc_listings.mean()) / (dc_listings.std())
normalized_listings['price'] = dc_listings['price']
#print(normalized_listings.head(3))

from scipy.spatial import distance

first_listing = normalized_listings.iloc[0][['accommodates', 'bathrooms']]
fifth_listing = normalized_listings.iloc[4][['accommodates', 'bathrooms']]
first_fifth_distance = distance.euclidean(first_listing, fifth_listing)
#print(first_fifth_distance)

from sklearn.neighbors import KNeighborsRegressor

train_df = normalized_listings.iloc[0:2792]
test_df = normalized_listings.iloc[2792:]
training_columns = ['accommodates', 'bathrooms']

knn = KNeighborsRegressor(n_neighbors = 5, algorithm = 'brute')
knn.fit(train_df[training_columns], train_df['price'])
predictions = knn.predict(test_df[['accommodates', 'bathrooms']])

from sklearn.metrics import mean_squared_error

train_columns = ['accommodates', 'bathrooms']
knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute', metric='euclidean')
knn.fit(train_df[train_columns], train_df['price'])
predictions = knn.predict(test_df[train_columns])

two_features_mse = mean_squared_error(test_df['price'], predictions)
two_features_rmse = np.sqrt(two_features_mse)
#print(two_features_mse, two_features_rmse)

features = ['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']

knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute')

knn.fit(train_df[features], train_df['price'])

four_predictions = knn.predict(test_df[features])

four_mse = mean_squared_error(test_df['price'], four_predictions)
four_rmse = np.sqrt(four_mse)

#print(four_mse, four_rmse)

features = train_df.columns.tolist()
features.remove('price')

knn.fit(train_df[features], train_df['price'])
all_features_predictions = knn.predict(test_df[features])

all_features_mse = mean_squared_error(test_df['price'], all_features_predictions)
all_features_rmse = np.sqrt(all_features_mse)
print(all_features_mse, all_features_rmse)
