import pandas as pd
import random
import pickle
import sklearn as sk
from sklearn.cluster import KMeans
import datetime
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import matplotlib as mpl
import mpl_toolkits.mplot3d as p3d
import numpy as np
import pickle


print(random.uniform(0, 1))
df = pd.read_csv('archive/listings.csv')
review = pd.read_csv('archive/reviews.csv')
#print(review)
df['price'] = df['price'].str.replace("\$|,", "").astype(float)
df['cleaning_fee'] = df['cleaning_fee'].str.replace("\$|,", "").astype(float)
df['extra_people'] = df['extra_people'].str.replace("\$|,", "").astype(float)
df['cleaning_fee'] = df['cleaning_fee'].fillna(0)



df['price'] = df['price']/df['accommodates'] + df['cleaning_fee']/df['accommodates']


print(len(df))
len(df['last_scraped'].unique())
#df = df.groupby('neighbourhood_cleansed')
#for d in df:
#    print(d)
# s = pd.Series(np.random.randn())
# a = pd.DataFrame(np.random.randn(0,1), columns=list('Occupancy Rate'))
# df['Occupancy Rate'] = random.uniform(0, 1)
#df.head()
#df['property_type'].unique()

selected_features = [u'id' ,u'price',u'accommodates',u'host_response_time',
       u'bathrooms', u'bedrooms', u'beds',u'security_deposit', u'cleaning_fee', u'guests_included',
       u'extra_people', u'minimum_nights', u'maximum_nights',
       u'availability_365', u'latitude', u'longitude',
       u'number_of_reviews', u'review_scores_rating',u'review_scores_cleanliness', u'review_scores_checkin',
       u'review_scores_communication', u'review_scores_location',
       u'review_scores_value', u'amenities','bed_type', 'room_type', 'cancellation_policy', 'property_type']
df = df.loc[:, selected_features]
df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))

plt.hist(list(df["price"]),bins=100)

plt.show()



la = list(df['latitude'])
lo = list(df['longitude'])

X = [[i,j] for i,j in zip(lo,la)]

n_clusters = 10
y_pred = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X)

#ploting

fig, ax1 = plt.subplots(1)
ax1.scatter(lo, la
        ,marker='o' #点的形状
        ,s=2 #点的大小
        ,c="#F58574"#y_pred
       )
plt.show()

df['cluster'] = y_pred
review_count = review.groupby("listing_id")['id'].agg('count')
df['review_scores_rating'] = df['review_scores_rating'].fillna(0)
accom_count = df.groupby("cluster").agg('sum')
df = pd.merge(df, review_count, left_on='id', right_index=True, how='left')
df['id_y'] = df['id_y'].fillna(0)

#print(review_count)

region_hotel = [[] for i in range(n_clusters)]
region_price = [[] for i in range(n_clusters)]
region_score = [[] for i in range(n_clusters)]
region_score_num = [[] for i in range(n_clusters)]
region_accommodate = [[] for i in range(n_clusters)]

guest_num = [[] for i in range(n_clusters)] # 365 days for each region
guest_mean_price = []
guest_std_price = []

review['date'] = review['date'].str.replace("-","").astype(int)%10000
#intdate = review['date'].str.replace("-","").astype(int)
#date2015 = intdate.loc[intdate%10000 == 2015]%10000

anual_review = review.groupby("date").agg("count")['id']

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.plot([i for i in range(len(anual_review.values))],list(anual_review.values))
plt.xlabel("Time / day")
plt.ylabel("Comments Number")
plt.title("Comments-Time Curve")
plt.tick_params(labelsize=10)
plt.show()

df = df.groupby("cluster")
clus_ind = 0
for d in df:
    #d = pd.merge(d[1], review_count, left_on='id', right_index=True, how='left')
    #d['id_y'] = d['id_y'].fillna(0)

    review_d = pd.merge(review, d[1], left_on='listing_id', right_on='id_x', how='right')
    review_num = review_d.groupby('date').agg('count')

    guest_num[clus_ind] = list(review_num['id'])
    guest_mean_price.append(review_d['price'].mean())
    guest_std_price.append(review_d['price'].std())

    region_hotel[clus_ind] = list(d[1]['id_x'])
    region_price[clus_ind] = list(d[1]['price'])
    region_accommodate[clus_ind] = list(d[1]['accommodates'])
    region_score[clus_ind] = list(d[1]['review_scores_rating'])
    region_score_num[clus_ind] = list(d[1]['number_of_reviews'])

    clus_ind += 1
    #print(d['review_scores_rating'])
    #plt.hist(list(anual_review), bins=100)
    #plt.show()
    #print(d)

dataset = [region_hotel, region_price, region_score, region_accommodate,
           guest_num, guest_mean_price, guest_std_price]

fw = open('dataFile.txt','wb')
pickle.dump(dataset,fw,-1)
fw.close()


time = review.groupby("date").agg('count')
pd.to_datetime(time)
print(list(time.index))
plt.bar(list(time.index), list(time['id']))
plt.show()






