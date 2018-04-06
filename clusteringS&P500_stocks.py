from matplotlib.pyplot import plt
import pandas as pd
import pandas_datareader as data
from math import sqrt
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np



#------------------Collecting the list of Compqnies in Dow&Jones Index------------------------------------#

sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

#------------------reading the data from website by scraping the company symbol data------------------------#
SP500_Dataframe = pd.read_html(sp500_url)


tickers = SP500_Dataframe[0][1:][0].tolist()
###----------------- Fetching the data from Yahoo Finance------------------------####

stock_prices = pd.DataFrame()
for t in tickers:
    try:
        stock_prices[t] = data.DataReader(t,'yahoo','01/01/2017')['Adj Close']
    except:
        pass
stock_prices.to_csv("S&P500-06-April-2017.csv")
#---------now the data is collected from DataFrame---------------------#
stock_prices = pd.read_csv("S&P500-06-April-2017.csv") #I prefer saving the data from Yahoo to csv
#to make further processing faster. The above code (except the import) can be commented.
stock_prices = stock_prices.set_index('Date')

return_and_volatility_datfarame = pd.DataFrame()
return_and_volatility_datfarame["Annual Returns"] = (stock_prices.pct_change().mean() * 252)*100
return_and_volatility_datfarame["Annual Risk"] = (stock_prices.pct_change().std() * sqrt(252))*100
return_and_volatility_datfarame.index.name = "Company Symbol"

#-----------Elbow Method to get the optimal number of cluster-----#

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(return_and_volatility_datfarame)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# After plotting the result of "Number of clusters" vs "WCSS" we can notice that
#the number of clusters reaches 4 (on the X axis), the reduction
# the within-cluster sums of squares (WCSS) begins to slow down for each increase in cluster number. Hence
# the optimal number of clusters for this data comes out to be 4. Therefore lets take number of cluster for k means = 4
#--------------------applying K-Means Clustering-------------#

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(return_and_volatility_datfarame)

return_and_volatility_datfarame.reset_index(level=['Company Symbol'], inplace =True)
return_and_volatility_datfarame["Cluster Name"] = y_kmeans



####----------------------Visualising the S&P 500-----------####
numeric_values = return_and_volatility_datfarame.iloc[:,[1,2]]

plt.scatter(numeric_values[y_kmeans == 0, 0], numeric_values[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(numeric_values[y_kmeans == 1, 0], numeric_values[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(numeric_values[y_kmeans == 2, 0], numeric_values[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(numeric_values[y_kmeans == 3, 0], numeric_values[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of Copmanies')
plt.xlabel('Returns (%)')
plt.ylabel('Risk (%)')
plt.legend()
plt.show()

##-------------------Renaming the Clusters-----------##

return_and_volatility_datfarame["Cluster Name"] = return_and_volatility_datfarame["Cluster Name"].replace(0,"Low")
return_and_volatility_datfarame["Cluster Name"] = return_and_volatility_datfarame["Cluster Name"].replace(1,"Highest")
return_and_volatility_datfarame["Cluster Name"] = return_and_volatility_datfarame["Cluster Name"].replace(2,"High")
return_and_volatility_datfarame["Cluster Name"] = return_and_volatility_datfarame["Cluster Name"].replace(3,"Medium")

#-----Getting the data on local----#
#I prefer taking data in csv or excel so that I can Visualise it in tableau#
return_and_volatility_datfarame.to_csv("Clustered_S&P500.csv")



##------------Identifying and ommiting the outlier---------#
#after Visualising I have discovered that there a outlier. In case of stock outliers can be interesting
#for the investors who would like to take higher risk in exchange of higher return.
return_and_volatility_datfarame = return_and_volatility_datfarame.set_index('Company Symbol')
returns_outlier = return_and_volatility_datfarame["Annual Returns"].idxmax()
risk_outlier = return_and_volatility_datfarame["Annual Risk"].idxmax()
# print(returns_outlier)
# print(risk_outlier)

return_and_volatility_datfarame.drop(risk_outlier,inplace=True)

# list_of_companies = list_of_companies.remove("Security")
return_and_volatility_datfarame.sort_index(inplace=True)

return_and_volatility_datfarame.to_csv("Outlier_Free_Clustered_S&P500.csv")
