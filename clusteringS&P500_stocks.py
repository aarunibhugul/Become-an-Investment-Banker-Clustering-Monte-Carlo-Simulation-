from matplotlib.pyplot import plt
import pandas as pd
import pandas_datareader as data
from math import sqrt
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np



#------------------Collecting the list of Companies in S&P500 Index------------------------------------#

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
#after Visualising I have discovered that there is an outlier across our dataset. In case of stock data, outliers can be interesting
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
###------------------Portfolio Optmisation-------------###
# now that we have created the segmented (clustered) stocks on the
# basis of their annual risk and return. Its the time to create a portfolio that gives the
# optmised combination of number share of companies that should be present in a portfolioself.

# you can select any companies based from all three/four clusters.For this example
# I am selecting companies with maximum annual return from each clusters

#-------Lets get the company with maximum annual return and risk from all the clusters---###
portfolio_dataFrame = return_and_volatility_datfarame.groupby("Cluster Name").idxmax()
portfolio_list = list(portfolio_dataFrame["Annual Returns"]) #to the list of companies with max return

##-------creating covariance matrix----------##

# from my previous project where I used to work for financial client
# I remember very well that I had designed an alogorithm to calculate the portfolio based
# on annual return which had led to a huge mess and riot followed by escalation ;)
# now I remeber very well that we need to calculate just the mean daily
# optimised_portfolio_dataFrame = stock_prices[portfolio_list]
daily_return = stock_prices[portfolio_list].pct_change()
daily_mean_returns = daily_return.mean() # this will generate a series with three rows
covariance_matrix = daily_return.cov() # this will generate a 3X3 matrix


##-----------setting up portfolio weights-----#####
portfolios_weight = 25000

#set up array to hold results
#We have increased the size of the array to hold the weight values for each stock
results = np.zeros((4+len(portfolio_list)-1,portfolios_weight))

###------------Monte Carlo Simulation----------####
for i in range(portfolios_weight):
    #select random weights for portfolio holdings
    weights = np.array(np.random.random(len(portfolio_list)))
    #rebalance weights to sum to 1
    weights /= np.sum(weights)
    #calculate portfolio return and volatility
    portfolio_return = np.sum(daily_mean_returns * weights) * 252
    portfolio_risk = np.sqrt(np.dot(weights.T,np.dot(covariance_matrix, weights))) * np.sqrt(252)

    #store results in results array
    results[0,i] = portfolio_return
    results[1,i] = portfolio_risk
    #store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
    results[2,i] = results[0,i] / results[1,i]
    #iterate through the weight vector and add data to results array
    for j in range(len(weights)):
        results[j+3,i] = weights[j]

#convert results array to Pandas DataFrame
results_frame = pd.DataFrame(results.T)#columns=['ret','stdev','sharpe',stocks[0],stocks[1],stocks[2],stocks[3]])
results_frame.columns = "Return Risk SharpeRatio".split() + portfolio_list

results_frame.to_csv("optimized_portfolio.csv")
#locate position of portfolio with highest and lowest Sharpe Ratio
highest_SharpeRatio = results_frame.iloc[results_frame['SharpeRatio'].idxmax()]
lowest_SharpeRatio = results_frame.iloc[results_frame['Risk'].idxmin()]

print("The latest best investment with highest risk and return in S&P 500 Companies is :\n ",highest_SharpeRatio)
print("The latest best investment with lowest risk and return in S&P 500 Companies is :\n ",lowest_SharpeRatio)
