#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 13:00:55 2017

@author: Kelsey Odenthal

Assistance in assembling this code was achieved with Quandl documentation websites,
and pythonprogramming.net for help with webscraping and data compiling.
"""
from __future__ import division


import quandl
#auth_tok = quandl.ApiConfig.api_key = "enter token here"
auth_tok = quandl.ApiConfig.api_key = "dB3psxSH5YBDYsqsvdjf"




def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import pandas as pd
import quandl.errors.quandl_error
import numpy as np

#api_key = auth_tok
from sklearn import svm, preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import os
import time
from time import mktime
from datetime import datetime
import re
from sklearn.exceptions import NotFittedError
import urllib2
import matplotlib.pyplot as plt
import scikitplot as skplt
from collections import Counter
start=datetime.now()
import enaml
from enaml.qt.qt_application import QtApplication
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def Input():
    x = 0
    ticker_list = []


    while x!= 3:
        ticker = raw_input('Please enter a ticker symbol: ')
        x = x + 1
        ticker_list.append(ticker)
        print "ticker list"
        print 'You chose: ' + ticker

    return ticker_list


improvement = 5


FEATURES = ['DE Ratio',
             'Trailing P/E',
             'Price/Sales',
             'Price/Book',
             'Profit Margin',
             'Operating Margin',
             'Return on Assets',
             'Return on Equity',
             'Revenue Per Share',
             'Market Cap',
             'Enterprise Value',
             'Forward P/E',
             'PEG Ratio',
             'Enterprise Value/Revenue',
             'Enterprise Value/EBITDA',
             'Revenue',
             'Gross Profit',
             'EBITDA',
             'Net Income Avl to Common ',
             'Diluted EPS',
             'Earnings Growth',
             'Revenue Growth',
             'Total Cash',
             'Total Cash Per Share',
             'Total Debt',
             'Current Ratio',
             'Book Value Per Share',
             'Cash Flow',
             'Beta',
             'Held by Insiders',
             'Held by Institutions',
             'Shares Short (as of',
             'Short Ratio',
             'Short % of Float',
             'Shares Short (prior ']
                            

def Status(stocks, sp500):
    diff = stocks - sp500
    
    if diff > improvement:
        return 1
    else:
        return 0


""" Create a method that accepts a data range (start_date, end_date), array(?) of ticker 
    symbols build model of stock behavior"""
def ticker_input():
    """Array of ticker symbols are requested and iterated through"""
  

    ticker_list = Input()

    start_date = raw_input('Please input start date (YYYY-MM-DD): ')
    end_date = raw_input('Please input end date (YYYY-MM-DD): ')

    dataset1 = None
    new_dataframe = pd.DataFrame()
    x = 0


    for ticker in ticker_list:
        try:

            dataset = quandl.get('WIKI/' + ticker.upper(), 
                                      start_date = str(start_date),
                                      end_date = str(end_date), 
                                      api_key = auth_tok)


            print ticker
            print dataset


            dataset.to_csv("ticker_input.csv")

            if dataset.empty:

                continue
            else:
                continue
        except quandl.errors.quandl_error.NotFoundError:

            start_date = raw_input('Try again. Please input start date (YYYY-MM-DD): ')            
            end_date = raw_input('Try again. Please input end date (YYYY-MM-DD): ')            
            continue
        except SyntaxError:
            continue
#        print "dataframe concat"
#        print ticker
#        dataframe = pd.DataFrame(dataset1)
#        new_dataframe = pd.concat(dataset1)
#        dataframe = pd.concat([dataframe, dataset1[ticker.upper()]], axis=1)

                

#    dataset1.to_csv("ticker_input.csv")

#    print dataframe
    return dataset
    
#    data = pd.DataFrame.from_csv("ticker_input.csv")
#    print data
    

    
stock_path = os.path.join('intraQuarter')



def Yahoo():
    stock_key_path  = stock_path + os.path.join('/_KeyStats')
    stock_key = [x[0] for x in os.walk(stock_key_path)]
                 
    for e in stock_key[1:50]:
        try:
            print "Loading..."
            e = e.replace("intraQuarter/_KeyStats/","")
            url = "https://finance.yahoo.com/q/ks?s="+e.upper()+"+Key+Statistics"
            response = urllib2.urlopen(url).read()
#            print str(e)
            save = "forward/"+str(e)+".html"
#            print "Saved"
#            print os.path.exists(save)
            dirname = os.path.dirname(save)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            with open(save, 'w'):
                storage = open(save,"w")
#                print "open saved storage"
                storage.write(str(response))
                storage.close()
            
        except Exception as e:
            print(str(e))
            time.sleep(2)
            
Yahoo()

def Forward_HTML(gather=["Total Debt/Equity",
                         'Trailing P/E',
                         'Price/Sales',
                         'Price/Book',
                         'Profit Margin',
                         'Operating Margin',
                         'Return on Assets',
                         'Return on Equity',
                         'Revenue Per Share',
                         'Market Cap',
                         'Enterprise Value',
                         'Forward P/E',
                         'PEG Ratio',
                         'Enterprise Value/Revenue',
                         'Enterprise Value/EBITDA',
                         'Revenue',
                         'Gross Profit',
                         'EBITDA',
                         'Net Income Avl to Common ',
                         'Diluted EPS',
                         'Earnings Growth',
                         'Revenue Growth',
                         'Total Cash',
                         'Total Cash Per Share',
                         'Total Debt',
                         'Current Ratio',
                         'Book Value Per Share',
                         'Cash Flow',
                         'Beta',
                         'Held by Insiders',
                         'Held by Institutions',
                         'Shares Short (as of',
                         'Short Ratio',
                         'Short % of Float',
                         'Shares Short (prior ']):
                             
#    print "def forward html"                         
                             
    dataframe = pd.DataFrame(columns = ['Date',
                                        'Unix',
                                        'Ticker',
                                        'Price',
                                        'stock_p_change',
                                        'SP500',
                                        'sp500_p_change',
                                        'Difference',
                                        ##############
                                        'DE Ratio',
                                        'Trailing P/E',
                                        'Price/Sales',
                                        'Price/Book',
                                        'Profit Margin',
                                        'Operating Margin',
                                        'Return on Assets',
                                        'Return on Equity',
                                        'Revenue Per Share',
                                        'Market Cap',
                                        'Enterprise Value',
                                        'Forward P/E',
                                        'PEG Ratio',
                                        'Enterprise Value/Revenue',
                                        'Enterprise Value/EBITDA',
                                        'Revenue',
                                        'Gross Profit',
                                        'EBITDA',
                                        'Net Income Avl to Common ',
                                        'Diluted EPS',
                                        'Earnings Growth',
                                        'Revenue Growth',
                                        'Total Cash',
                                        'Total Cash Per Share',
                                        'Total Debt',
                                        'Current Ratio',
                                        'Book Value Per Share',
                                        'Cash Flow',
                                        'Beta',
                                        'Held by Insiders',
                                        'Held by Institutions',
                                        'Shares Short (as of',
                                        'Short Ratio',
                                        'Short % of Float',
                                        'Shares Short (prior ',                                
                                        ##############
                                        'Status'])
                                        

    filelist = os.listdir("forward")
    

    for file in filelist:

        tick = file.split(".html")[0]

        fullfilepath = "forward/" + file
        source = open(fullfilepath, "r").read()

        
        try:
            valuearray = []

            for data in gather:
                try:

                    
                    regex = re.escape(data) + r'.*?(\d{1,8}\.\d{1,8}M?B?|N/A)%?'

                    value = re.search(regex, source)
                    
                    value = (value.group(1))

                    if "B" in value:
                        value = float(value.replace("B",''))*1000000000
                    elif "M" in value:
                        value = float(value.replace("M",''))*1000000

                    valuearray.append(value)

                        
                        
                except Exception as e:
                    value = "N/A"
                    valuearray.append(value)
#            print valuearray
                    
            if valuearray.count("N/A") > 15:
                pass
            else:
                
                dataframe = dataframe.append({'Date':"N/A",
                                              'Unix':"N/A",
                                              'Ticker':tick,
                                              'Price':"N/A",
                                              'stock_p_change':"N/A",
                                              'SP500':"N/A",
                                              'sp500_p_change':"N/A",
                                              'Difference':"N/A",
                                              'DE Ratio':valuearray[0],
                                              #'Market Cap':value_list[1],
                                              'Trailing P/E':valuearray[1],
                                              'Price/Sales':valuearray[2],
                                              'Price/Book':valuearray[3],
                                              'Profit Margin':valuearray[4],
                                              'Operating Margin':valuearray[5],
                                              'Return on Assets':valuearray[6],
                                              'Return on Equity':valuearray[7],
                                              'Revenue Per Share':valuearray[8],
                                              'Market Cap':valuearray[9],
                                              'Enterprise Value':valuearray[10],
                                              'Forward P/E':valuearray[11],
                                              'PEG Ratio':valuearray[12],
                                              'Enterprise Value/Revenue':valuearray[13],
                                              'Enterprise Value/EBITDA':valuearray[14],
                                              'Revenue':valuearray[15],
                                              'Gross Profit':valuearray[16],
                                              'EBITDA':valuearray[17],
                                              'Net Income Avl to Common ':valuearray[18],
                                              'Diluted EPS':valuearray[19],
                                              'Earnings Growth':valuearray[20],
                                              'Revenue Growth':valuearray[21],
                                              'Total Cash':valuearray[22],
                                              'Total Cash Per Share':valuearray[23],
                                              'Total Debt':valuearray[24],
                                              'Current Ratio':valuearray[25],
                                              'Book Value Per Share':valuearray[26],
                                              'Cash Flow':valuearray[27],
                                              'Beta':valuearray[28],
                                              'Held by Insiders':valuearray[29],
                                              'Held by Institutions':valuearray[30],
                                              'Shares Short (as of':valuearray[31],
                                              'Short Ratio':valuearray[32],
                                              'Short % of Float':valuearray[33],
                                              'Shares Short (prior ':valuearray[34],
                                              'Status':"N/A"},
                                              ignore_index=True)
        
        except Exception as e:
            pass
    


    dataframe.to_csv("forward_with_NA.csv")

    
Forward_HTML()
    
def Stock_Stats():

    df = pd.DataFrame()
    stock_key_path = stock_path + os.path.join('/_KeyStats')
    stock_key = [i[0] for i in os.walk(stock_key_path)]

    for dir in stock_key[80:95]:
        try:
            ticks = dir.split("intraQuarter/_KeyStats/")[1]
            tick_name  = "WIKI/" + ticks.upper()

            key_data = quandl.get(tick_name,
                                  start_date = "2000-12-12",
                                  end_date = "2014-12-30",
                                  api_key = auth_tok)
    

            one_year = None
            key_data[ticks.upper()] = key_data["Adj. Close"]
            df = pd.concat([df, key_data[ticks.upper()]], axis=1)

        except Exception as e:
            #print(str(e))
            pass
            time.sleep(10)
            
            try:
                ticks = dir.split("intraQuarter/_KeyStats/")[1]

                tick_name = "WIKI/" + ticks.upper()
                key_data = quandl.get(tick_name,
                                  start_date = "2000-12-12",
                                  end_date = "2014-12-30",
                                  api_key = auth_tok)
                df = pd.concat([df, key_data[ticks.upper()]], axis = 1)
            except Exception as e:
                #print(str(e))
                pass

    df.to_csv("stock_stats.csv")            
     
def Build():

    data_frame = pd.DataFrame.from_csv("stats_with_NA.csv")

    data_frame = data_frame.reindex(np.random.permutation(data_frame.index))

    
    data_frame["Status2"] = list(map(Status, data_frame["stock_p_change"], data_frame["sp500_p_change"]))

#    print data_frame.shape
    print data_frame

    X = np.array(data_frame[FEATURES].values)
    

 
    
#    print X.shape
#    print X
    X= np.nan_to_num(X)
    


    X = X / X.max(axis=0)

     
    y = (data_frame["Status2"]
         .replace("underperform", 0)
         .replace("outperform", 1)
         .values.tolist())


    
    X = preprocessing.scale(X)

    time.sleep(10)
    
    Z = np.array(data_frame[["stock_p_change","sp500_p_change"]])
    Z = np.nan_to_num(Z)
    Z = preprocessing.scale(Z)
            
            
#    p25 = np.percentile(data_frame[FEATURES], 25)
#    p75 = np.percentile(data_frame[FEATURES], 75)
#    print "percentile 25: " + str(p25)   
#    print "percentile 75: " + str(p75)
#
#    outlier_list = []
#    step = 1.5*(p75-p25)  
#    
#    outlier_array = []
#    outliers_col = (data_frame[~((data_frame[FEATURES].values >= p25 - step) & (data_frame[FEATURES].values <= p75 + step))])
#    print "outlier column"    
#    print outliers_col
#    outlier_list += list(outliers_col.index)
#    
#    outliers = [x for x, count in Counter(outlier_list).items() if count > 1]
#    outlier_array.append(outliers)
#    print "outliers"
#    print outlier_array           
#   
#    good_data = np.delete(data_frame, outliers)
#    print good_data
#    print data_frame      

    p25 = np.percentile(X, 25)
    p75 = np.percentile(X, 75)
    print "percentile 25: " + str(p25)   
    print "percentile 75: " + str(p75)

    outlier_list = []
    step = 1.5*(p75-p25)  
    
    outlier_array = []
    outliers_col = (data_frame[~((X >= p25 - step) & (X <= p75 + step))])
#    print "outlier column"    
#    print outliers_col
    outlier_list += list(outliers_col.index)
    
    outliers = [x for x, count in Counter(outlier_list).items() if count > 1]
    outlier_array.append(outliers)
#    print "outliers"    
#    print outliers
    print outlier_array

    good_data = np.delete(X, outliers)
    print good_data
#    print Z

    return X,y, Z
  
def Analyze():
#    print "def analyze start"    
    testsize = 100
    #originally 1000
    
    amountinvest = 10000
    totaltrades = 0
    #if invest in market
    market = 0
    #if invest in strategy
    strategy = 0
    

    
    X, y, Z = Build()


    """check if jagged array"""
#    print(len(X))
#    for i in range(len(X)):        
#        print(len(X[i]))

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#    print "x train: " + str(X_train)
#    print "z train: "+ str(Z_train)
    trees = tree.DecisionTreeClassifier(random_state=0)
#    trees = tree.DecisionTreeRegressor(random_state=0)
    trees.fit(X_train, y_train, sample_weight= None)
#    X_test_predict = clf.predict(X_test)
#    tree_scores = cross_val_score(clf, X, y, cv=10)
    tree_scores = accuracy_score(y, trees.predict(X), sample_weight=None)
    print "decision tree scores: " + str(tree_scores)
    predicted_tree = trees.predict_proba(X_test)
    skplt.metrics.plot_roc_curve(y_test, predicted_tree)
    plt.show()

    gnb = GaussianNB()
    y_pred = gnb.fit(X, y).predict(X)
    gnb_scores = accuracy_score(y, y_pred, sample_weight=None)
    predicted_gnb = gnb.predict_proba(X_test)
    print "gnb scores: " + str(gnb_scores)
    skplt.metrics.plot_roc_curve(y_test, predicted_gnb)
    plt.show()
    
    forest = RandomForestClassifier(max_depth=2, random_state=0)
    forest_pred = forest.fit(X, y).predict(X)
    forest_scores = accuracy_score(y, forest_pred, sample_weight=None)
    print "forest score: " + str(forest_scores)
    predicted_forest = forest.predict_proba(X_test)
    skplt.metrics.plot_roc_curve(y_test, predicted_forest)
    plt.show()

    
    try:

        clf = svm.SVC(kernel="linear", C = 1.0, probability = True)

        try:
            results = clf.fit(X[:-testsize], y[:-testsize])

        except NotFittedError as e:
            pass

    except ValueError:
        pass
    
    svm_scores = accuracy_score(y, clf.predict(X), sample_weight=None)
    predicted_svm = clf.predict_proba(X_test)
    skplt.metrics.plot_roc_curve(y_test, predicted_svm)
    plt.show()

    print "svm scores: " + str(svm_scores)    
    
    #print "classifying"
    correctcount = 0
#    print "ranging"
    for x in range(1, testsize+1):
                
        #if the 0th element of the list(the prediction) is true
        if clf.predict(X[-x]).reshape(1,-1)[0] == y[x]:
            #correct count then increases
            correctcount += 1

            
        if clf.predict(X[-x]).reshape(1,-1)[0] == 1:
            returninvest = amountinvest + (amountinvest * (Z[-x][0]/100))

            returnmarket = amountinvest + (amountinvest * (Z[-x][1]/100))

            totaltrades += 1

            market += returnmarket

            strategy += returninvest

            
            
    data_frame = pd.DataFrame.from_csv("forward_with_NA.csv")
    data_frame = data_frame.replace("N/A",0).replace("NaN",0).replace("nan",0)
    

    X = np.array(data_frame[FEATURES].values)

    
    
    X = X / X.max(axis=0)

    X = np.nan_to_num(X)
#    tree_clf = tree.DecisionTreeClassifier()


    X = preprocessing.scale(X)
#    print X
#    print "preprocessing yahoo"
    Z = data_frame["Ticker"].values.tolist()
    array = []
#    clf.fit(X, Z, sample_weight=None, check_input=True, X_idx_sorted=None)   
    for each in range(len(X)):   
        t = clf.predict(X[each])[0]
        array.append(t)

    investarray = []

    for i in range(len(X)):
        p = clf.predict(X[i])[0]

        if p == 1:
#            print"Z[i]: " + str(Z[i])
            investarray.append(Z[i])
#            print investarray
        

#    print "Percent of Tickers: " + str((len(investarray)/(len(Z)))*100) + "%"

 
 
    
    print("Accuracy:", (correctcount/testsize) * 100.00)
    print("Total Trades:", totaltrades)
    print("Ending with Strategy:",strategy)
    print("Ending with Market:",market)
    
    try:
        comparison = ((strategy - market)/market)*100.0
    except ZeroDivisionError:
        comparison = 0

    nothing_to_do = totaltrades * amountinvest
    
    try:
        marketaverage = ((market - nothing_to_do)/nothing_to_do)*100.0
    except ZeroDivisionError:
        marketaverage = 0
        
    try:
        strategyaverage = ((strategy  - nothing_to_do)/nothing_to_do)*100.0
    except ZeroDivisionError:
        strategyaverage = 0



    print("Compared to market, we earn",str(comparison)+"% more")
    print("Average investment return:", str(strategyaverage)+"%")
    print("Average market return:", str(marketaverage)+"%")



#    print "def analyze end"
    
    return investarray      

    
#Analyze()
Stock_Stats()

def Stats(gather=["Total Debt/Equity",
                      'Trailing P/E',
                      'Price/Sales',
                      'Price/Book',
                      'Profit Margin',
                      'Operating Margin',
                      'Return on Assets',
                      'Return on Equity',
                      'Revenue Per Share',
                      'Market Cap',
                        'Enterprise Value',
                        'Forward P/E',
                        'PEG Ratio',
                        'Enterprise Value/Revenue',
                        'Enterprise Value/EBITDA',
                        'Revenue',
                        'Gross Profit',
                        'EBITDA',
                        'Net Income Avl to Common ',
                        'Diluted EPS',
                        'Earnings Growth',
                        'Revenue Growth',
                        'Total Cash',
                        'Total Cash Per Share',
                        'Total Debt',
                        'Current Ratio',
                        'Book Value Per Share',
                        'Cash Flow',
                        'Beta',
                        'Held by Insiders',
                        'Held by Institutions',
                        'Shares Short (as of',
                        'Short Ratio',
                        'Short % of Float',
                        'Shares Short (prior ']):
#    print "def stats start"
    stock_key_path = stock_path + os.path.join('/_KeyStats')
    stock_key = [i[0] for i in os.walk(stock_key_path)]
#    print "stats path"
    df = pd.DataFrame(columns = ['Date',
                                 'Unix',
                                 'Ticker',
                                 'Price',
                                 'stock_p_change',
                                 'SP500',
                                 'sp500_p_change',
                                 'Difference',
                                 ##############
                                 'DE Ratio',
                                 'Trailing P/E',
                                 'Price/Sales',
                                 'Price/Book',
                                 'Profit Margin',
                                 'Operating Margin',
                                 'Return on Assets',
                                 'Return on Equity',
                                 'Revenue Per Share',
                                 'Market Cap',
                                 'Enterprise Value',
                                 'Forward P/E',
                                 'PEG Ratio',
                                 'Enterprise Value/Revenue',
                                 'Enterprise Value/EBITDA',
                                 'Revenue',
                                 'Gross Profit',
                                 'EBITDA',
                                 'Net Income Avl to Common ',
                                 'Diluted EPS',
                                 'Earnings Growth',
                                 'Revenue Growth',
                                 'Total Cash',
                                 'Total Cash Per Share',
                                 'Total Debt',
                                 'Current Ratio',
                                 'Book Value Per Share',
                                 'Cash Flow',
                                 'Beta',
                                 'Held by Insiders',
                                 'Held by Institutions',
                                 'Shares Short (as of',
                                 'Short Ratio',
                                 'Short % of Float',
                                 'Shares Short (prior ',                                
                                 ##############
                                 'Status'])
                                 

    
    sp500_dataframe = pd.DataFrame.from_csv("YAHOO-INDEX_GSPC.csv")
    stock_dataframe = pd.DataFrame.from_csv("stock_stats.csv")

    tick_list = []

    for dir in stock_key[1:]:
        file = os.listdir(dir)
        ticks = dir.split("intraQuarter/_KeyStats/")[1]
        tick_list.append(ticks)        
    
        if len(file) > 0:
            for eachfile in file:
                datestamp = datetime.strptime(eachfile, '%Y%m%d%H%M%S.html')
                untime = time.mktime(datestamp.timetuple())
                file_path = dir + '/' + eachfile
                source = open(file_path,'r').read()
                
                try:
                    val_list = []
                
                    for data in gather:
                        try:
                            reg = re.escape(data) + r'.*?(\d{1,8}\.\d{1,8}M?B?|N/A)%?</td>'
                            value = re.search(reg, source)
                            value = (value.group(1))
#                            print "Value in Stats"
#                            print value
                            
                            if "B" in value:
                                value = float(value.replace("B",''))*1000000000
                            elif "M" in value:
                                value = float(value.replace("M",''))*1000000
                            val_list.append(value)

                        except Exception as e:
                            value = "N/A"
                            val_list.append(value)
                
                    try:
                        sp500_date = datetime.fromtimestamp(untime).strftime('%Y-%m-%d')
                        row = sp500_dataframe[(sp500_dataframe.index == sp500_date)]
                        sp500_value = float(row["Adj Close"])
                    except:
                        #259200 is number of seconds in 3 days
                        sp500_date = datetime.fromtimestamp(untime-259200).strftime('%Y-%m-%d')
                        row = sp500_dataframe[(sp500_dataframe.index == sp500_date)]
                        sp500_value = float(row["Adj Close"])
                    try:
                        stock_price = float(source.split('</small><big><b>')[1].split('</b></big>')[0])
                    except Exception as e:
                        pass
                    
                    
                    one_year = int(untime + 31536000)
                    
                    try: 
                        sp500_last = datetime.fromtimestamp(one_year).strftime('%Y-%m-%d')
                        row = sp500_dataframe[(sp500_dataframe.index == sp500_last)]
                        sp500_last_value = float(row["Adj Close"])
                    except:
                        try:
                            sp500_last = datetime.fromtimestamp(one_year-259200)
                            row = sp500_dataframe[(sp500_dataframe.index == sp500_last)]
                            sp500_last_value = float(row["Adj Close"])
                        except Exception as e:
                            #print("sp500 1 year later issue", str(e))
                            pass
                            
                            
                    try:
                        stock_price_last = datetime.fromtimestamp(one_year).strftime('%Y-%m-%d')
                        row = stock_dataframe[(stock_dataframe.index == stock_price_last)][ticker.upper()]
                        
                        stock_ly_value = round(float(row),2)
#                        print(stock_ly_value)
                        time.sleep(1555)
                        
                    except Exception as e:
                        try:
                            stock_price_ly = datetime.fromtimestamp(one_year-259200).strftime('%Y-%m-%d')
                            row = stock_dataframe[(stock_dataframe.index == stock_price_ly)][ticks.upper()]
                            stock_last_value = round(float(row),2)
                        except Exception as e:
                            #print("stock price:",str(e))
                            pass
                            
                    try:
                        stock_price = datetime.fromtimestamp(untime).strftime('%Y-%m-%d')
                        row = stock_dataframe[(stock_dataframe.index == stock_price)][ticks.upper()]
                        stock_price = round(float(row),2)
                        
                    except Exception as e:
                        try:
                            stock_price = datetime.fromtimestamp(untime-259200).strftime('%Y-%m-%d')
                            row = stock_dataframe[(stock_dataframe.index == stock_price)][ticks.upper()]
                            stock_price = round(float(row),2)
                        except Exception as e:
                            #print("stock price:",str(e))
                            pass
                    stock_change = round((((stock_last_value - stock_price) / stock_price) * 100),2)
                    sp500_change = round((((sp500_last_value - sp500_value) / sp500_value) * 100),2)
                    
                    
                    diff = stock_change-sp500_change
                    if diff > 5:
                        stat = 1
                    else:
                        stat = 0
                                        
                    
                        
                    if val_list.count("N/A") > 15:
                        pass
                    else:
                        
                        df = df.append({'Date':datestamp,
                                        'Unix':untime,
                                        'Ticker':ticks,
                                        'Price':stock_price,
                                        'stock_p_change':stock_change,
                                        'SP500':sp500_value,
                                        'sp500_p_change':sp500_change,
                                        ###########
                                        'Difference':diff,
                                        'DE Ratio':val_list[0],
                                        #'Market Cap':value_list[1],
                                        'Trailing P/E':val_list[1],
                                        'Price/Sales':val_list[2],
                                        'Price/Book':val_list[3],
                                        'Profit Margin':val_list[4],
                                        'Operating Margin':val_list[5],
                                        'Return on Assets':val_list[6],
                                        'Return on Equity':val_list[7],
                                        'Revenue Per Share':val_list[8],
                                        'Market Cap':val_list[9],
                                        'Enterprise Value':val_list[10],
                                        'Forward P/E':val_list[11],
                                        'PEG Ratio':val_list[12],
                                        'Enterprise Value/Revenue':val_list[13],
                                        'Enterprise Value/EBITDA':val_list[14],
                                        'Revenue':val_list[15],
                                        'Gross Profit':val_list[16],
                                        'EBITDA':val_list[17],
                                        'Net Income Avl to Common ':val_list[18],
                                        'Diluted EPS':val_list[19],
                                        'Earnings Growth':val_list[20],
                                        'Revenue Growth':val_list[21],
                                        'Total Cash':val_list[22],
                                        'Total Cash Per Share':val_list[23],
                                        'Total Debt':val_list[24],
                                        'Current Ratio':val_list[25],
                                        'Book Value Per Share':val_list[26],
                                        'Cash Flow':val_list[27],
                                        'Beta':val_list[28],
                                        'Held by Insiders':val_list[29],
                                        'Held by Institutions':val_list[30],
                                        'Shares Short (as of':val_list[31],
                                        'Short Ratio':val_list[32],
                                        'Short % of Float':val_list[33],
                                        'Shares Short (prior ':val_list[34],
                                        'Status':stat},
                                        ignore_index=True)
                except Exception as e:
                    pass
    

                
#    averagestock = np.mean(stock_price)
    averagesp = df['SP500'].mean()
    averagediff = df['Difference'].mean()
    
    mediansp = df['SP500'].median()
    mediandiff = df['Difference'].median()
    
    stdsp = df['SP500'].std()
    stddiff = df['Difference'].std()
    
#    modesp = np.mode(sp500_value)
#    modediff = np.mode(diff)
    
    variancesp = df['SP500'].var()
    variancediff = df['Difference'].var()
    
    minsp = df['SP500'].min()
    mindiff = df['Difference'].min()
    
    maxsp = df['SP500'].max()
    maxdiff = df['Difference'].max()

        
    
#    print "Average Stock Adjusted Close: " + str(averagestock)
    print "Minimum S&P 500 Adjusted Close: " + str(minsp)
    print "Maximum S&P 500 Adjusted Close: " + str(maxsp)
    print "Average S&P 500 Adjusted Close: " + str(averagesp)
    print "Median S&P 500 Adjusted Close: " + str(mediansp)
    print "Standard Deviation S&P 500 Adjusted Close: " + str(stdsp)
    print "Variance S&P 500 Adjusted Close: " + str(variancesp)
    print "Minimum Difference: " + str(mindiff)
    print "Maximum Difference: " + str(maxdiff)
    print "Average Difference: " + str(averagediff)
    print "Median Difference: " + str(mediandiff)
    print "Standard Deviation of Difference: " + str(stddiff)
    print "Variance Difference: " + str(variancediff)
      
    df.to_csv("stats_with_NA.csv")

    return sp500_last_value, sp500_value

Stats()





list_final = []

loop = 8

for x in range(loop):

    stockarray = Analyze()
    for e in stockarray:

        list_final.append(e)

x = Counter(list_final)

print(15*"_")
print "Recommended stock tickers"
print(15*"_")
    
for each in x:
    if x[each] > loop - (loop/3):
        print each
        pass






ticker_input()
   

print datetime.now()-start
    

