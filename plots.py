import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import FactorAnalysis
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

def plot_conversion():
    # Define the touchpoints
    touchpoints = ["Landing Page", "Product Page", "Cart", "Checkout", "Payment"]

    # Number of users at each touchpoint (example data)
    user_flow = [1000, 800, 600, 400, 350]
    drop_offs = [0, 200, 200, 200, 50]

    # Create source and target lists for Sankey diagram
    sources = []
    targets = []
    values = []

    # Populate the sources, targets, and values for continued interactions
    for i in range(len(touchpoints) - 1):
        sources.append(i)
        targets.append(i + 1)
        values.append(user_flow[i + 1])

    # Populate the sources, targets, and values for drop-offs
    for i in range(len(touchpoints) - 1):
        sources.append(i)
        targets.append(len(touchpoints) + i)  # Drop-off nodes
        values.append(drop_offs[i + 1])

    # Create labels including drop-off labels
    labels = touchpoints + [f"Drop-off after {tp}" for tp in touchpoints[:-1]]

    # Define the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(           
            pad=15,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=labels,
            color="blue"                    
        ),
        link=dict(
            source=sources,  # Indices of the source nodes
            target=targets,  # Indices of the target nodes
            value=values,  # Number of users flowing between nodes
            color=["rgba(31, 119, 180, 0.8)"] * len(sources) + ["rgba(255, 0, 0, 0.4)"] * len(drop_offs)
        )
    )])

    # Set the layout of the diagram.
    fig.update_layout(font=dict(size=8),
                      width=400, 
                      height=200,
                      margin=dict(l=0,r=0,b=0,t=0,pad=0),
                      paper_bgcolor="black"
                      ) 

    st.plotly_chart(fig)

    # Display the Sankey diagram
    #fig.show()

def conversion_funnel_chart():
        # Define the touchpoints
    touchpoints = ["Landing Page", "Product Page", "Cart", "Checkout", "Payment"]

    # Number of users at each touchpoint (example data)
    user_flow = [1000, 800, 600, 400, 350]    

    # Create the funnel chart
    fig = go.Figure(go.Funnel(
        y=touchpoints,  # Labels for each stage
        x=user_flow,    # Number of users at each stage
        textinfo="value+percent previous+percent initial",  # Display the value and percentage
        # hoverinfo="label+percent previous+percent initial+text",
        textposition="inside",
        marker=dict(color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])        
    ))

    # Set the layout of the funnel chart
    fig.update_layout(
        xaxis_title="Number of Users",
        yaxis_title="Touchpoints",
        yaxis=dict(categoryorder="total ascending"),
        font=dict(size=10),
        width=400, 
        height=200,
        margin=dict(l=1,r=10,b=10,
                t=1,pad=0
                ),
    paper_bgcolor="black"
    )

    st.plotly_chart(fig)

def cluster_segment_radar_chart():        

    # Sample data for customer segments
    categories = ['Average Age', 'Purchase Frequency', 'Order Value', 'Loyalty Score']
    segment_data = {
        'Segment A (Age Gr.30-40s)': [35, 5, 150, 80],
        'Segment B (Age Gr. > 40s)': [42, 3, 120, 90],
        'Segment C (Age Gr. < 30s)': [28, 7, 100, 70]
    }

    segment_mean = list(pd.DataFrame(segment_data).mean(axis=1))
    segment_std = list(pd.DataFrame(segment_data).std(axis=1))

    for segment in segment_data:
        for i in range(len(segment_data[segment])):
            segment_data[segment][i] = (segment_data[segment][i] - segment_mean[i]) / segment_std[i]

    # Create radar chart
    fig = go.Figure()

    # Add traces for each segment
    for segment, values in segment_data.items():
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=segment
        ))

        fig.update_layout(
            title="Customer Segmentation Analysis for Chisamba Company.",
            title_x=0.18,  # Title aligned with grid
            title_y=0.95,  # Title positioned near the top vertically
            polar=dict(
                radialaxis=dict(
                    visible=True
                )
            ),

        margin=dict(l=100, r=100, t=100, b=100),  # Remove margins     
        showlegend=True  
        )

    st.plotly_chart(fig)

def plot_regression(): 

    # Generate a fictitious dataset
    np.random.seed(42)
    advertising_expenditure = np.random.normal(100, 20, 100)
    sales_revenue = 2.5 * advertising_expenditure + np.random.normal(0, 25, 100)

    # Introduce some outliers
    advertising_expenditure[95:] += 50
    sales_revenue[95:] += 200

    # Create a DataFrame
    data = pd.DataFrame({
        'Advertising Expenditure': advertising_expenditure,
        'Sales Revenue': sales_revenue
    })

    # Split the data into training and testing sets
    X = data[['Advertising Expenditure']]
    y = data['Sales Revenue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)  

    # Create a scatter plot with regression line
    fig = go.Figure()

    # Scatter plot of the data
    fig.add_trace(go.Scatter(x=data['Advertising Expenditure'], y=data['Sales Revenue'],
                            mode='markers', name='Data'))

    # Regression line
    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range)
    fig.add_trace(go.Scatter(x=x_range.flatten(), y=y_range, mode='lines', name='Regression Line'))

    # Highlight outliers
    fig.add_trace(go.Scatter(x=advertising_expenditure[95:], y=sales_revenue[95:], 
                            mode='markers', name='Outliers', marker=dict(color='red', size=10)))

    fig.update_layout(
                    #title='Regression Analysis of Advertising Expenditure vs Sales Revenue',
                    xaxis_title='Advertising Expenditure in EUR.',
                    yaxis_title='Sales Revenue in EUR.',                      
                    margin=dict(l=100,r=100,b=150,t=0,pad=0)                 
                )
    
    st.plotly_chart(fig)    


def plot_times_series_data():    
    
    dates = pd.date_range(start='2024-10-01', periods=100, freq='D')

    # Create a fictitious demand data with a trend and seasonality
    np.random.seed(42)
    demand = 50 + np.arange(100) * 0.5 + 10 * np.sin(np.linspace(0, 20, 100)) + np.random.normal(scale=5, size=100)    

    # Create a DataFrame
    data = pd.DataFrame({'Date': dates, 'Demand': demand})    
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Demand'], mode='lines', name='Demand'))
    fig.update_layout(title='Historical Demand', xaxis_title='Date', yaxis_title='Demand')    

    # Split the data into training and testing sets
    train_data = data['Demand'][:80] 

    # Fit the SARIMA model
    model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)

    # Forecast
    forecast = model_fit.forecast(steps=20)  

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Demand'], mode='lines', name='Historical Demand'))
    fig.add_trace(go.Scatter(x=data['Date'][80:], y=forecast, mode='lines', line=dict(color='red'),name='Forecast'))
    fig.update_layout(title='Demand Forecast.', title_x=0.40, title_y=0.95, xaxis_title='Date', yaxis_title='Demand')
               
    st.plotly_chart(fig)

def plot_market_basket_associations():
    # Sample fictitious dataset
    data = {
        'TransactionID': [1, 2, 3, 4, 5],
        'Items': [
            ['Milk', 'Bread', 'Butter'],
            ['Bread', 'Butter'],
            ['Milk', 'Bread'],
            ['Butter', 'Milk'],
            ['Bread', 'Butter', 'Milk']
        ]
    }

    # Convert dataset to DataFrame
    df = pd.DataFrame(data)

    # One-hot encoding the dataset
    basket = df['Items'].str.join('|').str.get_dummies()

    # Applying Apriori algorithm
    frequent_itemsets = apriori(basket, min_support=0.6, use_colnames=True)

    # Generating association rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    # Visualizing the rules using plotly.graph_objects
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(rules.columns),
                    fill_color='gray',
                    align='left'),
        cells=dict(values=[rules[col] for col in rules.columns],
                fill_color='blue',
                align='left'))
    ])

    fig.update_layout(title='Association Rules for Chisamba Marketing and E-commerce Company')
    st.plotly_chart(fig)

def plot_customer_lifetime_value():

    # Creating a fictitious dataset
    data = {
        'CustomerID': range(1, 101),
        'PurchaseAmount': np.random.uniform(50, 500, 100),
        'PurchaseFrequency': np.random.randint(1, 10, 100),
        'CustomerLifespan': np.random.randint(1, 5, 100)
    }

    df = pd.DataFrame(data)
   
    # Calculate CLV
    df['CLV'] = df['PurchaseAmount'] * df['PurchaseFrequency'] * df['CustomerLifespan']

    # Identify high-value customers
    high_value_threshold = df['CLV'].quantile(0.75)
    high_value_customers = df[df['CLV'] > high_value_threshold]

    # Visualize high-value customers
    # Plot CLV distribution
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=high_value_customers['CLV'],
        nbinsx=20,
        name='High-Value Customers',
        marker_color='red',
        opacity=0.75
    ))

    fig.update_layout(
        title='Customer Lifetime Value Distribution.',
        title_x=0.35,  # Title aligned with grid
        title_y=0.90,  # Title positioned near the top vertically
        xaxis_title='Customer Lifetime Value (CLV)',
        yaxis_title='Count',        
        margin=dict(l=100,r=100,b=100,t=100,pad=0),
        barmode='overlay',
        template='plotly'
    )
    st.plotly_chart(fig)


def plot_churn_prediction():

    # Fictitious dataset
    data = {
        'customer_id': range(1, 101),
        'age': np.random.randint(18, 70, size=100),
        'gender': np.random.choice([1, 0], size=100),
        'annual_income': np.random.randint(30000, 120000, size=100),
        'purchase_frequency': np.random.randint(1, 20, size=100),
        'last_purchase_days': np.random.randint(1, 365, size=100),
        'churn': np.random.choice([0, 1], size=100)
    	}

    df = pd.DataFrame(data)

    # Data preprocessing
    X = df.drop(['customer_id', 'churn'], axis=1)
    y = df['churn']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model building
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Model evaluation
    # print(classification_report(y_test, y_pred))

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plotting ROC Curve with threshold
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve (AUC = {:.2f})'.format(roc_auc)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Chance', line=dict(dash='dash')))

    # Adding threshold
    threshold = 0.60
    threshold_index = np.where(thresholds >= threshold)[0][0]
    fig.add_trace(go.Scatter(x=[fpr[threshold_index]], y=[tpr[threshold_index]], mode='markers', name='Threshold = {:.2f}'.format(threshold)))

    fig.update_layout(title='ROC Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    showlegend=True)

    st.plotly_chart(fig)


def plot_sentiments():
    np.random.seed(42)
    reviews = ['Review {}'.format(i) for i in range(1, 101)]
    sentiments = np.random.uniform(-1, 1, 100)

    # Create a DataFrame
    data = pd.DataFrame({
        'Review': reviews,
        'Sentiment': sentiments
    })

    # Classify sentiments
    data['Sentiment_Label'] = pd.cut(data['Sentiment'], bins=[-1, -0.6, 0.1, 1], labels=['Negative', 'Neutral', 'Positive'])

    # Count sentiment labels
    sentiment_counts = data['Sentiment_Label'].value_counts().sort_index()

    # Plot sentiment distribution using Plotly
    fig = go.Figure(data=[
        go.Bar(name='Sentiment', x=sentiment_counts.index, y=sentiment_counts.values)
    ])

    fig.update_layout(
        title='Chisamba Marketing Co. Sentiment Distribution of Customer Review.',
        title_x=0.22,  # Title aligned with grid
        title_y=0.90,  # Title positioned near the top vertically
        xaxis_title='Sentiment',
        yaxis_title='Count',
        template='plotly'
    )
    st.plotly_chart(fig)        


def plot_conjoint():
    # Fictitious dataset for Chisamba's product attributes
    data = {
        'Product': ['A', 'B', 'C', 'D', 'E'],
        'Price': [10, 15, 20, 25, 30],
        'Quality': [60, 8, 10, 40, 6],
        'Brand': [4, 8, 20, 6, 2],
        'Sales': [100, 150, 200, 250, 30]
    }

    df = pd.DataFrame(data)

    # Define the attributes and levels
    attributes = ['Price', 'Quality', 'Brand']
    X = df[attributes]
    y = df['Sales']

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Get the part-worth utilities (coefficients)
    part_worths = model.coef_
    intercept = model.intercept_    

    # Create a bar chart for part-worth utilities
    fig = go.Figure()

    for i, attribute in enumerate(attributes):
        fig.add_trace(go.Bar(
            x=[attribute],
            y=[part_worths[i]],
            name=attribute
        ))

    fig.update_layout(
        title="Part-worth Utilities for Chisamba's Product Attributes.",
        title_x=0.25,  # Title aligned with grid
        title_y=0.95,  # Title positioned near the top vertically
        xaxis_title="Attributes",
        yaxis_title="Part-worth Utility",
        barmode='group'
    )

    st.plotly_chart(fig)


def plot_multi_variate():

    # Sample fictitious dataset
    data = {
        'Marketing_Spend': [200, 300, 400, 500, 600],
        'Sales': [20, 30, 40, 50, 60],
        'Customer_Satisfaction': [3.5, 4.0, 4.5, 5.0, 4.8],
        'Website_Visits': [1000, 1500, 2000, 2500, 3000]
    }

    df = pd.DataFrame(data)
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Apply PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_data)
    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    # Create a scatter plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_pca['PC1'],
        y=df_pca['PC2'],
        mode='markers',
        marker=dict(size=10, color='blue'),
        text=df.index
    ))

    fig.update_layout(
        title='PCA of Chisamba Marketing Data.',
        title_x=0.38,  # Title aligned with grid
        title_y=0.90,  # Title positioned near the top vertically
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2'
    )

    st.plotly_chart(fig)


def plot_factor_analysis():

    # Creating a fictitious dataset
    np.random.seed(0)
    data = pd.DataFrame({
        'Customer_Satisfaction': np.random.rand(100),
        'Purchase_Frequency': np.random.rand(100),
        'Website_Visits': np.random.rand(100),
        'Ad_Clicks': np.random.rand(100),
        'Product_Reviews': np.random.rand(100)
    })

    # Performing Factor Analysis
    fa = FactorAnalysis(n_components=2)
    factors = fa.fit_transform(data)

    # Adding factor scores to the dataframe
    data['Factor1'] = factors[:, 0]
    data['Factor2'] = factors[:, 1]

    # Creating a scatter plot for the factors
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data['Factor1'], y=data['Factor2'],
        mode='markers',
        marker=dict(size=10, color=data['Customer_Satisfaction'], colorscale='Viridis', showscale=True),
        text=data.index
    ))

    fig.update_layout(
        title='Factor Analysis of Customer Behavior.',
        title_x=0.25,  # Title aligned with grid
        title_y=0.90,  # Title positioned near the top vertically
        xaxis_title='Factor 1',
        yaxis_title='Factor 2',
        showlegend=False
    )

    st.plotly_chart(fig)

def plot_prediction():

    # Create the dataset
    data = {
        'CustomerID': range(1, 101),
        'Age': np.random.randint(18, 70, 100),
        'AnnualIncome': np.random.randint(20000, 100000, 100),
        'PurchaseAmount': np.random.randint(100, 2000, 100)
    }
    df = pd.DataFrame(data)

    # Define the independent variable (AnnualIncome) and dependent variable (PurchaseAmount)
    X = df['AnnualIncome'].values.reshape(-1, 1)
    y = df['PurchaseAmount']

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Generate predictions
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_range = model.predict(x_range)

    # Create the scatter plot and regression line
    fig = go.Figure()

    # Add scatter plot
    fig.add_trace(go.Scatter(x=df['AnnualIncome'], y=df['PurchaseAmount'], mode='markers', name='Data Points'))

    # Create the dataset
    data = {
        'CustomerID': range(1, 101),
        'Age': np.random.randint(18, 70, 100),
        'AnnualIncome': np.random.randint(20000, 100000, 100),
        'PurchaseAmount': np.random.randint(100, 2000, 100)
    }
    df = pd.DataFrame(data)

    # Define the independent variable (AnnualIncome) and dependent variable (PurchaseAmount)
    X = df['AnnualIncome'].values.reshape(-1, 1)
    y = df['PurchaseAmount']

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Generate predictions
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_range = model.predict(x_range)

    # Create the scatter plot and regression line
    fig = go.Figure()

    # Add scatter plot
    fig.add_trace(go.Scatter(x=df['AnnualIncome'], y=df['PurchaseAmount'], mode='markers', name='Data Points'))

    # Add regression line
    fig.add_trace(go.Scatter(x=x_range.flatten(), y=y_range, mode='lines', line=dict(color="#0000ff"), name='Regression Line'))

    # Update layout
    fig.update_layout(title='Linear Regression of Purchase Amount vs Annual Income',
                    xaxis_title='Annual Income',
                    yaxis_title='Purchase Amount')

    fig.add_trace(go.Scatter(x=x_range.flatten(), y=y_range, mode='lines', name='Regression Line'))

    # Update layout
    fig.update_layout(title='Linear Regression of Purchase Amount vs Annual Income.',
                    title_x=0.27,  # Title aligned with grid
                    title_y=0.90,  # Title positioned near the top vertically
                    xaxis_title='Annual Income',
                    yaxis_title='Purchase Amount')
    st.plotly_chart(fig)
