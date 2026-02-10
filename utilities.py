import streamlit as st

# constant Variables
star_and_space = '&#x2605;&nbsp;'
square_bullet_point ='&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#x2751'

dict_theories={
        "Theory 0: A/B Testing": "A/B Testing is used to compare two versions of a webpage or app to see which one performs better. It can be used in other sectors like healthcare to compare treatment methods.",
        "Theory 1: A/B Testing": "Customer Journey with Drop-offs at Each Touchpoint.",
        "Theory 2: Regression Analysis": "Regression Analysis helps in understanding the relationship between variables. It is widely used in finance for risk management.",
        "Theory 3: Time Series Analysis.": "Time Series Analysis (TSA) is a statistical technique used to analyze time-ordered data points to identify patterns, trends, and seasonal variations.",
        "Theory 5: Market Basket Analysis.":"Market Basket Analysis (MBA) is a data mining technique used to uncover associations between items in large datasets.",
        "Churn Prediction.":"Customer churn prediction involves identifying customers who are likely to stop using a company’s products or services."
         
}

theory_explained=[
    "Conversion Analysis at each Touchpoint is a Customer behavioural analytics & benchmark that measures the performance of marketing campaigns.",
    "Customer Segmentation Analysis is the practice of dividing a company’s customers into groups that reflect similarity in each group.",
    "Linear Regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables. The goal is to find the best-fitting line through the data points ",
    "Time Series Analysis (TSA) is a statistical technique used to analyze time-ordered data points to identify patterns, trends, and seasonal variations.",
    #"Market Basket Analysis (MBA) is a data mining technique used to uncover associations between items in large datasets.",
    "Customer Lifetime Value (CLV) is a metric that estimates the total revenue a business can expect from a single customer account throughout the business relationship. ",
    "Customer churn prediction involves identifying customers who are likely to stop using a company’s products or services based on features like age, gender, annual income, purchase frequency, last purchase days, and churn stat",
    "Sentiment analysis, also known as opinion mining, is a natural language processing (NLP) technique used to determine whether data is positive, negative, or neutral. It involves analyzing text data to extract subjective information, which helps businesses understand customer opinions and feedback.",
    "Conjoint Analysis is a statistical technique used in market research to determine how people value different features that make up an individual product or service. ",
    "Multivariate Analysis is a statistical technique used to analyze data that involves multiple variables.",
    #"Factor Analysis is a statistical method used to identify the underlying relationships between a large set of variables. It reduces the data complexity by grouping correlated variables into factors. This helps in understanding the structure of the data and identifying key drivers.",
    "Predictive modeling uses statistical techniques and machine learning algorithms to predict future outcomes based on historical data."
]

purpose_expertise = [
    "Showcasing how to optimize to marketing campaigns, revenue generation or reduced churn, and identification of communication flaws marketing strategies. ",
    "This shows the importance a company's knowledge about its customers in order to maximize the value of each customer to the business.",
    "The purpose was to understand how advertising expenditure impacts sales revenue. By doing so, Chisamba Marketing can optimize their advertising budget to maximize sales.",
    "My primary goal was to forecast future values based on historical data. The project can be used in the  context of supply chain and procurement, predict future demand, optimize inventory levels, and improve procurement strategies.",
    #"This helps in understanding the purchase customers' behaviour by identifying items that are frequently bought together, done using association rule learning algorithms.",
    "With it the businesses understands the value of their customers and make informed decisions about marketing strategies, customer retention, and resource allocation.",
    "The purpose of this classification project was to develop a predictive model that can identify customers at risk of churning by using Random Forest algorithm.",
    "The purpose of this project is to analyze customer reviews for Chisamba, a fictitious marketing and e-commerce company, to gain insights into customer satisfaction and identify areas for improvement. By understanding customer sentiment, Chisamba can enhance its products, services, and overall customer experience.",
    "The purpose of this analysis is to understand customer preferences and predict their decision-making process. This helps businesses in product development, pricing strategies, and marketing.",
    "The project and the intent helps in understanding the relationships between variables and how they interact with each other. This is particularly useful in marketing and e-commerce to identify patterns, trends, and insights that can drive business decisions.",
    #"The Purpose of the technique is to reduce the dimensionality of the dataset and identify latent variables (factors) that explain the observed correlations and finally improve marketing strategies by understanding customer behaviour.",
    f"This project used a regression model to predict customer purchases. The goal is to help Chisamba Marketing Co. to optimize their marketing strategies by predicting which ctomers are likely to make purchases, allowing for targeted marketing campaigns. Expertise Acquired are: <br>{square_bullet_point} Understanding of predictive modeling and regression analysis. <br>{square_bullet_point} Skills in data preprocessing, model training, and evaluation. <br>{square_bullet_point} Proficiency in data visualization using plotly.graph_objects. <br>{square_bullet_point} Ability to tell a compelling data story through visualizations."

]




