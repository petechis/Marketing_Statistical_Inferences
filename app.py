import streamlit as st
import matplotlib.pyplot as plt
import plots as ps
import utilities as us

st.set_page_config(page_title="Statistical Inferences!", layout="wide")

# Define the list of marketing statistical theories
theories = [
    "Theory 1: Customer Conversions.",
    "Theory 2: Customer Segmentation Analysis.",
    "Theory 3: Regression Analysis.",
    "Theory 4: Time Series Analysis.",    
    "Theory 5: Customer Lifetime Value.",
    "Theory 6: Churn Prediction.",
    "Theory 7: Sentiment Analysis.",
    "Theory 8: Conjoint Analysis.",
    "Theory 9: Multivariate Analysis.",
    "Theory 10: Predictive Modeling."]

def custom_title(title, color, fsize, weight):
    st.markdown(f"<p style='text-align: center; color : {color}; font-size:{fsize}px; font-weight:{weight}'>{title}</p>", unsafe_allow_html=True)

def custom_sidebar_title(title, color, fsize, weight):
    st.sidebar.markdown(f"<p style='text-align: left; color : {color}; font-size:{fsize}px; font-weight:{weight}'>{title}</p>", unsafe_allow_html=True)

def custom_text(text, color, fsize, weight):
    st.sidebar.markdown(f"<p style='text-align: left; color : {color}; font-size:{fsize}px; font-weight:{weight}'>{text}</p>", unsafe_allow_html=True)

def custom_text_main(title, color, fsize, weight, align):
    st.markdown(f"<p style='text-align: {align}; color : {color}; font-size:{fsize}px; font-weight:{weight}'>{title}</p>", unsafe_allow_html=True)

st.sidebar.image('img/chisamba_marketing.png')

# Sidebar for selecting a theory
selected_theory = st.sidebar.selectbox("Select a Marketing Theory.", theories)
#st.sidebar.divider()
custom_sidebar_title(f'{us.star_and_space}Theory Explained:','lightgray',24,'normal')

with st.sidebar:
    if "1:" in selected_theory:
        custom_text(us.theory_explained[0],'white',12,'normal')
    elif "2:" in selected_theory:
       custom_text(us.theory_explained[1],'white',12,'normal')
    elif "3:" in selected_theory:
        custom_text(us.theory_explained[2],'white',12,'normal')
    elif "4:" in selected_theory:
        custom_text(us.theory_explained[3],'white',12,'normal')        
    elif "5:" in selected_theory:
        custom_text(us.theory_explained[4],'white',12,'normal')          
    elif "6:" in selected_theory:
        custom_text(us.theory_explained[5],'white',12,'normal') 
    elif "7:" in selected_theory:
        custom_text(us.theory_explained[6],'white',12,'normal')
    elif "8:" in selected_theory:
        custom_text(us.theory_explained[7],'white',12,'normal')
    elif "9:" in selected_theory:
        custom_text(us.theory_explained[8],'white',12,'normal')
    elif "10:" in selected_theory:
        custom_text(us.theory_explained[9],'white',12,'normal')
    elif "11:" in selected_theory:
        custom_text(us.theory_explained[10],'white',12,'normal')
    elif "12:" in selected_theory:
        custom_text(us.theory_explained[11],'white',12,'normal')
    else:
        st.write("No more data to analyse ...")    
    
custom_sidebar_title(f'{us.star_and_space} Purpose & Expertise:','lightgray',24,'normal')
with st.sidebar:
    if "1:" in selected_theory:
        custom_text(us.purpose_expertise[0],'white',12,'normal')
    elif "2:" in selected_theory:
        custom_text(us.purpose_expertise[1],'white',12,'normal')
    elif "3:" in selected_theory:
        custom_text(us.purpose_expertise[2],'white',12,'normal')
    elif "4:" in selected_theory:
        custom_text(us.purpose_expertise[3],'white',12,'normal')
    elif "5:" in selected_theory:
        custom_text(us.purpose_expertise[4],'white',12,'normal')
    elif "6:" in selected_theory:
        custom_text(us.purpose_expertise[5],'white',12,'normal')
    elif "7:" in selected_theory:
        custom_text(us.purpose_expertise[6],'white',12,'normal')
    elif "8:" in selected_theory:
        custom_text(us.purpose_expertise[7],'white',12,'normal')
    elif "9:" in selected_theory:
        custom_text(us.purpose_expertise[8],'white',12,'normal')
    elif "10:" in selected_theory:
        custom_text(us.purpose_expertise[9],'white',12,'normal')
    elif "11:" in selected_theory:
        custom_text(us.purpose_expertise[10],'white',12,'normal')
    elif "12:" in selected_theory:
        custom_text(us.purpose_expertise[11],'white',12,'normal')
    else:
        st.write("Work in progress!")


# Function to generate a chart based on the selected theory
def generate_chart(theory):
    fig, ax = plt.subplots()
    # Example: Generate a simple bar chart
    ax.bar(["Feature 1", "Feature 2", "Feature 3"], [10, 20, 15])
    ax.set_title(f"Chart for {theory}")
    return fig

# Function to explain the theory
def explain_theory(theory):
    explanations = {  
        "Theory 0: A/B Testing.": "A/B Testing is used to compare two versions of a webpage or app to see which one performs better. It can be used in other sectors like healthcare to compare treatment methods.",
        "Theory 1: Customer Conversions.": f"Chisamba Marketing Co., is experiencing significant customer drop-offs at various touchpoints in their marketing funnel. They were significant drop-off rates at each stage of their customer journey. <br>{us.square_bullet_point} By leveraging data storytelling, they can visualize and understand where and why these drop-offs occur and propose actionable measures to improve customer retention. For example, they discovered that 20% of visitors left the website due to poor navigation and slow loading times or only 35% of customers complete the transactions. <br> {us.square_bullet_point} By analysing graphs, Chisamba Marketing can obtain insights of the problem areas and clearly communicate the issues and solutions related to customer drop-offs at each touchpoint. This approach not only helps Chisamba Marketing Co. in identifying problem areas but also in implementing effective strategies to enhance customer retention and drive business growth.",
        "Theory 2: Customer Segmentation Analysis.": f"Chisamba Marketing Co. has segmented its customer base into three distinct groups based on age: <br>{us.square_bullet_point} Segment A (Age Group 30-40s), <br>{us.square_bullet_point} Segment B (Age Group > 40s), <br>{us.square_bullet_point} and Segment C (Age Group < 30s). <br><br>By analysing these segments, Chisamba Marketing Co. can gain insights into their purchasing behaviours and loyalty, which will help Chisamba Marketing Co.  tailor its marketing strategies effectively. Special emphasis was put on categories: 1. Average Age, 2. Purchase Frequency, 3. Order Value and 4. Loyalty Score. <br>{us.square_bullet_point} <b>Segment A</b> has a moderate purchase frequency and high order value, indicating they are willing to spend more per purchase. Their loyalty score is also high, suggesting they are relatively loyal customers. <br>{us.square_bullet_point} <b>Segment B</b> has the highest loyalty score but the lowest purchase frequency. They spend moderately per order, indicating they value the brand but may need more incentives to purchase more frequently. <br>{us.square_bullet_point} <b>Segment C</b> has the highest purchase frequency but the lowest order value and loyalty score. They are frequent buyers but spend less per purchase and are less loyal compared to other segments. <br>By understanding the unique characteristics and behaviours of each customer segment, Chisamba Marketing Co.  can tailor its marketing strategies to better meet the needs of its customers. This targeted approach will help in improving customer satisfaction, increasing sales, and fostering long-term loyalty.",
        "Theory 3: Regression Analysis.": f"Chisamba Marketing Co. conducted a regression analysis to understand the relationship between their advertising expenditure and sales revenue. <br>{us.square_bullet_point} By analysing this data, they aim to optimize their marketing budget and maximize sales. <br>{us.square_bullet_point} Chisamba Marketing Co.’s regression analysis reveals a strong positive correlation between advertising expenditure and sales revenue. <br>{us.square_bullet_point} The analysis shows that every dollar spent on advertising yields an additional $2.50 in sales revenue. <br>This insight is crucial for Chisamba as it highlights the effectiveness of their advertising strategies. However, the analysis reveals outliers suggesting that while the overall strategy is effective, there are instances where spending significantly more on advertising leads to disproportionately higher sales. This could be due to: <br>{us.square_bullet_point} seasonal promotions, <br>{us.square_bullet_point} special campaigns, or other external factors. <br>By understanding and acting on these insights, Chisamba Marketing Co. can optimize their marketing strategies, improve sales, and achieve better overall business performance.",
        "Theory 4: Time Series Analysis.": f"Chisamba Marketing Co.’s Time Series analysis of historical demand data reveals a clear upward trend and seasonal fluctuations. <br>{us.square_bullet_point} By leveraging this analysis, Chisamba Marketing Co. can forecast future demand with greater accuracy. For instance, the forecast in RED indicates a continued increase in demand over the next month, with expected peaks and troughs due to seasonal factors. To optimize their procurement strategy, Chisamba can align Inventory with Demand by anticipating periods of high demand. Chisamba Marketing Co. can therefore ensure sufficient stock levels, reducing the risk of stockouts and lost sales. <br {us.square_bullet_point} By understanding and acting on these insights, Chisamba Marketing Co.:<br>{us.square_bullet_point} can enhance its procurement strategy, <br>{us.square_bullet_point} reduce costs, <br>{us.square_bullet_point} and improve overall efficiency, ultimately driving business growth. <br><br>In conclusion, Time series analysis provides Chisamba Marketing Co. with valuable insights into historical demand patterns and future forecasts. By leveraging these insights, Chisamba can make data-driven decisions to optimize their procurement strategy, reduce costs, and improve operational efficiency.",
        #"Theory 5: Market Basket Analysis.": "The project helps Chisamba Marketing and E-commerce Company optimize their product placement, cross-selling strategies, and marketing campaigns by understanding customer purchasing patterns.",
        "Theory 5: Customer Lifetime Value.": f"Customer Lifetime Value (CLV) is a crucial metric that helps businesses understand the total revenue a customer is expected to generate over their relationship with the company. By identifying high-value customers, Chisamba Marketing Co. could tailor their strategies to enhance customer retention and maximize profitability. They could conducted a Customer Lifetime Value (CLV) analysis to identify their most valuable customers. <br {us.square_bullet_point} The analysis revealed that the top 25% of customers, classified as high-value customers, contribute significantly more to the company’s revenue compared to the rest. <br>{us.square_bullet_point} By focusing on these high-value customers, Chisamba could implement targeted strategies to enhance their experience and loyalty. For instance, personalized marketing campaigns, exclusive offers, and premium customer service can be tailored to meet the needs of these customers. <br>{us.square_bullet_point} By understanding their purchasing behaviour and preferences, Chisamba Marketing Co. could proactively address any issues and ensure a positive customer experience. <br> Overall, the CLV analysis provides Chisamba Marketing Co. with valuable insights to optimize their customer retention strategies, reduce churn, and drive long-term profitability.",
        "Theory 6: Churn Prediction.":f"This project uses Random data generation. Therefore, different outcome on the chart should be expected. Using a RandomForestClassifier for a classification problem, Chisamba Marketing Co. could classify churn issues in their company. Analysing the above graph is crucial. <br>{us.square_bullet_point} Assuming the ROC curve shows an AUC of 0.80 and above, this indicates that the churn prediction model has a good ability to distinguish between customers who will churn and those who will not. <br>{us.square_bullet_point} The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the model. <br><br>The vital takeaways for Chisamba Marketing Co. is that their business, can deduce some actionable insights based on the ROC curve analysis like: <br>{us.square_bullet_point} (1.) <b>Model Effectiveness:</b> With an AUC of > 0.80, the model is quite effective in predicting customer churn. This means Chisamba Marketing Co. could rely on this model to identify at-risk customers with a high degree of confidence. <br>{us.square_bullet_point} (2.) <b>Targeted Interventions:</b> Chisamba Marketing Co. could develop targeted retention strategies such as personalized offers, loyalty programs, or improved customer service to retain these customers. <br>By leveraging these insights, Chisamba Marketing can proactively address customer churn, improve customer satisfaction, and ultimately enhance business performance.",
        "Theory 7: Sentiment Analysis.":f"Chisamba Marketing Co. conducted a sentiment analysis on 100 loyal customer reviews. The analysis revealed that: <br>{us.square_bullet_point} the majority of the reviews are positive, indicating that most customers are satisfied with their experience. <br>{us.square_bullet_point} However, there are also a significant number of negative reviews, highlighting areas that need improvement. By diving deeper into the negative reviews, Chisamba Marketing Co. identified two main issues: customer service and packaging quality. Customers expressed dissatisfaction with the responsiveness and helpfulness of the customer service team, as well as the quality and condition of the packaging upon delivery.<br><br> To address these issues, Chisamba Marketing could take the following measures: <br>{us.square_bullet_point} Improve Customer Service and <br>{us.square_bullet_point} Enhance Packaging Quality. <br>By prioritizing these improvements, Chisamba Marketing Co. could: <br {us.square_bullet_point} enhance customer satisfaction, <br>{us.square_bullet_point} reduce negative reviews, <br>{us.square_bullet_point} and foster a more positive brand image. <br>By addressing the issues highlighted in negative reviews, Chisamba Marketing can: <br>{us.square_bullet_point} improve their services, <br>{us.square_bullet_point} leading to higher customer satisfaction <br>{us.square_bullet_point} and loyalty.",
        "Theory 8: Conjoint Analysis.":f"By conducting a Conjoint Analysis, Chisamba could gain insights into: <br>{us.square_bullet_point} Customer preferences for different product attributes. <br>{us.square_bullet_point} The relative importance of each attribute. The optimal combination of product features. <br>{us.square_bullet_point} Pricing strategies that maximize customer satisfaction and company profit. <br>With this framework for conducting Conjoint Analysis and visualizing the results, Chisamba Marketing Co. could be able to leverage more business advantages. <br> By conducting a Conjoint Analysis, Chisamba Marketing Co. could be able to gain insights into: <br>{us.square_bullet_point} Customer preferences for different product attributes, <br>{us.square_bullet_point} the relative importance of each attribute, <br>{us.square_bullet_point} the optimal combination of product features or <br>{us.square_bullet_point} pricing strategies that maximize customer satisfaction and company profit.",
        "Theory 9: Multivariate Analysis.":f"Chisamba Marketing Co. conducted a multivariate analysis using Principal Component Analysis (PCA) to understand the relationships between key business metrics: <br>{us.square_bullet_point} marketing spend, <br>{us.square_bullet_point} sales, <br>{us.square_bullet_point} customer satisfaction, <br>{us.square_bullet_point} and website visits. <br> The PCA reduced the data to two principal components that explain the most variance in the dataset.The scatter plot of the principal components reveals distinct patterns: <br>{us.square_bullet_point} Principal Component 1 (PC1): This component captures the overall trend in the data, showing a strong positive correlation between marketing spend, sales, and website visits. As marketing spend increases, both sales and website visits also increase, indicating effective marketing strategies. <br>{us.square_bullet_point} Principal Component 2 (PC2): This component highlights variations in customer satisfaction. While customer satisfaction generally increases with higher marketing spend and sales, there are slight deviations that suggest other factors might influence customer satisfaction. <br> By understanding these relationships, Chisamba Marketing Co. could optimize their strategies:<br>{us.square_bullet_point} increase Marketing Spend or enhance Customer Satisfaction. <br>Multivariate analysis using PCA could help Chisamba Marketing Co. to identify key components that explain the most variance in their business metrics. By leveraging these insights, they can optimize their strategies to drive growth and enhance customer satisfaction.",
       # "Theory 11: Factor Analysis.":"The scatter plot visualizes the two main factors derived from the dataset. Each point represents a customer, colored by their satisfaction level.Insights: (1) Factor 1 might represent overall engagement, combining variables like website visits and purchase frequency. (2)Factor 2 could indicate marketing effectiveness, influenced by ad clicks and product reviews. (3) Customers custered in the top-right quadrant are highly engaged and satisfied, suggesting successful marketing strategies. By understanding these factors, Chisamba can tailor their marketing efforts to target specific customer segments more effectively.",
        "Theory 10: Predictive Modeling.":f"The scatter plot shows the relationship between the actual and predicted purchase amounts. Each point represents a customer, with the x-axis showing the actual purchase amount and the y-axis showing the predicted purchase amount.<br> The Key Insights are: <br>{us.square_bullet_point} Points close to the diagonal line indicate accurate predictions. <br>{us.square_bullet_point} Points far from the diagonal line indicate discrepancies between actual and predicted values. <br>This visualization helps Chisamba Marketing Co. to identify how well the model is performing and where it might need improvement. By analysing this graph, Chisamba can better understand:<br>{us.square_bullet_point} their customer purchasing behaviour <br>{us.square_bullet_point} and refine their marketing strategies <br>to target customers more effectively.",
    }
    return explanations.get(theory, "Explanation not available.")

def main():
    # Main window to display the chart and explanation
    custom_title('Marketing Statistical Theories.','orange',32,'bold')
    custom_title(f'{selected_theory}','red',18,'bold')   

    if "1:" in selected_theory:          
        # Display the result in the Streamlit app
        col1, col2 = st.columns(2)
        with col1:
                (ps.plot_conversion()) 
        with col2:                
                (ps.conversion_funnel_chart()) 
        st.divider()
        st.write("Data Story-Telling:")  
        custom_text_main(explain_theory(selected_theory),'orange',14,'italic','justified')       
    elif "2:" in selected_theory:
        (ps.cluster_segment_radar_chart()) 
        st.divider()
        st.write("Data Story-Telling:")                
        custom_text_main(explain_theory(selected_theory),'orange',14,'italic','justified')  
    elif "3:" in selected_theory:
        (ps.plot_regression()) 
        st.divider()
        st.write("Data Story-Telling:")                
        custom_text_main(explain_theory(selected_theory),'orange',14,'italic','justified')    
    elif "4:" in selected_theory:
        (ps.plot_times_series_data()) 
        st.divider()
        st.write("Data Story-Telling:")                
        custom_text_main(explain_theory(selected_theory),'orange',14,'italic','justified')    
    elif "5:" in selected_theory:
        (ps.plot_customer_lifetime_value())
        st.divider()    
        st.write("Data Story-Telling:")                   
        custom_text_main(explain_theory(selected_theory),'orange',14,'italic','justified')  
    elif "6:" in selected_theory:
        (ps.plot_churn_prediction()) 
        st.divider()
        st.write("Data Story-Telling:")                
        custom_text_main(explain_theory(selected_theory),'orange',14,'italic','justified')  
    elif "7:" in selected_theory:
        (ps.plot_sentiments()) 
        st.divider()
        st.write("Data Story-Telling:")                
        custom_text_main(explain_theory(selected_theory),'orange',14,'italic','justified')  
    elif "8:" in selected_theory:
        (ps.plot_conjoint())         
        st.divider()   
        st.write("Data Story-Telling:")                
        custom_text_main(explain_theory(selected_theory),'orange',14,'italic','justified')  
    elif "9:" in selected_theory:
        (ps.plot_multi_variate()) 
        st.divider()   
        st.write("Data Story-Telling:")                
        custom_text_main(explain_theory(selected_theory),'orange',14,'italic','justified')       
    elif "10:" and "Predictive Modeling." in selected_theory:
        (ps.plot_prediction())
        st.divider()    
        st.write("Data Story-Telling:")                
        custom_text_main(explain_theory(selected_theory),'orange',14,'italic','justified')  
    else:
        st.pyplot(generate_chart(selected_theory))
        st.write(explain_theory(selected_theory))

if __name__ == "__main__":
        main()


