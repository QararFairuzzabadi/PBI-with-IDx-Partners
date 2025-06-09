# Application of Machine Learning for Credit Risk Prediction in Customer Loans

## Overview

### Introduction
<p style="text-align: justify; font-family: 'Georgia', cursive, sans-serif; line-height: 1.5;">
Hi, I'm M. Razy Qarar Fairuzzabadi, an Enthusiast Data Scientist who is doing his final project as a Data Scientist in ID/X Partners Project Based Internship program. This dataset consists of <b>466,285 rows</b> and <b>70 columns</b>. In this project, I will build an optimal model to predict the loan status of customers by using classification algorithms in supervised learning.
</p>

### Contents
| **Variables**                     | Description                                                                                                               |
|-----------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| **ID**                            | A unique LC assigned ID for the loan listing                                                                              |
| **Member ID**                     | A unique LC assigned Id for the borrower member                                                                           |
| **Loan Amnt**                     | Last month payment was received                                                                                           |
| **Funded Amnt**                   | The total amount committed to that loan at that point in time                                                             |
| **Funded Amnt Inv**               |                                                                                                                           |
| **Term**                          | The number of payments on the loan. Values are in months and can be either 36 or 60                                       |
| **Int Rate**                      | Indicates if income was verified by LC, not verified, or if the income source was verified                                |
| **Installment**                   | The monthly payment owed by the borrower if the loan originates                                                           |
| **Grade**                         | LC assigned loan grade                                                                                                    |
| **Sub Grade**                     | LC assigned loan subgrade                                                                                                 |
| **Emp Title**                     | The job title supplied by the Borrower when applying for the loan                                                         |
| **Emp Length**                    | Employment length in years. Possible values are between 0 and 10                                                          |
| **Home Ownership**                | The home ownership status provided by the borrower during registration                                                    |
| **Annual Inc**                    | The self-reported annual income provided by the borrower during registration                                              |
| **Verification Status**           |                                                                                                                           |
| **Issue d**                       | The month which the loan was funded                                                                                       |
| **Loan Status**                   | Current status of the loan                                                                                                |
| **Pymnt Plan**                    |                                                                                                                           |
| **Url**                           | URL for the LC page with listing data                                                                                     |
| **Desc**                          | Loan description provided by the borrower                                                                                 |
| **Purpose**                       | A category provided by the borrower for the loan request                                                                  | 
| **Title**                         | The loan title provided by the borrower                                                                                   |
| **Zip Code**                      | The first 3 numbers of the zip code provided by the borrower in the loan application                                      |
| **Addr State**                    | The state provided by the borrower in the loan application                                                                |
| **Dti**                           | A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations                         |
| **Delinq 2yrs**                   | The number of 30+ days past-due incidences of delinquency in the borrower's credit file                                   |
| **Earliest Cr Line**              | The month the borrower's earliest reported credit line was opened                                                         |
| **Inq Last 6mths**                | The number of inquiries in past 6 months (excluding auto and mortgage inquiries)                                          |
| **Mths Since Last Delinq**        | The number of months since the borrower's last delinquency                                                                |
| **Mths Since Last Record**        | The number of months since the last public record                                                                         |
| **Open Acc**                      | The number of open credit lines in the borrower's credit file                                                             |
| **Pub Rec**                       | Number of derogatory public records                                                                                       |
| **Revol Bal**                     | Total credit revolving balance                                                                                            |
| **Revol Util**                    | Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit |
| **Total Acc**                     | The total number of credit lines currently in the borrower's credit file                                                  |
| **Initial List Status**           | The initial listing status of the loan. Possible values are – Whole, Fractional                                           |
| **Out Prncp**                     | Remaining outstanding principal for total amount funded                                                                   |
| **Out Prncp Inv**                 | Remaining outstanding principal for portion of total amount funded by investors                                           |
| **Total Pymnt**                   | Payments received to date for total amount funded                                                                         |
| **Total Pymnt Inv**               | Payments received to date for portion of total amount funded by investors                                                 |
| **Total Rec Prncp**               | Principal received to date                                                                                                |
| **Total Rec Int**                 | Interest received to date                                                                                                 |
| **Total Rec Late Fee**            | Late fees received to date                                                                                                |
| **Recoveries**                    | Indicates if a payment plan has been put in place for the loan                                                            |
| **Collection Recovery Fee**       | Post charge off collection fee                                                                                            |
| **Last Pymnt d**                  | Last month payment was received                                                                                           |
| **Last Pymnt Amnt**               | Last total payment amount received                                                                                        |
| **Next Pymnt d**                  | Next scheduled payment date                                                                                               |
| **Last Credit Pull d**            |                                                                                                                           |
| **Collections 12 Mths Ex Med**    | Number of collections in 12 months excluding medical collections                                                          |
| **Mths Since Last Major Derog**   | Months since most recent 90-day or worse rating                                                                           |
| **Policy Code**                   | Publicly available policy_code = 1; New products not publicly available policy_code = 2                                   |
| **Application Type**              | Indicates whether the loan is an individual application or a joint application with two co-borrowers                      |
| **Annual Inc Joint**              | The combined self-reported annual income provided by the co-borrowers during registration                                 |
| **Dti Joint**                     | A ratio calculated using the co-borrowers' total monthly payments on the total debt obligations                           |
| **Verification Status Joint**     |                                                                                                                           |
| **Acc Now Delinq**                | The number of accounts on which the borrower is now delinquent                                                            |
| **Tot Coll Amt**                  | Total collection amounts ever owed                                                                                        |
| **Tot Cur Bal**                   | Total current balance of all accounts                                                                                     |
| **Open Acc 6m**                   | Number of open trades in last 6 months                                                                                    |
| **Open Il 12m**                   | Number of installment accounts opened in past 12 months                                                                   |
| **Open Il 24m**                   | Number of installment accounts opened in past 24 months                                                                   |
| **Mths Since Rcnt Il**            | Months since most recent installment accounts op                                                                          |

## Background & The Steps

### Background
<ul style="font-family: 'Georgia', serif; text-align: justify; line-height: 1.5;">
  <li> Credit risk greatly affects the stability of financial institutions, because if loans are not paid as agreed, financial institutions can suffer huge losses. Credit risk management   is therefore of paramount importance. </li>

  <li> Traditional approaches such as rule-based credit scores remain less flexible in handling the complexity of customer data and often struggle to detect patterns or signs of default risk. </li>

  <li> Machine learning can be an effective solution by utilizing large historical data. These algorithms can detect patterns in customer behavior, estimate default risk, and improve the accuracy of credit decisions. </li>

  <li> By building reliable predictive models, financial institutions can reduce credit risk and strengthen loan management systems. </li>
</ul>

### The Steps in this Project
<ul style="font-family: 'Georgia', serif; text-align: justify; line-height: 1.5;">
  <li> Data Collection. </li>

  <li> Data understanding by conducting an initial exploration of the distribution of variables, descriptive statistics, and general patterns present in the data. </li>

  <li> Data preprocessing which includes checking and handling duplicate and missing data, feature engineering, data encoding, data standardization, feature selection, and handling class imbalance. </li>

  <li> Supervised classification modeling. </li>

  <li> Evaluation of results (best model) using Accuracy and Precision metrics. </li>
</ul>

## Conclusion
<p style="text-align: justify; font-family: 'Georgia', cursive, sans-serif; line-height: 1.5;">
The most effective model for predicting loan status in this project is <b>Extreme Gradient Boosting (XGBoost)</b>. At the initial stage, the base model achieved <b>92.4% accuracy</b> and <b>93.4% precision</b>. After hyperparameter tuning, the accuracy increased to <b>92.6%</b>, while the precision remained <b>93.4%</b>.
</p>
