
# NLP_Sentiment_client_response
```python
import pandas as pd
import random
from nltk.sentiment import SentimentIntensityAnalyzer

# Generating imaginary data for 100 clients
client_data = {
    'Age': [random.randint(20, 60) for _ in range(100)],
    'Income': [random.randint(20000, 100000) for _ in range(100)],
    'Education': [random.choice(['High School', 'Bachelor', 'Master', 'PhD']) for _ in range(100)],
    'Gender': [random.choice(['Male', 'Female']) for _ in range(100)],
    'Country': [random.choice(['USA', 'UK', 'Canada', 'Australia']) for _ in range(100)],
}

# Generating fictional responses for the 'Response' column
responses = [
    "The product is amazing! I love it.",
    "Not satisfied with the service.",
    "It's okay, could be better.",
    "Great experience overall, highly recommend!",
    "I didn't like it much, needs improvement.",
    "Outstanding service and quality!",
    "Could have been better, not satisfied.",
    "Excellent customer support, very helpful.",
    "Poor quality, wouldn't recommend it.",
    "Satisfied with the product, good value for money.",
    "Disappointed with the service, won't come back."
]

# Filling the 'Response' column with fictional responses (repeated if necessary)
client_data['Response'] = random.choices(responses, k=100)

# Creating a DataFrame from the generated data
df = pd.DataFrame(client_data)

# Performing Sentiment Analysis using NLTK's SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Function to get sentiment polarity scores
def get_sentiment_score(text):
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']

# Applying sentiment analysis to the 'Response' column
df['Sentiment_Score'] = df['Response'].apply(get_sentiment_score)

# Classifying sentiment based on the sentiment score
df['Sentiment'] = df['Sentiment_Score'].apply(lambda score: 'Positive' if score > 0 else 'Neutral' if score == 0 else 'Negative')

# Displaying the resulting DataFrame with sentiment analysis
print(df.head(60))

```

        Age  Income    Education  Gender    Country  \
    0    49   83487  High School    Male        USA   
    1    39   89604  High School  Female        USA   
    2    47   25003          PhD  Female     Canada   
    3    55   20877     Bachelor  Female  Australia   
    4    54   83635       Master    Male        USA   
    5    57   30739          PhD    Male         UK   
    6    48   46103  High School    Male  Australia   
    7    28   64801     Bachelor  Female        USA   
    8    53   46399     Bachelor  Female        USA   
    9    21   63816     Bachelor  Female  Australia   
    10   59   99140       Master  Female         UK   
    11   30   62786  High School    Male        USA   
    12   59   93540       Master  Female        USA   
    13   52   97252     Bachelor  Female  Australia   
    14   20   90655       Master    Male  Australia   
    15   41   41395     Bachelor    Male     Canada   
    16   34   93141  High School  Female  Australia   
    17   42   45696       Master  Female        USA   
    18   21   51051          PhD    Male        USA   
    19   38   79450     Bachelor    Male  Australia   
    20   56   54059       Master  Female     Canada   
    21   60   86768          PhD  Female     Canada   
    22   55   37842       Master  Female        USA   
    23   40   91590       Master  Female         UK   
    24   53   88710  High School  Female         UK   
    25   50   25644  High School  Female  Australia   
    26   51   93564       Master    Male         UK   
    27   43   66043     Bachelor  Female  Australia   
    28   34   88397     Bachelor  Female     Canada   
    29   26   52551       Master    Male     Canada   
    30   42   20248  High School    Male     Canada   
    31   46   89205  High School  Female        USA   
    32   44   70151       Master  Female  Australia   
    33   33   45567          PhD    Male     Canada   
    34   49   75712  High School  Female  Australia   
    35   22   52284       Master    Male  Australia   
    36   20   94628  High School  Female     Canada   
    37   50   63700     Bachelor  Female        USA   
    38   36   24719     Bachelor    Male        USA   
    39   32   69781  High School    Male  Australia   
    40   22   24905     Bachelor  Female        USA   
    41   54   41074     Bachelor    Male  Australia   
    42   42   26003     Bachelor    Male         UK   
    43   31   22082     Bachelor    Male        USA   
    44   24   79740       Master  Female  Australia   
    45   41   21476          PhD  Female     Canada   
    46   41   45563          PhD  Female     Canada   
    47   58   20856          PhD    Male  Australia   
    48   29   43426     Bachelor    Male     Canada   
    49   43   99807     Bachelor  Female     Canada   
    50   30   96922       Master    Male        USA   
    51   26   32712  High School    Male        USA   
    52   32   45149  High School    Male         UK   
    53   27   32605     Bachelor    Male        USA   
    54   55   25851       Master  Female  Australia   
    55   21   33667          PhD    Male  Australia   
    56   27   94982     Bachelor  Female  Australia   
    57   56   92604     Bachelor  Female     Canada   
    58   28   35688          PhD    Male        USA   
    59   25   24533  High School    Male     Canada   
    
                                                 Response  Sentiment_Score  \
    0                     Not satisfied with the service.          -0.3252   
    1                Poor quality, wouldn't recommend it.          -0.6381   
    2     Disappointed with the service, won't come back.          -0.4767   
    3   Satisfied with the product, good value for money.           0.7964   
    4                  The product is amazing! I love it.           0.8516   
    5           I didn't like it much, needs improvement.           0.2240   
    6                  The product is amazing! I love it.           0.8516   
    7                         It's okay, could be better.           0.5859   
    8                     Not satisfied with the service.          -0.3252   
    9         Great experience overall, highly recommend!           0.8012   
    10          Excellent customer support, very helpful.           0.8588   
    11  Satisfied with the product, good value for money.           0.7964   
    12        Great experience overall, highly recommend!           0.8012   
    13                   Outstanding service and quality!           0.6476   
    14               Poor quality, wouldn't recommend it.          -0.6381   
    15             Could have been better, not satisfied.           0.1451   
    16    Disappointed with the service, won't come back.          -0.4767   
    17                    Not satisfied with the service.          -0.3252   
    18                   Outstanding service and quality!           0.6476   
    19          I didn't like it much, needs improvement.           0.2240   
    20                        It's okay, could be better.           0.5859   
    21                        It's okay, could be better.           0.5859   
    22          I didn't like it much, needs improvement.           0.2240   
    23        Great experience overall, highly recommend!           0.8012   
    24    Disappointed with the service, won't come back.          -0.4767   
    25             Could have been better, not satisfied.           0.1451   
    26                        It's okay, could be better.           0.5859   
    27    Disappointed with the service, won't come back.          -0.4767   
    28          I didn't like it much, needs improvement.           0.2240   
    29               Poor quality, wouldn't recommend it.          -0.6381   
    30                   Outstanding service and quality!           0.6476   
    31                    Not satisfied with the service.          -0.3252   
    32          Excellent customer support, very helpful.           0.8588   
    33                 The product is amazing! I love it.           0.8516   
    34                   Outstanding service and quality!           0.6476   
    35        Great experience overall, highly recommend!           0.8012   
    36        Great experience overall, highly recommend!           0.8012   
    37  Satisfied with the product, good value for money.           0.7964   
    38  Satisfied with the product, good value for money.           0.7964   
    39               Poor quality, wouldn't recommend it.          -0.6381   
    40             Could have been better, not satisfied.           0.1451   
    41                 The product is amazing! I love it.           0.8516   
    42          I didn't like it much, needs improvement.           0.2240   
    43  Satisfied with the product, good value for money.           0.7964   
    44        Great experience overall, highly recommend!           0.8012   
    45               Poor quality, wouldn't recommend it.          -0.6381   
    46          I didn't like it much, needs improvement.           0.2240   
    47               Poor quality, wouldn't recommend it.          -0.6381   
    48                        It's okay, could be better.           0.5859   
    49        Great experience overall, highly recommend!           0.8012   
    50    Disappointed with the service, won't come back.          -0.4767   
    51                   Outstanding service and quality!           0.6476   
    52  Satisfied with the product, good value for money.           0.7964   
    53        Great experience overall, highly recommend!           0.8012   
    54             Could have been better, not satisfied.           0.1451   
    55                 The product is amazing! I love it.           0.8516   
    56                    Not satisfied with the service.          -0.3252   
    57                 The product is amazing! I love it.           0.8516   
    58             Could have been better, not satisfied.           0.1451   
    59    Disappointed with the service, won't come back.          -0.4767   
    
       Sentiment  
    0   Negative  
    1   Negative  
    2   Negative  
    3   Positive  
    4   Positive  
    5   Positive  
    6   Positive  
    7   Positive  
    8   Negative  
    9   Positive  
    10  Positive  
    11  Positive  
    12  Positive  
    13  Positive  
    14  Negative  
    15  Positive  
    16  Negative  
    17  Negative  
    18  Positive  
    19  Positive  
    20  Positive  
    21  Positive  
    22  Positive  
    23  Positive  
    24  Negative  
    25  Positive  
    26  Positive  
    27  Negative  
    28  Positive  
    29  Negative  
    30  Positive  
    31  Negative  
    32  Positive  
    33  Positive  
    34  Positive  
    35  Positive  
    36  Positive  
    37  Positive  
    38  Positive  
    39  Negative  
    40  Positive  
    41  Positive  
    42  Positive  
    43  Positive  
    44  Positive  
    45  Negative  
    46  Positive  
    47  Negative  
    48  Positive  
    49  Positive  
    50  Negative  
    51  Positive  
    52  Positive  
    53  Positive  
    54  Positive  
    55  Positive  
    56  Negative  
    57  Positive  
    58  Positive  
    59  Negative  

