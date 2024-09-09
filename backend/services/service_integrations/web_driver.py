import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager
import logging
import time
import re
from datetime import datetime
import pytz
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from langdetect import detect
from deep_translator import GoogleTranslator
import groq
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import random

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Groq clients for five agents
api_keys = [
    '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',
    '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',
    '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',
    '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',
    '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'
]
clients = [groq.Client(api_key=key) for key in api_keys]

# Google CSE API configuration
GOOGLE_API_KEY = "AIzaSyBf719x1yQKQKmy1v0yuFOsXWWpG2vPf7c"
GOOGLE_CSE_ID = "3707af7df34764449"

# Initialize translator
translator = GoogleTranslator(source='auto', target='en')

def get_random_client():
    return random.choice(clients)

def detect_intent(query):
    query = query.lower()
    if any(keyword in query for keyword in ["time", "baje", "clock", "time kya hai", "current time", "time now", "what time is it", "time in india",
        "kitna baje", "time kya hai", "abhi kitna baje", "abhi time kya hai",
        "india time", "bharat ka time", "time in india right now", "time in india now",
        "what is the time", "current hour", "what's the time in india",
        "time right now", "what is the time in india", "current time in india"]):
        return "time"
    elif any(keyword in query for keyword in ["date", "aaj ki date", "what's the date","current date", "today's date", "what's the date", "date today",
        "aaj ki date", "date kya hai", "aaj ki tarikh", "tarikh",
        "kitni tarikh", "date in india", "today date in india",
        "what is today's date", "what's the date today", "current date in india",
        "today's date in india"]):
        return "date"
    elif any(keyword in query for keyword in ["day", "din", "what's the day","day", "what day is it", "today's day", "day today", "aaj ka din",
        "day kya hai", "din", "aaj kya din hai", "aaj ka din kya hai",
        "which day is it", "current day", "what day of the week is it",
        "what day is today", "today is", "current day in india"]):
        return "day"
    elif any(keyword in query for keyword in ["time difference", "difference in time", "time between", "time gap",
        "time difference between", "time difference with", "difference of time with",
        "time difference india and", "time difference between india and",
        "difference in time between india and", "time gap between india and",
        "how much time difference between india and" ]):
        return "time_difference"
    else:
        return "search"

def get_search_results(query, num_results=5):
    try:
        url = f"https://www.googleapis.com/customsearch/v1"
        params = {
            'key': GOOGLE_API_KEY,
            'cx': GOOGLE_CSE_ID,
            'q': query,
            'num': num_results
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        search_results = response.json().get('items', [])
        return [item['link'] for item in search_results]
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error in get_search_results: {str(e)}")
        return []
    except Exception as e:
        logging.error(f"Error in get_search_results: {str(e)}")
        return []

def extract_content(url):
    try:
        options = FirefoxOptions()
        options.add_argument('--headless')
        driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()), options=options)
        driver.set_page_load_timeout(15)
        driver.get(url)
        time.sleep(2)
        page_source = driver.page_source
        driver.quit()

        soup = BeautifulSoup(page_source, 'html.parser')
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        content = soup.get_text(separator=' ', strip=True)
        content = re.sub(r'\s+', ' ', content)
        return content
    except Exception as e:
        logging.error(f"Error in extract_content for {url}: {str(e)}")
        return ""

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10),
       retry=retry_if_exception_type(groq.RateLimitError))
def analyze_and_summarize(content, query, client):
    try:
        chat_history = [
            {"role": "system", "content": "Summarize the following content related to the query."},
            {"role": "user", "content": f"Query: {query}\nContent: {content}"}
        ]

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=chat_history,
            max_tokens=300,
            temperature=1.0
        )

        summary = response.choices[0].message.content
        time.sleep(2)  # Add a 2-second delay after each API call
        return summary
    except groq.RateLimitError as e:
        logging.warning(f"Rate limit exceeded. Retrying with exponential backoff.")
        raise  # This will trigger the retry decorator
    except Exception as e:
        logging.error(f"Error in analyze_and_summarize: {str(e)}")
        return ""

def get_current_time_india():
    india_tz = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(india_tz)
    return f"The current time in India is {current_time.strftime('%I:%M %p')} on {current_time.strftime('%B %d, %Y')}."

def get_current_date():
    india_tz = pytz.timezone('Asia/Kolkata')
    current_date = datetime.now(india_tz).strftime('%B %d, %Y')
    return f"Today's date is {current_date}."

def get_current_day():
    india_tz = pytz.timezone('Asia/Kolkata')
    current_day = datetime.now(india_tz).strftime('%A')
    return f"Today is {current_day}."

def get_time_difference(place):
    try:
        india_tz = pytz.timezone('Asia/Kolkata')
        india_time = datetime.now(india_tz)
        target_tz = pytz.timezone(place)
        target_time = india_time.astimezone(target_tz)
        time_diff = target_time - india_time
        return f"The time difference between India and {place} is {time_diff}."
    except Exception as e:
        logging.error(f"Error in get_time_difference: {str(e)}")
        return f"Could not determine the time difference with {place}."

def refine_query(query):
    try:
        tokens = word_tokenize(query)
        stop_words = set(stopwords.words('english'))
        filtered_query = [word for word in tokens if word.lower() not in stop_words]
        refined_query = " ".join(filtered_query)
        return refined_query
    except Exception as e:
        logging.error(f"Error in refine_query: {str(e)}")
        return query

def process_url(url, refined_query):
    content = extract_content(url)
    if content:
        summary = analyze_and_summarize(content, refined_query, get_random_client())
        return summary
    return None

def handle_general_query(query):
    query_lower = query.lower()

    if any(phrase in query_lower for phrase in [
        "current time", "time now", "what time is it", "time in india",
        "kitna baje", "time kya hai", "abhi kitna baje", "abhi time kya hai",
        "india time", "bharat ka time", "time in india right now", "time in india now",
        "what is the time", "current hour", "what's the time in india",
        "time right now", "what is the time in india", "current time in india"
    ]):
        return get_current_time_india()

    elif any(phrase in query_lower for phrase in [
        "current date", "today's date", "what's the date", "date today",
        "aaj ki date", "date kya hai", "aaj ki tarikh", "tarikh",
        "kitni tarikh", "date in india", "today date in india",
        "what is today's date", "what's the date today", "current date in india",
        "today's date in india"
    ]):
        return get_current_date()

    elif any(phrase in query_lower for phrase in [
        "day", "what day is it", "today's day", "day today", "aaj ka din",
        "day kya hai", "din", "aaj kya din hai", "aaj ka din kya hai",
        "which day is it", "current day", "what day of the week is it",
        "what day is today", "today is", "current day in india"
    ]):
        return get_current_day()

    elif any(phrase in query_lower for phrase in [
        "time difference", "difference in time", "time between", "time gap",
        "time difference between", "time difference with", "difference of time with",
        "time difference india and", "time difference between india and",
        "difference in time between india and", "time gap between india and",
        "how much time difference between india and"
    ]):
        return get_time_difference(query_lower.split("with")[-1].strip())

    else:
        lang = detect(query)
        if lang != 'en':
            query = translator.translate(query)

        refined_query = refine_query(query)
        search_results = get_search_results(refined_query)

        summaries = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(process_url, url, refined_query): url for url in search_results}
            for future in as_completed(future_to_url):
                try:
                    summary = future.result()
                    if summary:
                        summaries.append(summary)
                except Exception as e:
                    logging.error(f"Error in processing URL: {str(e)}")

        return " ".join(summaries) if summaries else "No results found."

def process_query(query):
    intent = detect_intent(query)
    if intent == "time":
        return get_current_time_india()
    elif intent == "date":
        return get_current_date()
    elif intent == "day":
        return get_current_day()
    elif intent == "time_difference":
        return get_time_difference(query)
    else:
        return handle_general_query(query)

def main():
    user_query = input("Ask me something: ")
    response = process_query(user_query)
    print(response)

if __name__ == "__main__":
    main()