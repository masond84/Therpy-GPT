import random 
import pandas as pd
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from textblob import TextBlob  # for sentiment analysis
from collections import Counter
import openpyxl
import os
import time
from datetime import datetime
import geocoder
import pyttsx3
import threading

# Setup API Keys
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY
os.environ["OPENAI_API_KEY"] = OPANAI_API_KEY

# Create an instance of the OpenAI model
llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# user location
g = geocoder.ip('me')

# Personal Agent Class
class PersonalAgent:
    """
    Create an AI agent that will be used in the users terminal/command prompt.
    The agents goal is to build a real-time, constantly evolving and changing relationship with the user.
    The agents responses will be based off of memory (which is saved in an excel file and constantly refrenced before each response)
    The agent will use custom methods to allow for a personalized personality that adapts to the user and their conversation/history
    The agent will have emotions that change due to the conversation and past data
    The agent will have a consciusness that will allow the agent to engage in conversations more personably
    The agents feelings towards the user will change over time based off of how the user talks to the agent and how the agent feels.
    """

    # Initialize the bot
    def __init__(self, username=None):
        # the name of the user interacting with the bot
        self.username = username
        self.meta_info_sheet = "MetaInfo"
        # a dataframe to keep track of conversation, emotions, relationship status, and topics    
        self.memory = pd.DataFrame(columns=["User", "Bot", "Emotion", "Relationship", "Topic"])
        # a list of possible emotional states the bot can have
        self.emotions = ["happy", "sad", "excited", "calm", "confused", "angry", "curious"]
        # the bots current emotional state
        self.current_emotion = None
        # the bots relationship status with the user
        self.relationship = "Neutral"
        # filename where conversation history is stored
        self.conversation_data_file = "conversation_history.xlsx"  # Define the file to store the conversation data
        # user location
        self.user_city = g.city

    # Type the output/print statements to console
    def typing_effect(self, text, delay=0.03):
        for char in text:
            print(char, end='', flush=True)
            time.sleep(delay)
        print()

    # Add text-to-speech functionality
    def speak(self, text):
        # Uncomment the next lines to enable text-to-speech
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        

    # Update the username
    def update_username(self, new_name):
        self.username = new_name
        self.save_conversation_data()

    
    def get_time_of_day(self):
        current_time = datetime.now().time()
        hour = current_time.hour
        if 0 <= hour < 5:
            return "Night Time"
        elif 5 <= hour < 7:
            return "Waking Up"
        elif 7 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 18:
            return "Afternoon"
        elif 18 <= hour < 22:
            return "Evening"
        else:
            return "Night Time"
    
    def save_conversation_data(self):
        '''
        Saves the DataFrame to an Excel file. If the file already exists, it appends the new data.
        '''
        # Read existing data if any
        if os.path.exists(self.conversation_data_file):
            existing_data = pd.read_excel(self.conversation_data_file)
            combined_data = pd.concat([existing_data, self.memory])
        else:
            combined_data = self.memory

        # Save the combined data to an excel file
        combined_data.to_excel(self.conversation_data_file, index=False)

        meta_info = pd.DataFrame([{'Username': self.username}])
        with pd.ExcelWriter(self.conversation_data_file, engine="openpyxl", mode="a") as writer:
            meta_info.to_excel(writer, sheet_name=self.meta_info_sheet, index=False)

    
    def load_conversation_data(self):
        '''
        Loads existing conversation data from the Excel file, if any.
        '''
        # Load existing data if any
        if os.path.exists(self.conversation_data_file):
            self.memory = pd.read_excel(self.conversation_data_file)
            if not self.memory.empty:
                self.current_emotion = self.memory["Emotion"].iloc[-1]  # Load last emotional state
                self.relationship = self.memory["Relationship"].iloc[-1]  # Load last relationship status

            try:
                meta_info = pd.read_excel(self.conversation_data_file, sheet_name=self.meta_info_sheet)
                self.username = meta_info.at[0, 'Username']
                print(f"Loaded username: {self.username}")
            except Exception as e:
                print(f"Could not read metainfo: {e}")
    
    # Start the conversation - bot types first
    def start_conversation(self):
        '''
        Load existing conversation data and starts a new conversation with a greeting generated by the generate_random_greeting method.
        '''
        self.load_conversation_data()

        if not self.username:
            self.typing_effect("Nice to meet you, what is your name?")
            self.username = input()
            print(f"Set username to: {self.username}")
            self.save_conversation_data()

        # Generate a random greeting using OpenAI's GPT-3 LLM
        greeting = self.generate_random_greeting()
        self.speak(greeting)
        return greeting
    
    # Generate a random greeting - (Start Only)
    def generate_random_greeting(self):
        '''
        Uses the OpenAI GPT-3 model to generate a greeting.
        '''
        # Get the time of day
        time_of_day = self.get_time_of_day()
        # If past data exists
        if not self.memory.empty:
            # Talking To The User Again
            last_emotion = self.memory["Emotion"].iloc[-1]
            print(f"Bots Emotion: {last_emotion}")
            
            conversation_history = ""
            for index, row in self.memory.iterrows():
                user_chat = row["User"]
                bot_chat = row["Bot"]
                conversation_history += f"User: {user_chat}\nBot (YOU): {bot_chat}\n"

            # For topics, you can implement some logic to identify recurring topics from past conversations
            prompt = f"Given that the last time we talked, the emotional state was {last_emotion}, " \
                f"greet the user in a new and unique way." \
                f"The users name is {self.username}." \
                f"The time of day is {time_of_day}"
            prompt += f"This is past conversations and data you have with the user:\n{conversation_history}"
        else:
            # First Time Meeting User
            # Create a random list of phrases, questions, or compliments
            random_phrases = [
                "Ask a question to get to know the user.",
                "Tell the user something that happend in your day today.",
                "Ask the user what their day was like.",
            ]

            prompt = f"You are a personal AI companion created to provide comfort, socialization, and advice to your user."\
                    f"You are assigned to the user who's name is {self.username}." \
                    f"It's the {time_of_day}." \
                    f"{random.choice(random_phrases)}" \
                    f"Greet the user in a unique and engaging way."

        response = llm.generate(["In one sentence, respond to the users input: "+prompt], max_tokens=50)  # Adjust max_tokens as needed
        
        # Check if the list is not empty and then access its first element
        if response.generations and len(response.generations[0]) > 0:
            # Assuming the text attribute exists in the first element
            return response.generations[0][0].text.strip()
        else:
            return None
        
    def generate_response(self, user_input):
        '''
        This is the main function that produces the bot's overall response. It:
            Analyzes the sentiment of the user input.
            Maps the sentiment to an emotion.
            Saves the user input and emotion to the DataFrame.
            Calls generate_response_with_llm to get the bot's response.
            Saves the bot's response, relationship, and topics to the DataFrame.
            Saves the updated DataFrame to the Excel file.
        '''
        # Update name if applicable
        name_keywords = ["my name is", "call me", "name is", "i am now", "I am called"]
        for keyword in name_keywords:
            if keyword in user_input.lower():
                new_name = user_input.split(keyword)[-1].strip()
                self.update_username(new_name)
                print(f"Updated username to {new_name}")
        
        # Analyze sentiment of user input
        sentiment = self.analyze_sentiment(user_input)

        # Simulate emotional responses (replace this with actual emotion analysis)
        self.current_emotion = self.map_sentiment_to_emotion(sentiment)
        
        # Store the conversation in memory
        new_row = {"User": user_input, "Bot": "", "Emotion": self.current_emotion}
        self.memory = pd.concat([self.memory, pd.DataFrame([new_row])], ignore_index=True)
        
        # Respond based on user input and current emotion
        response = self.generate_response_with_llm(user_input)
        self.speak(response)  # Bot will speak the response

        # Store bot's response in memory
        self.memory.at[len(self.memory) - 1, "Bot"] = response
        # Add a relationship and topic tracking
        self.memory.at[len(self.memory) - 1, "Relationship"] = self.relationship
        topics = self.identify_topics()
        self.memory.at[len(self.memory) - 1, "Topic"] = ", ".join(topics)

        # Save the conversation data to the data file
        self.save_conversation_data()

        return response
    
    def _generate_response_based_on_emotion(self, user_input):
        '''
        Generates a response based on the bot's current emotional state, which is determined by the sentiment of the user's input.
        '''
        sentiment = self.analyze_sentiment(user_input)
        if sentiment in ["very positive", "positive"]:
            emotions = ["happy", "excited"]
        elif sentiment in ["very negative", "negative"]:
            emotions = ["sad", "angry"]
        else:
            emotions = ["neutral", "calm", "curious"]
        selected_emotion = random.choice(emotions)
        return selected_emotion  # You can map this to a corresponding response
  

    def get_memory(self):
        return self.memory

    def identify_topics(self):
        '''
        Placeholder for a method that would identify topics in the conversation.
        '''
        # Combine all the user's past inputs into a single string
        all_user_inputs = " ".join(self.memory["User"].dropna())

        # Tokenize the string into individual words
        words = all_user_inputs.split()

        # Use a counter to find the most common words
        word_freq = Counter(words)

        # Get the 3 most common words as topics (you can choose more or fewer as you like)
        most_common_words = word_freq.most_common(3)

        # Extract just the words from the list of tuples ('word', frequency)
        topics = [word[0] for word in most_common_words]

        return topics
   
    def generate_response_with_llm(self, user_input):
        '''
        Uses the OpenAI GPT-3 model to generate a response based on the current conversation history and the bot's emotional state.
        '''
        # Get the user;s conversation history from memory
        conversation_history = self.memory["User"].tolist()
        #print(conversation_history)
        conversation_history.append(f"User: {user_input}\nBot:")

        # Generate an emotional response based on user input and sentiment
        emotional_response = self._generate_response_based_on_emotion(user_input)

        # Combine the conversation history into a single string
        conversation_text = "\n".join([str(emotional_response)] + [str(x) for x in conversation_history])
    
        # Generate a response using OpenAI's GPT-3 LLM
        prompt = conversation_text

        # Incorporate current emotion, relationsip, and topics into prompt
        response = llm.generate([prompt], max_tokens=50)  # Adjust max_tokens as needed
        
        # Format the response
        if response.generations and len(response.generations[0]) > 0:
            # Assuming the text attribute exists in the first element
            return response.generations[0][0].text.strip()
        else:
            return "I couldn't generate a response."

    def analyze_sentiment(self, text):
        '''
        Uses TextBlob to analyze the sentiment of a given text and returns "positive," "negative," or "neutral."
        '''
        # Perform sentiment analysis on the user's input
        analysis = TextBlob(text)
        sentiment_score = analysis.sentiment.polarity
        
        # Map sentiment score to emotion
        if sentiment_score > 0.5:
            return "very positive"
        elif sentiment_score > 0.2:
            return "positive"
        elif sentiment_score > -0.5:
            return "very negative"
        elif sentiment_score < -0.2:
            return "negative"
        else:
            return "neutral"

    def map_sentiment_to_emotion(self, sentiment):
        '''
        Maps the sentiment ("positive," "negative," "neutral") to one of the predefined emotional states.
        '''
        # Map sentiment to predefined emotions (you can customize this mapping)
        if sentiment == "very positive":
            return "excited"
        elif sentiment == "positive":
            return "happy"
        elif sentiment == "very negative":
            return "angry"
        elif sentiment == "negative":
            return "sad"
        else:
            return random.choice(self.emotions) 
        
    def generate_and_append_code(self, user_input):
        # Step 1: Parse the user input
        if user_input.startswith("depict-coding"):
            phrase = user_input[len("depict-coding "):]

            # Step 2: Generate code
            generated_code = self.generate_code_with_llm(phrase)
            
            # Step 3: Verify code (Here, I just check for emptiness. You might want to add more checks.)
            if not generated_code:
                return "Couldn't generate any code."

            # Step 4: Write to file
            try:
                with open("C:\\Users\\dmaso\\Downloads\\ai-agent\\ai-agent\\agent-feeling.py", 'a') as f:
                    f.write("\n# Generated Code\n")
                    f.write(generated_code)
                return "Code has been appended successfully."
            except Exception as e:
                return f"An error occurred: {e}"

    # This method will generate Python code based on the user's phrase using GPT-3
    def generate_code_with_llm(self, phrase):
        code = llm.generate([f"Write a Python script that {phrase}. Debug and review the entire script before finalizing. Give me ONLY the script, no additional words, text or context."], max_tokens=100).generations[0][0].text.strip()
        print(code)
        return code
    
    def get_memory(self):
        '''
        Returns the current state of the DataFrame that holds the conversation history.
        '''
        return self.memory


###############
## MAIN CODE ##
###############
MAX_TOKENS = 4096

# Initialize both agents
therapist_agent = PersonalAgent()  # You can specify the user's name here
language_agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# start the conversation with having the bot greet the user
greeting = (therapist_agent.start_conversation())
therapist_agent.typing_effect(greeting)

''' Test Input - One Response'''
# user_input = input("You: ")
# therapist_agent.typing_effect(user_input)
# response = therapist_agent.generate_response(user_input)
# therapist_agent.typing_effect(f"{response}")

while True:
    user_input = input("You: ")
    print(user_input)
    if user_input.lower() == "bye":
        farewell_message = f"Goodbye! {therapist_agent.username}, Take care."
        therapist_agent.typing_effect(f"Bot: {farewell_message}")
        therapist_agent.speak(farewell_message)
        therapist_agent.save_conversation_data()  # Save data when the conversation ends
        break
    elif user_input.startswith("depict-coding"):
        result = therapist_agent.generate_and_append_code(user_input)
        therapist_agent.speak(result)
        therapist_agent.typing_effect(f"Bot: {result}")
    else:
        conversation_history = therapist_agent.get_memory()

        prompt_with_history = f"{user_input}"

        # Check token count
        #token_count = len(prompt_with_history.split())
        
        # Truncate if too long
        #if token_count > MAX_TOKENS:
        #    truncated_history = " ".join(prompt_with_history.split()[-MAX_TOKENS:])
        #else:
        #    truncated_history = prompt_with_history

        response = therapist_agent.generate_response(prompt_with_history)
        therapist_agent.typing_effect(f"Bot: {response}")
"""
# Access the conversation history
conversation_history = therapist_agent.get_memory()"""
