import tkinter as tk
from tkinter import ttk
from threading import Thread
import pyttsx3

class ChatGUI:
    def __init__(self, master):
        self.master = master
        self.master.title('Chatbot')
        
        self.text_widget = tk.Text(self.master, wrap='word', height=20, width=50)
        self.text_widget.pack(padx=20, pady=20)

        self.entry = ttk.Entry(self.master, width=40)
        self.entry.pack(padx=20, pady=5)
        self.entry.bind('<Return>', self.get_response)

        self.engine = pyttsx3.init()
        
    def get_response(self, event=None):
        user_input = self.entry.get()
        if user_input:
            self.append_text(f"You: {user_input}", 'blue')
            self.entry.delete(0, tk.END)
            # Replace the line below with code to generate bot's response
            bot_response = "Hello, this is your bot speaking."  
            Thread(target=self.speak_and_type, args=(bot_response,)).start()
            
    def append_text(self, text, color):
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.insert(tk.END, text + '\n')
        self.text_widget.tag_add(color, "end-2c", "end-1c")
        self.text_widget.tag_config(color, foreground=color)
        self.text_widget.config(state=tk.DISABLED)
        
    def speak_and_type(self, text):
        self.append_text(f"Bot: {text}", 'green')
        self.engine.say(text)
        self.engine.runAndWait()
        
root = tk.Tk()
app = ChatGUI(root)
root.mainloop()