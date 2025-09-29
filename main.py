import config
from google import genai
from google.genai import types

# Initialize API key
client = genai.Client(api_key=config.GEMINI_API_KEY)

def generate_response(prompt, temperature=0.3):
    try:
        contents = [types.Content(role="user", parts=types.Part.from_text(text=prompt))]
        config_params = types.GenerateContentConfig(temperature=temperature)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config=config_params
        )

        if response.candidates:
            return response.candidates[0].content.parts[0].text
        else:
            return "No response generated."
    except Exception as e:
        return f"Error: {str(e)}"

# Zero-shot learning
category = input("Enter a category i.e Animals, Plants, Science etc: ")
item = input(f"Enter a specific {category} to explain: ")

print("\n----------- Zero-Shot Learning ---------")
zero_shot = f"Is {item} a {category}? Yes or No."
print(f"Prompt: {zero_shot}")
print(f"Response: {generate_response(zero_shot)}")

# One-shot learning
print("\n----------- One-Shot Learning ----------")
one_shot = f"""Determine if the item belongs to the category.
Example:
Category: Fruit
Item: Apple
Answer: Yes, apple is a fruit.
Now you try:
Category: {category}
Item: {item}
Answer:"""
print(f"Response: {generate_response(one_shot)}")

# Few-shot learning
print("\n ----------- Few-Shot Learning --------")
few_shot = f"""Determine if item belongs to the category.
Example1:
Category: Vehicle
Item: Bus
Answer: Yes, bus is a vehicle
Example2:
Category: Fruit
Item: Carrot
Answer: No, carrot is not a fruit
Example3:
Category: Ocean
Item: Bat
Answer: No, bat is not in the ocean
Now you try:
Category: {category}
Item: {item}
Answer:"""
print(f"Response: {generate_response(few_shot)}")

# Creative response
print("\n------ CREATIVE RESPONSE ---------")
creative_prompt = f"""Write a one sentence story about a given word.
Example1:
Word: Moon
Story: The moon winked at the sun during the lunar eclipse.
Word: {item}
Story:"""
print(f"Response: {generate_response(creative_prompt, temperature=0.7)}")
