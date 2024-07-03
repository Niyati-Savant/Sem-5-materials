import pandas as pd
import math

data = pd.read_csv('naive_bayes_play.csv')

Outlook = input("Enter 'Outlook ' (Sunny, Overcast, Rainy): ")
Temperature = input("Enter 'Temperature' (Hot, Mild, Cool): ")
Humidity = input("Enter 'Humidity' (Normal, High): ")
Windy = bool(input("Enter 'Windy' (True, False): "))

print("\n")

# Prior probabilities
total_instances = len(data)
play_yes = len(data[data['Play'] == 'Yes'])
play_no = len(data[data['Play'] == 'No'])

p_yes =round(play_yes / total_instances, 4)
p_no = round(play_no / total_instances, 4)

print(f"Total occurances of Yes: {play_yes}")
print(f"Total occurances of No: {play_no}")

print(f"Prior probability for Yes {p_yes}")
print(f"Prior probability for No: {p_no}")

# Likelihood probabilities
def calculate_likelihood(attribute, value, play):
    subset = data[data['Play'] == play]
    print(subset[subset[attribute] == value])
    count = len(subset[subset[attribute] == value])
    total = len(subset)
    return count / total


print("\n")
print("\n")
print("The likelihood probabilities are: ")

p_play_given_yes = round(calculate_likelihood('Outlook', Outlook, 'Yes'),4)
p_play_given_no = round(calculate_likelihood('Outlook', Outlook, 'No'),4)

print(f"P(Outlook = {Outlook} / buy='Yes') = {p_play_given_yes}")
print(f"P(Outlook = {Outlook} / buy='No') = {p_play_given_no}")
print("\n")

p_temp_given_yes = round(calculate_likelihood('Temperature', Temperature, 'Yes'),4)
p_temp_given_no = round(calculate_likelihood('Temperature', Temperature, 'No'),4)
print(f"P(Temperature = {Temperature} / buy='Yes') = {p_temp_given_yes}")
print(f"P(Temperature = {Temperature} / buy='No') = {p_temp_given_no}")
print("\n")

p_hum_given_yes =round(calculate_likelihood('Humidity', Humidity, 'Yes'),4)
p_hum_given_no =round(calculate_likelihood('Humidity', Humidity, 'No'),4)
print(f"P(Humidity = {Humidity} / buy='Yes') = {p_hum_given_yes}")
print(f"P(Humidity = {Humidity} / buy='No') = {p_hum_given_no}")
print("\n")


p_windy_given_yes = round(calculate_likelihood('Windy', Windy, 'Yes'),4)
p_windy_given_no =round(calculate_likelihood('Windy', Windy, 'No'),4)
print(f"P(Windy = {Windy} / buy='Yes') = {p_windy_given_yes}")
print(f"P(Windy = {Windy}  / buy='No') = {p_windy_given_no}")
print("\n")

# Posterior probabilities
print("The posterior probabilities are: ")

p_yes_given_x = round(p_yes * p_play_given_yes * p_temp_given_yes * p_hum_given_yes * p_windy_given_yes,4)
p_no_given_x = round(p_no * p_play_given_no * p_temp_given_no  * p_hum_given_no * p_windy_given_no,4)

if p_yes_given_x > p_no_given_x:
    prediction = 'Yes'
else:
    prediction = 'No'


print(f'Probability of Purchased=Yes: {p_yes_given_x}')
print(f'Probability of Purchased=No: {p_no_given_x:}')
print(f'Prediction: {prediction}')
