#!/usr/bin/env python3

import requests
import json
import random

# FIXME: Implement logic to ensure two words are not used twice!!! - think of algorithm that efficiently takes care of this
# FIXME: Loop through each word option until a valid result is found for haiku

def generate_init_word(topic, num_syllables):
    """Generates word one for each line of the haiku given the base_url for API and topic for haiku as arguments.
    Args:
        base_url (str): base url for Datamuse API 
        topic (str): topic for the haiku
        num_syllables (int): number of syllables

    Returns:
        list: array of words given the topic
    """
    base_url = "https://api.datamuse.com"

    end_point = f'/words?md=s&rel_trg={topic}'

    muse_api_full_url = f'{base_url}{end_point}'
    
    # print(muse_api_full_url)

    response = requests.get(muse_api_full_url)

    if response:
        data = json.loads(response.text)
        
        word_list = []
        
        for line in data:
            if line["numSyllables"] == num_syllables and line != '.':
                entry = {
                    "word" : line["word"],
                    "numSyllables" : line["numSyllables"] 
                }
                word_list.append(entry)
 
        # print(json.dumps(word_list, indent=4))
        return word_list
    else:
        print("Connection error. Request failed")
        print(f'Status code: {response.status_code}')
        
def generate_word_left_context(previous_word, num_syllables):
    """Generates word two for each line of the haiku given the base_url and previous word of haiku as arguments.

    Args:
        base_url (str): base url for Datamuse API 
        previous_word (str): previous word to given following context

    Returns:
        list: array of words that best follow the previous word
    """
    
    base_url = "https://api.datamuse.com"    
    end_point = f'/words?md=s&sp=*&lc={previous_word}'

    muse_api_full_url = f'{base_url}{end_point}'
    print(muse_api_full_url)

    response = requests.get(muse_api_full_url)

    if response:
        data = json.loads(response.text)
        words_list = []
        for line in data:
            if line["numSyllables"] == num_syllables and line != '.':
                entry = {
                    "word" : line["word"],
                    "numSyllables" : line["numSyllables"] 
                }
                words_list.append(entry)
        # print(json.dumps(words_list, indent=4))
        return words_list
    else:
        print("Connection error. Request failed")
        print(f'Status code: {response.status_code}')
        
def generate_word_left_context_rhyme(previous_word, num_syllables, rhyme_word):
    """Generates word three for each line of haiku given base_url, previous line, and word to rhyme with.

    Args:
        base_url (str): base url for Datamuse API 
        previous_word (str): previous word to given following context
        rhyme_word (str): word that word three must rhyme with

    Returns:
        list: array of words that follow word two and rhyme with word in argument
    """
    base_url = "https://api.datamuse.com"    
    end_point = f'/words?md=s&sp=*&lc={previous_word}&rel_rhy={rhyme_word}'
    
    muse_api_full_url = f'{base_url}{end_point}'
    print(muse_api_full_url)

    response = requests.get(muse_api_full_url)

    if response:
        data = json.loads(response.text)
        
        words_list = []
        for line in data:
            if line["numSyllables"] == num_syllables and line != '.':
                entry = {
                    "word" : line["word"],
                    "numSyllables" : line["numSyllables"] 
                }
                words_list.append(entry)
        # print(json.dumps(words_list, indent=4))    
        return words_list
    else:
        print("Connection error. Request failed")
        print(f'Status code: {response.status_code}')

def generate_random_word(words_list):
    if not words_list:
        return None
    random_word = random.choice(words_list)
    return random_word["word"]

def print_format_json(json_data):
    print(json.dumps(json_data, indent=4))

def create_haiku(topic):
    
    while True:
        
        word_bank = set()
        
        # Word 0
        word_0_list = generate_init_word(topic, 3)
        # print('word_0_list: ', format_json(word_0_list))
        
        word_0 = generate_random_word(word_0_list)
        word_bank.add(word_0)
    
        # print("word 0", word_0)

        # Word 1
        word_1_list = generate_word_left_context(word_0, 2)
        # print('word_1_list: ', format_json(word_1_list))

        word_1 = generate_random_word(word_1_list)
        # print("word 1", word_1)
        word_bank.add(word_1)

        # Word 2
        word_2_list = generate_word_left_context(word_1, 3)
        # print('word_2_list: ', format_json(word_2_list))

        word_2 = generate_random_word(word_2_list)
        # print("word 2", word_2)
        word_bank.add(word_2)

        # Word 3
        word_3_list = generate_word_left_context(word_2, 2)
        # print('word_3_list: ', format_json(word_3_list))

        word_3 = generate_random_word(word_3_list)
        # print("word 3", word_3)
        word_bank.add(word_3)

        # Word 4
        word_4_list = generate_word_left_context_rhyme(word_3, 2, word_1)
        # print('word_4_list: ', format_json(word_4_list))
        
        word_4 = generate_random_word(word_4_list)
        # print("word 4", word_4)
        word_bank.add(word_4)

        # Word 5
        word_5_list = generate_word_left_context(word_4, 3)
        # print('word_5_list: ', format_json(word_5_list))

        word_5 = generate_random_word(word_5_list)
        # print("word 5", word_5)
        word_bank.add(word_5)

        # Word 6
        word_6_list = generate_word_left_context_rhyme(word_5, 2, word_1)
        # print('word_6_list: ', format_json(word_6_list))

        word_6 = generate_random_word(word_6_list)
        # print("word 6", word_6)
        word_bank.add(word_6)

        
        # Loop until no word has None value
        # if (word_0 == None or 
        #     word_1 == None or 
        #     word_2 == None or
        #     word_3 == None or
        #     word_4 == None or
        #     word_5 == None or
        #     word_6 == None):
        #     continue
        if None in word_bank:
            return 
        haiku = f'{word_0} {word_1}\n{word_2} {word_3} {word_4}\n{word_5} {word_6}'
                
        print(word_bank)
        
        return haiku

def test():
    topic = 'money'
    words = generate_init_word(topic, 3)
    print(json.dumps(words, indent=4))

def main():
    
    while True:
        print('Hello, welcome to the predictive text Haiku generator!')
        topic = input('What would you like to see a Haiku about? ').strip().lower()

        # Testing inputs
        # topics = ["win", 'lose', 'music', 'money', 'basketball', 'confidence', 'friends', 'work', 'tech', 'programming', 'finance', 'men', 'women']
        # topic = random.choices(topics)[0]
        
        # print('Topic:', topic)
        haiku = create_haiku(topic)
        
        print('Haiku:')
        print(haiku)
        
        run_again = input('Would you like to see another Haiku (yes/no)? ').lower()
        
        if run_again == 'yes':
            continue
        else:
            break
    
            

if __name__ == "__main__":
    main()
    # test()
