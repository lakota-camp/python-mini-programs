import requests
import json
import random

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
        return words_list
    else:
        print("Connection error. Request failed")
        print(f'Status code: {response.status_code}')

def generate_random_word(words_list):
    if not words_list:
        return None
    random_word = random.choice(words_list)
    return random_word["word"]

def create_haiku(topic):
    
    while True:
        
        word_bank = set()
        
        # Word 0
        word_0_list = generate_init_word(topic, 3)
        
        word_0 = generate_random_word(word_0_list)
        word_bank.add(word_0)
    
        # Word 1
        word_1_list = generate_word_left_context(word_0, 2)

        word_1 = generate_random_word(word_1_list)
        word_bank.add(word_1)

        # Word 2
        word_2_list = generate_word_left_context(word_1, 3)

        word_2 = generate_random_word(word_2_list)
        word_bank.add(word_2)

        # Word 3
        word_3_list = generate_word_left_context(word_2, 2)

        word_3 = generate_random_word(word_3_list)
        word_bank.add(word_3)

        # Word 4
        word_4_list = generate_word_left_context_rhyme(word_3, 2, word_1)
        
        word_4 = generate_random_word(word_4_list)
        word_bank.add(word_4)

        # Word 5
        word_5_list = generate_word_left_context(word_4, 3)

        word_5 = generate_random_word(word_5_list)
        word_bank.add(word_5)

        # Word 6
        word_6_list = generate_word_left_context_rhyme(word_5, 2, word_1)

        word_6 = generate_random_word(word_6_list)
        word_bank.add(word_6)
        
        if None in word_bank:
            return 
        haiku = f'{word_0} {word_1}\n{word_2} {word_3} {word_4}\n{word_5} {word_6}'
                
        return haiku

def test():
    topic = 'money'
    words = generate_init_word(topic, 3)
    print(json.dumps(words, indent=4))

def main():
    
    while True:
        print('Hello, welcome to the predictive text Haiku generator!')
        topic = input('What would you like to see a Haiku about? ').strip().lower()

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

