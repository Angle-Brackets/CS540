import sys
import math


language_probabilities = {
    "english": 0.6,
    "spanish": 0.4
}

def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X = dict()

    #Some preprocessing for the dictionary to include all of the characters at the get-go
    for i in range(97, 123):
        X[chr(i)] = 0

    with open (filename,encoding='utf-8') as f:
        for line in f:
            for char in line.strip():
                if char.isalpha():
                    X[char.lower()] += 1

    return X

def log_normalize(X_list, lang, english, spanish):
    lang = lang.lower()
    if lang not in language_probabilities:
        print("Invalid Language Specified")

    s = 0
    for i in range(len(X_list)):
        s += (X_list[i] * math.log(english[i] if lang == "english" else spanish[i]))

    return math.log(language_probabilities[lang]) + s

def bayes_identification_eng(X_list, english, spanish):
    norm_english = log_normalize(X_list, "english", english, spanish)
    norm_spanish = log_normalize(X_list, "spanish", english, spanish)

    # To avoid divide by zero errors.
    if norm_spanish - norm_english >= 100:
        return 0
    if norm_spanish - norm_english <= -100:
        return 1

    return 1 / (1 + math.exp(norm_spanish - norm_english))

def main():
    english, spanish = get_parameter_vectors()
    #Q1
    X = shred("letter.txt") # Each index corresponds to the letter of the alphabet
    X_list = list(X.values())

    print("Q1")
    for char, count in X.items():
        print(f"{char.upper()} {count}")

    #Q2
    print("Q2")
    print(f"{((X_list[0]) * math.log(english[0])):.4f}")
    print(f"{((X_list[0]) * math.log(spanish[0])):.4f}")

    #Q3
    print("Q3")
    print("{0:.4f}".format(log_normalize(X_list, "english", english, spanish)))
    print("{0:.4f}".format(log_normalize(X_list, "spanish", english, spanish)))

    #Q4
    print("Q4")
    print(f"{bayes_identification_eng(X_list, english, spanish):.4f}")

if __name__ == "__main__":
    main()

# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!
