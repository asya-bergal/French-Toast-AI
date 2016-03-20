from gensim.models import word2vec
import logging
import numpy
import random
import os.path

def start_game(model, words):
    valid = False
    word = ""
    while not valid:
        word = input("Choose a word, I'll tell you if it's valid:\n")
        if word in words:
            print("Your word is valid. Let's play!\n")
            valid = True
        else:
            print("Your word is invalid.\n")
    iterate(model, words, "hitler", "toast", word)

def iterate(model, words, word1, word2, correct_answer):
    # if correct_answer in words:
    #     print("Not buggy!\n")
    answer, victory = ask_question(model, word1, word2, correct_answer)
    if answer == correct_answer:
        print("I WON!\n")
        return
    new_words = filter_words(model, words, answer, word1, word2)
    choice_words = new_words[:]
    choice_words.remove(answer)
    new_word = random.choice(choice_words)

    iterate(model, new_words, answer, new_word, correct_answer)

def filter_words(model, words, answer, word1, word2):
    correct = answer
    incorrect = word1 if answer == word2 else word2

    # print("Old word size: {}\n", len(words))
    new_words = [word for word in words if closer_to(model, word, correct, incorrect)]
    # print("New word size: {}\n", len(new_words))

    return new_words

def closer_to(model, word, correct, incorrect):

    dist_correct = numpy.linalg.norm(model[word] - model[correct])
    dist_incorrect = numpy.linalg.norm(model[word] - model[incorrect])

    if dist_incorrect > dist_correct:
        return True
    return False

def ask_question(model, word1, word2, correct_answer):
    # Uncomment this line to enable hints for debugging:
    print("\nHint: {}".format(word1.upper() if closer_to(model, correct_answer, word1, word2) else word2.upper()))

    choice = input("Is it closer to\n(1) {} or\n(2) {}?\n".format(word1.upper(), word2.upper()))
    if choice == "2" or choice.lower() == word2:
        return word2, False
    else:
        return word1, False

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    if not os.path.isfile("text.model.bin"):
        print("Be patient, initializing model...\n")
        if not os.path.isfile("text8"):
            print("Please download the 'text8' corpus and save it to this directory as 'text8'.\n")
            return
        else:
            sentences = word2vec.Text8Corpus('text8')
            model = word2vec.Word2Vec(sentences, size=200)
            model.save_word2vec_format('text.model.bin', binary=True)

    model = word2vec.Word2Vec.load_word2vec_format('text.model.bin', binary=True)
    words = list(model.vocab.keys())

    start_game(model, words)

if __name__ == "__main__":
    main()


# Structure:
# Start with hitler and french toast
# if more similar to x than y, cut off all points closer to x than y, then choose a random point in remaining region
