
def prepare_file_for_n_gram(file_name, n, vocabulary=None):
    # file_name - name of file to read
    # n - n gram to create (used for padding <START> and <STOP>)
    with open(file_name, encoding="utf8") as file:
        file_lines = file.readlines()

    # Get counts of tokens
    token_counts = dict()
    for i in range(len(file_lines)):
        file_lines[i] = file_lines[i].split()

        for j in range(n - 1):
            file_lines[i].insert(0, "<START>")
        file_lines[i].append("<STOP>")

        for token in file_lines[i]:
            if token not in token_counts:
                token_counts[token] = 0
            token_counts[token] += 1

    for i in range(len(file_lines)):
        for j in range(len(file_lines[i])):
            if (token_counts[file_lines[i][j]] < 3 and vocabulary is None) or (vocabulary is not None and file_lines[i][j] not in vocabulary) or file_lines[i][j] == "<START>":
                file_lines[i][j] = "UNK"

    token_counts = dict()
    total_tokens = 0
    for line in file_lines:
        for i in range(len(line) - n + 1):
            if n == 1:
                ngram = line[i]
            elif n == 2:
                ngram = line[i], line[i + 1]
            else:  # n == 3
                ngram = line[i], line[i + 1], line[i + 2]
            if ngram not in token_counts:
                token_counts[ngram] = 0
            token_counts[ngram] += 1
            total_tokens += 1

    return file_lines, token_counts, total_tokens

def get_unigram_probabilities(file_lines, unigram_token_counts, n):
    unigram_token_probabilities = dict()
    for line in file_lines:
        for token in line:
            unigram_token_probabilities[token] = unigram_token_counts[token] / n
    return unigram_token_probabilities

def get_bigram_probabilities(file_lines, unigram_token_counts, bigram_token_counts, n):
    token_to_bigrams = dict()
    bigram_token_probabilities = dict()
    for line in file_lines:
        for i in range(len(line) - 2 + 1):
            pre = line[i]
            bigram = line[i], line[i + 1]
            bigram_token_probabilities[bigram] = bigram_token_counts[bigram] / unigram_token_counts[pre]
            if pre not in token_to_bigrams:
                token_to_bigrams[pre] = list()
            token_to_bigrams[pre].append(bigram)
    return bigram_token_probabilities, token_to_bigrams

def get_top_three_tokens(vocabulary, bigram_token_probabilities, token_to_bigrams):
    top_three_tokens = dict()

def get_trigram_probabilities(file_lines, bigram_token_counts, trigram_token_counts, n):
    trigram_token_probabilities = dict()
    for line in file_lines:
        for i in range(len(line) - 3 + 1):
            pre = line[i], line[i + 1]
            trigram = line[i], line[i + 1], line[i + 2]
            trigram_token_probabilities[trigram] = trigram_token_counts[trigram] / bigram_token_counts[pre]
    return trigram_token_probabilities

def calculate_unigram_perplexity(file_lines, unigram_token_probabilities, n):
    total_log_likelihood = 0
    for line in file_lines:
        for token in line:
            total_log_likelihood += np.log2(unigram_token_probabilities[token])

    cross_entropy = -total_log_likelihood / n
    perplexity = 2 ** cross_entropy
    return perplexity

def calculate_bigram_perplexity(file_lines, bigram_token_probabilities, n):
    total_log_likelihood = 0
    for line in file_lines:
        for i in range(1, len(line)):
            bigram = line[i - 1], line[i]
            if bigram in bigram_token_probabilities:
                total_log_likelihood += np.log2(bigram_token_probabilities[bigram])
            else:
                with np.errstate(invalid='ignore'):
                    total_log_likelihood += np.log2(0)

    cross_entropy = -total_log_likelihood / n
    perplexity = 2 ** cross_entropy
    return perplexity

def calculate_trigram_perplexity(file_lines, trigram_token_probabilities, n):
    total_log_likelihood = 0
    for line in file_lines:
        #for i in range(len(line) - 3 + 1):
        for i in range(2, len(line)):
            #trigram = line[i], line[i + 1], line[i + 2]
            trigram = line[i - 2], line[i - 1], line[i]
            if trigram in trigram_token_probabilities:
                total_log_likelihood += np.log2(trigram_token_probabilities[trigram])
            else:
                with np.errstate(invalid='ignore'):
                    total_log_likelihood += np.log2(0)

    cross_entropy = -total_log_likelihood / n
    perplexity = 2 ** cross_entropy
    return perplexity

def calculate_interpolated_perplexity(file_lines, unigram_token_probabilities, bigram_token_probabilities, trigram_token_probabilities, l1, l2, l3, n):
    total_log_likelihood = 0
    for line in file_lines:
        for i in range(2, len(line)):
            unigram = line[i]
            bigram = line[i - 1], line[i]
            trigram = line[i - 2], line[i - 1], line[i]

            probability = 0
            if unigram in unigram_token_probabilities:
                probability += l1 * unigram_token_probabilities[unigram]

            if bigram in bigram_token_probabilities:
                probability += l2 * bigram_token_probabilities[bigram]

            if trigram in trigram_token_probabilities:
                probability += l3 * trigram_token_probabilities[trigram]

            total_log_likelihood += np.log2(probability)

    cross_entropy = -total_log_likelihood / n
    perplexity = 2 ** cross_entropy
    return perplexity
