import heapq
from math import log

def pred(N, char_count, unknown_chars, vocab, bigram_sum, bigram_count, trigram_sum, trigram_count, lambdas, tok_1, tok_2):
    l_1, l_2, l_3 = lambdas[0], lambdas[1], lambdas[2]
    min_heap = []
    heapq.heapify(min_heap)
    for token in vocab:
        if token in unknown_chars:
            token = '<unk>'

        # unigram
        unigram_prob = l_1 * char_count[token]/N

        #bigram
        bi_num, bi_denom = 0, 0
        if tok_2 in bigram_count:
            bi_num = bigram_count[tok_2][token] if token in bigram_count[tok_2] else 0
            bi_denom = bigram_sum[tok_2]
        bigram_prob = l_2 * (bi_num/bi_denom) if bi_num != 0 else 0

        #trigram
        tri_num, tri_denom = 0, 0
        base_str = tok_1 + tok_2
        if base_str in trigram_count:
            tri_num = trigram_count[base_str][token] if token in trigram_count[base_str] else 0
            tri_denom = trigram_sum[base_str]
        trigram_prob = l_3 * (tri_num/tri_denom) if tri_num != 0 else 0

        total_token_prob = unigram_prob + bigram_prob + trigram_prob
        log_token_prob = log(total_token_prob, 2)
        heapq.heappush(min_heap, (-log_token_prob, token))

    # Get best 3
    result = []
    while len(result) != 3:
        elem = heapq.heappop(min_heap)
        if elem[1] != '<unk>' and elem[1] != ' ' :
            result.append(elem[1])
    return ''.join(result)
